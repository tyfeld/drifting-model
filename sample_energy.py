"""
sample_energy.py — Energy-Guided Sampling for Drifting Models
==============================================================
Extended sampling script that adds --energy_fn, --guidance_scale, and
--n_steps arguments on top of the original sample.py interface.

Example usage:
    python sample_energy.py \\
        --checkpoint outputs/mnist/checkpoint_final.pt \\
        --dataset mnist \\
        --energy_fn color \\
        --guidance_scale 0.3 \\
        --grad_clip 0.5 \\
        --n_steps 1 \\
        --output_dir samples_guided

    python sample_energy.py \\
        --checkpoint outputs/cifar10/checkpoint_final.pt \\
        --dataset cifar10 \\
        --energy_fn composite \\
        --guidance_scale 0.5 \\
        --sweep_guidance \\
        --output_dir samples_sweep
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import DriftDiT_models
from utils import load_checkpoint, save_image_grid, set_seed
from energy_guidance import (
    ColorEnergyFn,
    GrayscaleEnergyFn,
    EdgeEnergyFn,
    FrequencyEnergyFn,
    CompositeEnergyFn,
    EnergyGuidedSampler,
    compute_guidance_metrics,
    run_guidance_sweep,
)


ENERGY_FNS = {
    "color":     lambda: ColorEnergyFn(),
    "grayscale": lambda: GrayscaleEnergyFn(),
    "edge":      lambda: EdgeEnergyFn(negative_mode=True),
    "smooth":    lambda: EdgeEnergyFn(negative_mode=False),
    "freq":      lambda: FrequencyEnergyFn(penalise_high=True),
    "composite": lambda: CompositeEnergyFn([
        (ColorEnergyFn(), 1.0),
        (EdgeEnergyFn(negative_mode=True), 0.5),
    ]),
}


def load_model(checkpoint_path, dataset, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    if dataset.lower() == "mnist":
        model_name = config.get("model", "DriftDiT-Tiny")
        in_channels, img_size, num_classes = 1, 32, 10
    else:
        model_name = config.get("model", "DriftDiT-Small")
        in_channels, img_size, num_classes = 3, 32, 10

    model = DriftDiT_models[model_name](
        img_size=img_size, in_channels=in_channels, num_classes=num_classes
    ).to(device)

    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()
    return model, in_channels, img_size, num_classes


def generate_comparison_grid(
    sampler: EnergyGuidedSampler,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 8,
    alpha: float = 1.5,
):
    """
    Generate a side-by-side comparison grid:
      Top half:    raw (unguided) samples
      Bottom half: energy-guided samples
    """
    raw_rows, guided_rows = [], []

    for c in range(num_classes):
        z = torch.randn(samples_per_class, in_channels, img_size, img_size, device=device)
        labels = torch.full((samples_per_class,), c, device=device, dtype=torch.long)

        x_guided, x_raw = sampler.sample(
            z, labels, alpha=alpha, return_intermediates=True
        )
        raw_rows.append(x_raw)
        guided_rows.append(x_guided)

    raw = torch.cat(raw_rows, dim=0).clamp(-1, 1)
    guided = torch.cat(guided_rows, dim=0).clamp(-1, 1)

    # Interleave: for each class row, show raw then guided
    interleaved = []
    n = samples_per_class
    for i in range(num_classes):
        interleaved.append(raw[i*n:(i+1)*n])
        interleaved.append(guided[i*n:(i+1)*n])

    return torch.cat(interleaved, dim=0), raw, guided


def _to_display_image(x: torch.Tensor) -> np.ndarray:
    """Convert one CHW image in [-1, 1] to a displayable numpy image."""
    x = ((x.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
    if x.shape[0] == 1:
        return x.squeeze(0).numpy()
    return x.permute(1, 2, 0).numpy()


def save_sweep_image_grid(
    rows: list,
    path: Path,
):
    """Save a labelled raw/guided sweep grid for visual inspection."""
    n_rows = len(rows)
    n_cols = rows[0][1].shape[0]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.35 * n_cols, 1.35 * n_rows),
        squeeze=False,
    )

    for row_idx, (label, images) in enumerate(rows):
        for col_idx in range(n_cols):
            image = _to_display_image(images[col_idx])
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=38)

    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sweep_metric_plot(results: dict, path: Path):
    """Save metric curves for a guidance-scale sweep."""
    scales = list(results.keys())
    energy_raw = [results[s]["energy_raw"] for s in scales]
    energy_guided = [results[s]["energy_guided"] for s in scales]
    reduction = [100.0 * results[s]["energy_reduction"] for s in scales]
    psnr = [results[s]["psnr_guided_vs_raw"] for s in scales]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    axes[0].plot(scales, energy_raw, marker="o", label="raw")
    axes[0].plot(scales, energy_guided, marker="o", label="guided")
    axes[0].set_title("Energy")
    axes[0].set_xlabel("guidance scale")
    axes[0].legend()

    axes[1].plot(scales, reduction, marker="o", color="tab:green")
    axes[1].set_title("Energy reduction")
    axes[1].set_xlabel("guidance scale")
    axes[1].set_ylabel("%")

    axes[2].plot(scales, psnr, marker="o", color="tab:red")
    axes[2].set_title("PSNR vs raw")
    axes[2].set_xlabel("guidance scale")
    axes[2].set_ylabel("dB")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sweep_visualizations(
    model,
    energy_fn,
    z: torch.Tensor,
    labels: torch.Tensor,
    guidance_scales: list,
    results: dict,
    output_dir: Path,
    energy_name: str,
    alpha: float,
    grad_clip: float,
    n_steps: int,
    step_decay: float,
    normalize_grad: bool,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    max_images: int = 8,
):
    """Save image and metric visualizations for a guidance sweep."""
    n_show = min(max_images, z.shape[0])
    z_show = z[:n_show]
    labels_show = labels[:n_show]
    rng_devices = []
    if z.is_cuda:
        rng_devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()]

    rows = []
    for idx, gs in enumerate(guidance_scales):
        sampler = EnergyGuidedSampler(
            model=model,
            energy_fn=energy_fn,
            guidance_scale=gs,
            grad_clip=grad_clip,
            n_steps=n_steps,
            step_decay=step_decay,
            normalize_grad=normalize_grad,
        )
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=z.device)
            guided, raw = sampler.sample(
                z_show, labels_show, alpha=alpha, return_intermediates=True
            )
        if idx == 0:
            rows.append(("raw", raw.clamp(-1, 1)))
        reduction = 100.0 * results[gs]["energy_reduction"]
        rows.append((f"gs={gs:g}\n-{reduction:.1f}%", guided.clamp(-1, 1)))

    grid_path = output_dir / f"sweep_grid_{energy_name}.png"
    metric_path = output_dir / f"sweep_metrics_{energy_name}.png"
    save_sweep_image_grid(rows, grid_path)
    save_sweep_metric_plot(results, metric_path)
    return grid_path, metric_path


def main():
    parser = argparse.ArgumentParser(description="Energy-Guided Sampling for Drifting Models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./samples_energy")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument(
        "--energy_fn", type=str, default="color",
        choices=list(ENERGY_FNS.keys()),
        help="Energy function to use for guidance",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=0.3,
        help="Step size for gradient descent on energy (larger = stronger guidance)",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.5,
        help="Clip value for the normalized energy-gradient direction",
    )
    parser.add_argument(
        "--n_steps", type=int, default=1,
        help="Number of gradient steps at inference time",
    )
    parser.add_argument(
        "--step_decay", type=float, default=0.7,
        help="Decay factor for guidance_scale across steps",
    )
    parser.add_argument(
        "--no_normalize_grad",
        action="store_false",
        dest="normalize_grad",
        help="Use raw energy gradients with per-sample L2 clipping",
    )
    parser.add_argument("--alpha", type=float, default=1.5, help="CFG scale")
    parser.add_argument("--samples_per_class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep_guidance", action="store_true",
        help="Sweep over guidance scales to find the optimal value",
    )
    parser.add_argument(
        "--sweep_scales", type=float, nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        help="Guidance scales to sweep over",
    )
    parser.add_argument(
        "--sweep_vis_samples",
        type=int,
        default=8,
        help="Number of samples to show in the sweep visualization grid",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, in_channels, img_size, num_classes = load_model(args.checkpoint, args.dataset, device)

    # Build energy function
    energy_fn = ENERGY_FNS[args.energy_fn]().to(device)
    print(f"Energy function: {args.energy_fn}")

    # Build sampler
    sampler = EnergyGuidedSampler(
        model=model,
        energy_fn=energy_fn,
        guidance_scale=args.guidance_scale,
        grad_clip=args.grad_clip,
        n_steps=args.n_steps,
        step_decay=args.step_decay,
        normalize_grad=args.normalize_grad,
    )

    # -----------------------------------------------------------------------
    # Guidance scale sweep
    # -----------------------------------------------------------------------
    if args.sweep_guidance:
        print("\n--- Guidance Scale Sweep ---")
        z = torch.randn(num_classes * 4, in_channels, img_size, img_size, device=device)
        labels = torch.arange(num_classes, device=device).repeat_interleave(4)
        sweep_cpu_rng_state = torch.get_rng_state()
        sweep_cuda_rng_state = torch.cuda.get_rng_state(z.device) if z.is_cuda else None

        results = run_guidance_sweep(
            model, energy_fn, z, labels,
            guidance_scales=args.sweep_scales,
            alpha=args.alpha,
            grad_clip=args.grad_clip,
            n_steps=args.n_steps,
            step_decay=args.step_decay,
            normalize_grad=args.normalize_grad,
            cpu_rng_state=sweep_cpu_rng_state,
            cuda_rng_state=sweep_cuda_rng_state,
        )

        # Save metric summary
        with open(output_dir / "sweep_results.txt", "w") as f:
            f.write("guidance_scale,energy_raw,energy_guided,energy_reduction,psnr\n")
            for gs, m in results.items():
                f.write(
                    f"{gs},{m['energy_raw']:.4f},{m['energy_guided']:.4f},"
                    f"{m['energy_reduction']:.4f},{m['psnr_guided_vs_raw']:.2f}\n"
                )
        print(f"Sweep results saved to {output_dir / 'sweep_results.txt'}")
        grid_path, metric_path = save_sweep_visualizations(
            model=model,
            energy_fn=energy_fn,
            z=z,
            labels=labels,
            guidance_scales=args.sweep_scales,
            results=results,
            output_dir=output_dir,
            energy_name=args.energy_fn,
            alpha=args.alpha,
            grad_clip=args.grad_clip,
            n_steps=args.n_steps,
            step_decay=args.step_decay,
            normalize_grad=args.normalize_grad,
            cpu_rng_state=sweep_cpu_rng_state,
            cuda_rng_state=sweep_cuda_rng_state,
            max_images=args.sweep_vis_samples,
        )
        print(f"Sweep image grid saved to {grid_path}")
        print(f"Sweep metric plot saved to {metric_path}")
        return

    # -----------------------------------------------------------------------
    # Comparison grid generation
    # -----------------------------------------------------------------------
    print(f"\nGenerating comparison grid ({num_classes} classes × {args.samples_per_class} samples)...")
    grid, raw, guided = generate_comparison_grid(
        sampler, in_channels, img_size, num_classes, device,
        samples_per_class=args.samples_per_class, alpha=args.alpha,
    )

    # Save interleaved grid (raw rows then guided rows alternating)
    grid_path = output_dir / f"comparison_{args.energy_fn}_gs{args.guidance_scale}.png"
    save_image_grid(grid, str(grid_path), nrow=args.samples_per_class)
    print(f"Comparison grid saved to {grid_path}")

    # Save raw and guided separately
    save_image_grid(raw,    str(output_dir / "raw_unguided.png"),           nrow=args.samples_per_class)
    save_image_grid(guided, str(output_dir / f"guided_{args.energy_fn}.png"), nrow=args.samples_per_class)

    # Compute and report guidance metrics
    metrics = compute_guidance_metrics(raw, guided, energy_fn)
    print("\n--- Guidance Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"Energy fn:      {args.energy_fn}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Grad clip:      {args.grad_clip}\n")
        f.write(f"N steps:        {args.n_steps}\n")
        f.write(f"Normalize grad: {args.normalize_grad}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"Metrics saved to {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
