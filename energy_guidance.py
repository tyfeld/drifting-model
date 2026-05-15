"""
energy_guidance.py — Energy-Guided Test-Time Intervention for 1-NFE Drifting Models
=====================================================================================
This module implements training-free, gradient-based guidance at inference time
for the Drifting Model (Deng et al., 2026).

Core idea:
  After one-step generation x = x0 + vθ(x0), we define an external energy
  function E(x) and inject the gradient −α∇_x E(x) to steer the sample
  toward user-defined visual conditions WITHOUT any retraining.

Usage:
    from energy_guidance import EnergyGuidedSampler, ColorEnergyFn, EdgeEnergyFn

    sampler = EnergyGuidedSampler(model, energy_fn=ColorEnergyFn(target_hue=0.6),
                                   guidance_scale=0.3, grad_clip=0.5)
    images = sampler.sample(noise, labels, alpha=1.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Energy function base class
# ---------------------------------------------------------------------------

class EnergyFn(nn.Module):
    """Abstract base class for energy functions.

    All energy functions should return a SCALAR (or per-sample mean) that
    is differentiable w.r.t. the input image x (shape B,C,H,W in [-1,1]).
    Lower energy = more desired.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return one energy value per sample, shape (B,)."""
        return self.forward(x).expand(x.shape[0])


# ---------------------------------------------------------------------------
# Concrete energy functions
# ---------------------------------------------------------------------------

class ColorEnergyFn(EnergyFn):
    """
    Color-shift energy: penalises distance from a target mean colour.

    E(x) = || mean_pixels(x) - target_color ||^2

    Args:
        target_color: Tensor of shape (C,) with target mean pixel values in [-1,1].
                      Defaults to a warm orange tint for RGB (red-heavy).
        channel_weights: Optional per-channel weighting.
    """
    def __init__(
        self,
        target_color: Optional[torch.Tensor] = None,
        channel_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if target_color is None:
            # Default: push images toward a warm orange tone
            target_color = torch.tensor([0.4, 0.0, -0.4])   # (R high, G neutral, B low)
        self.register_buffer("target_color", target_color)

        if channel_weights is None:
            channel_weights = torch.ones_like(target_color)
        self.register_buffer("channel_weights", channel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  values in [-1, 1]
        return self.per_sample(x).mean()

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        mean_color = x.mean(dim=[2, 3])              # (B, C)
        target = self.target_color.to(device=x.device, dtype=x.dtype)      # (C,)
        weights = self.channel_weights.to(device=x.device, dtype=x.dtype)
        diff = (mean_color - target) * weights
        return (diff ** 2).sum(dim=1)


class GrayscaleEnergyFn(EnergyFn):
    """
    Grayscale energy: penalises deviation from grayscale (|R-G|+|G-B|+|B-R|).
    Encourages the model to produce desaturated images.
    Only meaningful for RGB (3-channel) images.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.per_sample(x).mean()

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        energy = (r - g).abs() + (g - b).abs() + (b - r).abs()
        return energy.flatten(1).mean(dim=1)


class EdgeEnergyFn(EnergyFn):
    """
    Edge-sharpness energy: maximises high-frequency content via Laplacian magnitude.
    Minimising E means LESS edge energy, so use negative_mode=True to sharpen.

    E(x) = -mean( |Laplacian(x)| )        when negative_mode=True  (sharpening)
    E(x) =  mean( |Laplacian(x)| )        when negative_mode=False (smoothing)
    """
    def __init__(self, negative_mode: bool = True):
        super().__init__()
        self.negative_mode = negative_mode
        # Laplacian kernel
        kernel = torch.tensor([[0, 1, 0],
                                [1,-4, 1],
                                [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel.to(device=x.device, dtype=x.dtype).expand(C, 1, 3, 3)
        return F.conv2d(x, k, padding=1, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.per_sample(x).mean()

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        lap = self._laplacian(x)
        edge_energy = lap.abs().flatten(1).mean(dim=1)
        return -edge_energy if self.negative_mode else edge_energy


class FrequencyEnergyFn(EnergyFn):
    """
    Frequency energy: penalises high-frequency components via FFT.
    Minimising E encourages smoother images (low-pass behaviour).

    Args:
        high_freq_threshold: Fraction of frequency bins considered 'high'.
                             E.g. 0.5 = upper half of spectrum.
        penalise_high: If True, penalise high-freq (smoothing); else penalise low-freq.
    """
    def __init__(self, high_freq_threshold: float = 0.5, penalise_high: bool = True):
        super().__init__()
        self.high_freq_threshold = high_freq_threshold
        self.penalise_high = penalise_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.per_sample(x).mean()

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 2D FFT
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft, dim=[-2, -1])
        magnitude = fft_shift.abs()

        # Build frequency mask
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=x.device).float() - cy
        xs = torch.arange(W, device=x.device).float() - cx
        dist = torch.sqrt(ys[:, None]**2 + xs[None, :]**2)
        max_dist = math.sqrt(cy**2 + cx**2) + 1e-8

        threshold = self.high_freq_threshold * max_dist
        if self.penalise_high:
            mask = (dist > threshold).float()
        else:
            mask = (dist <= threshold).float()

        return (magnitude * mask).flatten(1).mean(dim=1)


class CompositeEnergyFn(EnergyFn):
    """
    Weighted sum of multiple energy functions.

    Example:
        energy = CompositeEnergyFn([
            (ColorEnergyFn(), 1.0),
            (EdgeEnergyFn(negative_mode=True), 0.5),
        ])
    """
    def __init__(self, fns_and_weights):
        super().__init__()
        self.fns = nn.ModuleList([fn for fn, _ in fns_and_weights])
        self.weights = [w for _, w in fns_and_weights]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.per_sample(x).mean()

    def per_sample(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for fn, w in zip(self.fns, self.weights):
            total = total + w * fn.per_sample(x)
        return total


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

class EnergyGuidedSampler:
    """
    Wraps a trained DriftDiT model and applies energy-guided intervention
    at inference time using a single forward + gradient step.

    The sampling procedure is:
        1. x_raw = model(z, labels, alpha)          [standard 1-NFE sample]
        2. x_guided = x_raw - guidance_scale * clip(∇_x E(x_raw))

    The clipping prevents large gradient steps that push samples off-manifold.

    Args:
        model:           Trained DriftDiT model (in eval mode).
        energy_fn:       EnergyFn instance defining the guidance objective.
        guidance_scale:  Step size α for gradient descent on E.
        grad_clip:       Clip value for the normalized gradient direction.
        n_steps:         Number of gradient steps to apply (default 1).
                         More steps = stronger guidance but risk of artifacts.
        step_decay:      Decay factor for guidance_scale across steps.
        normalize_grad:  If True, normalize each sample's gradient to unit RMS
                         before clipping, so guidance_scale is image-scale.
    """

    def __init__(
        self,
        model: nn.Module,
        energy_fn: EnergyFn,
        guidance_scale: float = 0.3,
        grad_clip: float = 0.5,
        n_steps: int = 1,
        step_decay: float = 0.7,
        normalize_grad: bool = True,
    ):
        self.model = model
        self.energy_fn = energy_fn
        self.guidance_scale = guidance_scale
        self.grad_clip = grad_clip
        self.n_steps = n_steps
        self.step_decay = step_decay
        self.normalize_grad = normalize_grad

    @torch.no_grad()
    def _generate_raw(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.5,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """Standard 1-NFE forward pass (no gradient tracking)."""
        self.model.eval()
        if use_cfg:
            x = self.model.forward_with_cfg(z, labels, alpha=alpha)
        else:
            alpha_tensor = torch.full((z.shape[0],), alpha, device=z.device)
            x = self.model(z, labels, alpha_tensor)
        return x.clamp(-1, 1)

    def _compute_energy_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇_x E(x) with gradient clipping for manifold stability.
        x is detached from the original graph; we create a fresh leaf for autograd.
        """
        x_leaf = x.detach().requires_grad_(True)
        # Sum per-sample energies so the guidance strength does not shrink when
        # batch size changes. Plain forward() returns a batch mean for reporting.
        energy = self.energy_fn.per_sample(x_leaf).sum()
        if not energy.requires_grad:
            return torch.zeros_like(x)
        grad = torch.autograd.grad(energy, x_leaf, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(x)
        grad = grad.detach()

        if self.normalize_grad:
            # Pixel-space energies such as mean colour have gradients scaled by
            # 1 / (H * W). Normalize the direction so guidance_scale controls
            # the actual image-space intervention size.
            grad_rms = grad.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt()
            grad = grad / grad_rms.clamp(min=1e-8)
            if self.grad_clip is not None:
                grad = grad.clamp(min=-self.grad_clip, max=self.grad_clip)
        elif self.grad_clip is not None:
            # Backward-compatible L2 clipping for users who disable
            # normalization and want raw energy-gradient steps.
            grad_norm = grad.norm(p=2, dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(self.grad_clip / grad_norm, max=1.0)
            grad = grad * scale

        return grad

    def sample(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.5,
        use_cfg: bool = True,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate energy-guided samples.

        Args:
            z:      Gaussian noise, shape (B, C, H, W).
            labels: Class labels, shape (B,).
            alpha:  CFG guidance scale for the base model.
            use_cfg: Whether to use classifier-free guidance in the base model.
            return_intermediates: If True, also return the raw (unguided) sample.

        Returns:
            x_guided: Guided images in [-1, 1], shape (B, C, H, W).
            x_raw (optional): Unguided images, returned if return_intermediates=True.
        """
        # Step 1: Standard 1-NFE generation
        x_raw = self._generate_raw(z, labels, alpha=alpha, use_cfg=use_cfg)

        # Step 2: Iterative gradient descent on energy
        x = x_raw.clone()
        step_size = self.guidance_scale
        for step_idx in range(self.n_steps):
            grad = self._compute_energy_gradient(x)
            x = x - step_size * grad
            x = x.clamp(-1, 1)   # keep in valid image range
            step_size = step_size * self.step_decay

        if return_intermediates:
            return x, x_raw
        return x

    def energy_value(self, x: torch.Tensor) -> float:
        """Convenience: compute the scalar energy of a batch (no grad)."""
        with torch.no_grad():
            return self.energy_fn(x).item()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_guidance_metrics(
    x_raw: torch.Tensor,
    x_guided: torch.Tensor,
    energy_fn: EnergyFn,
) -> dict:
    """
    Compute metrics to quantify the effect of guidance.

    Returns:
        dict with:
          - energy_raw:    E(x_raw)
          - energy_guided: E(x_guided)
          - energy_reduction: (energy_raw - energy_guided) / energy_raw
          - lpips_proxy:   mean pixel-wise L1 change (lightweight LPIPS proxy)
          - psnr:          PSNR between raw and guided images (image fidelity)
    """
    with torch.no_grad():
        e_raw = energy_fn(x_raw).item()
        e_guided = energy_fn(x_guided).item()
        reduction = (e_raw - e_guided) / (abs(e_raw) + 1e-8)

        l1_change = (x_guided - x_raw).abs().mean().item()

        # PSNR (raw vs guided as fidelity metric)
        mse = F.mse_loss(x_guided, x_raw).item()
        psnr = 10 * math.log10(4.0 / (mse + 1e-8))   # images in [-1,1], max range = 2

    return {
        "energy_raw": e_raw,
        "energy_guided": e_guided,
        "energy_reduction": reduction,
        "l1_change": l1_change,
        "psnr_guided_vs_raw": psnr,
    }


def run_guidance_sweep(
    model: nn.Module,
    energy_fn: EnergyFn,
    z: torch.Tensor,
    labels: torch.Tensor,
    guidance_scales: list,
    alpha: float = 1.5,
    grad_clip: Optional[float] = 0.5,
    n_steps: int = 1,
    step_decay: float = 0.7,
    normalize_grad: bool = True,
    cpu_rng_state: Optional[torch.Tensor] = None,
    cuda_rng_state: Optional[torch.Tensor] = None,
) -> dict:
    """
    Sweep over guidance_scale values and return metrics for each.
    Useful for finding the optimal scaling without off-manifold collapse.

    Returns:
        dict mapping guidance_scale -> metrics dict
    """
    results = {}
    if cpu_rng_state is None:
        cpu_rng_state = torch.get_rng_state()
    if cuda_rng_state is None and z.is_cuda:
        cuda_rng_state = torch.cuda.get_rng_state(z.device)
    rng_devices = []
    if z.is_cuda:
        rng_devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()]

    for gs in guidance_scales:
        sampler = EnergyGuidedSampler(
            model,
            energy_fn,
            guidance_scale=gs,
            grad_clip=grad_clip,
            n_steps=n_steps,
            step_decay=step_decay,
            normalize_grad=normalize_grad,
        )

        # Keep stochastic style embeddings identical across scales. Otherwise
        # E_raw changes per row and the sweep mixes scale effects with sampling
        # noise from the model's random StyleEmbedder.
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=z.device)
            x_guided, x_raw = sampler.sample(
                z, labels, alpha=alpha, return_intermediates=True
            )

        metrics = compute_guidance_metrics(x_raw, x_guided, energy_fn)
        results[gs] = metrics
        print(
            f"guidance_scale={gs:.3f} | "
            f"E_raw={metrics['energy_raw']:.4f} | "
            f"E_guided={metrics['energy_guided']:.4f} | "
            f"reduction={metrics['energy_reduction']*100:.1f}% | "
            f"PSNR={metrics['psnr_guided_vs_raw']:.1f}dB"
        )
    return results


# ---------------------------------------------------------------------------
# Quick demo (runs standalone with random tensors, no model required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Energy Guidance Module — Self-Test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 4, 3, 32, 32

    # Fake images
    x_fake = torch.randn(B, C, H, W, device=device).clamp(-1, 1)

    # Test each energy function
    fns = {
        "ColorEnergyFn": ColorEnergyFn(),
        "GrayscaleEnergyFn": GrayscaleEnergyFn(),
        "EdgeEnergyFn (sharpen)": EdgeEnergyFn(negative_mode=True),
        "EdgeEnergyFn (smooth)": EdgeEnergyFn(negative_mode=False),
        "FrequencyEnergyFn (low-pass)": FrequencyEnergyFn(penalise_high=True),
        "CompositeEnergyFn": CompositeEnergyFn([
            (ColorEnergyFn(), 1.0),
            (EdgeEnergyFn(negative_mode=True), 0.5),
        ]),
    }

    for name, fn in fns.items():
        fn = fn.to(device)
        x_leaf = x_fake.detach().requires_grad_(True)
        e = fn(x_leaf)
        e.backward()
        grad = x_leaf.grad
        print(f"  {name}: E={e.item():.4f}, grad_norm={grad.norm():.4f}  ✓")

    print("\nAll energy functions passed gradient check.")
    print("\nTo use with a trained model:")
    print("  from energy_guidance import EnergyGuidedSampler, ColorEnergyFn")
    print("  sampler = EnergyGuidedSampler(model, ColorEnergyFn(), guidance_scale=0.3)")
    print("  images  = sampler.sample(noise, labels, alpha=1.5)")
