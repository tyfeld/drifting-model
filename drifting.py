"""
Drifting field computation (Algorithm 2 from the paper, Sec A.1).
Implements the core V computation for training drifting models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
) -> torch.Tensor:
    """
    Compute the drifting field V (Algorithm 2 from paper, Page 12).

    This is the EXACT implementation from the paper's pseudocode.

    Args:
        x: Generated samples in feature space, shape (N, D)
        y_pos: Positive (real data) samples, shape (N_pos, D)
        y_neg: Negative (generated) samples, shape (N_neg, D)
        temperature: Temperature for softmax (smaller = sharper)
        mask_self: Whether to mask self-distances (when y_neg == x)

    Returns:
        V: Drifting field, shape (N, D)
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    # 1. Compute pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    # 2. Mask self-distances (when y_neg contains x)
    if mask_self and N == N_neg:
        mask = torch.eye(N, device=device) * 1e6
        dist_neg = dist_neg + mask

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 4. Concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Normalize along BOTH dimensions (key insight from paper)
    A_row = torch.softmax(logit, dim=1)   # softmax over y (columns)
    A_col = torch.softmax(logit, dim=0)   # softmax over x (rows)
    A = torch.sqrt(A_row * A_col)         # geometric mean

    # 6. Split back to pos and neg
    A_pos = A[:, :N_pos]  # (N, N_pos)
    A_neg = A[:, N_pos:]  # (N, N_neg)

    # 7. Compute weights (cross-weighting from paper)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)

    # 8. Compute drift
    drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
    drift_neg = torch.mm(W_neg, y_neg)  # (N, D)

    V = drift_pos - drift_neg

    return V


def compute_V_multi_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float] = [0.02, 0.05, 0.2],
    mask_self: bool = True,
    normalize_each: bool = True,
) -> torch.Tensor:
    """
    Compute drifting field with multiple temperatures (Sec A.6).

    Uses multiple temperature scales to capture both local and global structure.
    Each V is normalized before summing.

    Args:
        x: Generated samples, shape (N, D)
        y_pos: Positive samples, shape (N_pos, D)
        y_neg: Negative samples, shape (N_neg, D)
        temperatures: List of temperature values
        mask_self: Whether to mask self-distances
        normalize_each: Whether to normalize each V before summing

    Returns:
        V: Combined drifting field, shape (N, D)
    """
    V_total = torch.zeros_like(x)

    for tau in temperatures:
        V_tau = compute_V(x, y_pos, y_neg, tau, mask_self)

        if normalize_each:
            # Normalize so E[||V||^2] ~ 1
            V_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / V_norm

        V_total = V_total + V_tau

    return V_total


def normalize_features(
    features: torch.Tensor,
    scale: Optional[float] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Normalize features by standardizing to zero mean and unit variance per dimension,
    then scaling so average pairwise distance ~ sqrt(D) (Sec A.6).

    Args:
        features: Feature tensor, shape (N, D)
        scale: If provided, use this scale factor directly (for consistency across batches).
        mean: If provided, use this mean for standardization.
        std: If provided, use this std for standardization.

    Returns:
        Normalized features, scale factor, mean, and std used
    """
    D = features.shape[1]
    target_dist = D ** 0.5  # ~32 for 1024 dims

    # Compute or use provided mean/std
    with torch.no_grad():
        if mean is None:
            mean = features.mean(dim=0, keepdim=True)
        if std is None:
            std = features.std(dim=0, keepdim=True) + 1e-8

    features_std = (features - mean) / std

    if scale is None:
        # Compute scale to achieve target average pairwise distance
        with torch.no_grad():
            # Sample subset for efficiency
            n_sample = min(features.shape[0], 256)
            idx = torch.randperm(features.shape[0], device=features.device)[:n_sample]
            subset = features_std[idx]

            dists = torch.cdist(subset, subset, p=2)
            # Exclude diagonal
            mask = ~torch.eye(n_sample, dtype=torch.bool, device=features.device)
            avg_dist = dists[mask].mean()

            # Scale to target distance
            scale = (target_dist / (avg_dist + 1e-8)).item()

    return features_std * scale, scale, mean, std


def normalize_drift(
    V: torch.Tensor,
    target_variance: float = 1.0,
) -> torch.Tensor:
    """
    Normalize drift field so E[V^2] ~ target_variance (Sec A.6).

    Args:
        V: Drift field, shape (N, D)
        target_variance: Target mean squared value

    Returns:
        Normalized drift field
    """
    # Compute current mean squared value
    current_var = torch.mean(V ** 2)

    # Scale to target
    scale = (target_variance / (current_var + 1e-8)) ** 0.5
    return V * scale


class DriftingLoss(nn.Module):
    """
    Complete drifting loss computation for training.

    Combines:
    - Feature extraction (optional)
    - Multi-temperature V computation
    - Feature and drift normalization
    - MSE loss computation
    """

    def __init__(
        self,
        feature_encoder: Optional[nn.Module] = None,
        temperatures: List[float] = [0.02, 0.05, 0.2],
        normalize_features: bool = True,
        normalize_drift: bool = True,
        use_pixel_space: bool = False,
    ):
        """
        Args:
            feature_encoder: Optional encoder to map images to feature space
            temperatures: List of temperatures for V computation
            normalize_features: Whether to normalize features
            normalize_drift: Whether to normalize drift field
            use_pixel_space: If True, flatten images and use as features
        """
        super().__init__()
        self.feature_encoder = feature_encoder
        self.temperatures = temperatures
        self.do_normalize_features = normalize_features
        self.do_normalize_drift = normalize_drift
        self.use_pixel_space = use_pixel_space

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        if self.use_pixel_space or self.feature_encoder is None:
            # Flatten to feature vectors
            return x.flatten(start_dim=1)
        else:
            return self.feature_encoder(x)

    def forward(
        self,
        x_gen: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute drifting loss.

        Args:
            x_gen: Generated samples, shape (N, C, H, W)
            x_pos: Positive (real) samples, shape (N_pos, C, H, W)
            x_neg: Negative samples, shape (N_neg, C, H, W). If None, uses x_gen.

        Returns:
            loss: Scalar loss value
            info: Dict with additional information
        """
        if x_neg is None:
            x_neg = x_gen

        # Extract features
        feat_gen = self.get_features(x_gen)
        feat_pos = self.get_features(x_pos)
        feat_neg = self.get_features(x_neg)

        # Normalize features
        if self.do_normalize_features:
            feat_gen, scale_gen = normalize_features(feat_gen)
            feat_pos, _ = normalize_features(feat_pos, target_scale=scale_gen * feat_gen.shape[1] ** 0.5)
            feat_neg, _ = normalize_features(feat_neg, target_scale=scale_gen * feat_neg.shape[1] ** 0.5)

        # Compute drifting field
        V = compute_V_multi_temperature(
            feat_gen,
            feat_pos,
            feat_neg,
            self.temperatures,
            mask_self=(x_neg is x_gen),
        )

        # Normalize drift
        if self.do_normalize_drift:
            V = normalize_drift(V)

        # Compute loss: MSE(phi(x), stopgrad(phi(x) + V))
        target = (feat_gen + V).detach()
        loss = torch.mean((feat_gen - target) ** 2)

        # Info for logging
        info = {
            "loss": loss.item(),
            "drift_norm": torch.mean(V ** 2).item() ** 0.5,
            "feat_norm": torch.mean(feat_gen ** 2).item() ** 0.5,
        }

        return loss, info


class ClassConditionalDriftingLoss(nn.Module):
    """
    Class-conditional drifting loss.

    For class-conditional generation, computes drifting loss per class,
    using same-class samples as positives and unconditional negatives for CFG.
    """

    def __init__(
        self,
        feature_encoder: Optional[nn.Module] = None,
        temperatures: List[float] = [0.02, 0.05, 0.2],
        use_pixel_space: bool = False,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.temperatures = temperatures
        self.use_pixel_space = use_pixel_space

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        if self.use_pixel_space or self.feature_encoder is None:
            return x.flatten(start_dim=1)
        else:
            return self.feature_encoder(x)

    def forward(
        self,
        x_gen: torch.Tensor,
        labels_gen: torch.Tensor,
        x_real: torch.Tensor,
        labels_real: torch.Tensor,
        x_uncond_neg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute class-conditional drifting loss.

        Args:
            x_gen: Generated samples, shape (N, C, H, W)
            labels_gen: Labels for generated samples, shape (N,)
            x_real: Real samples, shape (N_real, C, H, W)
            labels_real: Labels for real samples, shape (N_real,)
            x_uncond_neg: Unconditional negatives for CFG, shape (N_uncond, C, H, W)

        Returns:
            loss: Scalar loss value
            info: Dict with additional information
        """
        device = x_gen.device
        num_classes = labels_gen.max().item() + 1

        total_loss = 0.0
        total_drift_norm = 0.0
        count = 0

        for c in range(num_classes):
            # Get samples for this class
            mask_gen = labels_gen == c
            mask_real = labels_real == c

            if not mask_gen.any() or not mask_real.any():
                continue

            x_gen_c = x_gen[mask_gen]
            x_pos_c = x_real[mask_real]

            # Negatives: other generated samples + unconditional negatives
            mask_neg = labels_gen != c
            x_neg_c = x_gen[mask_neg] if mask_neg.any() else x_gen_c

            if x_uncond_neg is not None and len(x_uncond_neg) > 0:
                x_neg_c = torch.cat([x_neg_c, x_uncond_neg], dim=0)

            # Get features
            feat_gen = self.get_features(x_gen_c)
            feat_pos = self.get_features(x_pos_c)
            feat_neg = self.get_features(x_neg_c)

            # Compute V
            V = compute_V_multi_temperature(
                feat_gen,
                feat_pos,
                feat_neg,
                self.temperatures,
                mask_self=False,
            )

            # Normalize drift
            V = normalize_drift(V)

            # Loss
            target = (feat_gen + V).detach()
            loss_c = torch.mean((feat_gen - target) ** 2)

            total_loss = total_loss + loss_c * len(x_gen_c)
            total_drift_norm = total_drift_norm + torch.mean(V ** 2).item() ** 0.5 * len(x_gen_c)
            count += len(x_gen_c)

        if count == 0:
            return torch.tensor(0.0, device=device), {"loss": 0.0, "drift_norm": 0.0}

        loss = total_loss / count

        info = {
            "loss": loss.item(),
            "drift_norm": total_drift_norm / count,
        }

        return loss, info


# Convenience function for toy 2D experiments
def drift_step_2d(
    points: torch.Tensor,
    target_dist: torch.Tensor,
    temperature: float = 0.1,
    step_size: float = 0.1,
) -> torch.Tensor:
    """
    Single drift step for 2D toy experiments (Fig. 3-4 of paper).

    Args:
        points: Current points, shape (N, 2)
        target_dist: Target distribution samples, shape (M, 2)
        temperature: Temperature for V computation
        step_size: Step size for drift

    Returns:
        Updated points after one drift step
    """
    V = compute_V(
        points,
        target_dist,
        points,
        temperature,
        mask_self=True,
    )
    return points + step_size * V
