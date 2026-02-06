"""
Simple CNN feature encoder for drifting loss.
Multi-scale ResNet-style architecture for MNIST and CIFAR-10.

Phase 1: Can skip feature encoder and compute drifting loss in pixel space
Phase 2: Use this encoder trained with MAE objective for better results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import torchvision.models as models


class PretrainedResNetEncoder(nn.Module):
    """
    Feature encoder using pretrained ResNet.
    Returns multi-scale feature MAPS (not pooled vectors) for per-location loss.

    Following paper Section A.5: compute drifting loss at each scale/location.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Extract layers (don't include final fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale feature maps.

        Returns:
            List of feature maps at different scales, each (B, C, H, W)
        """
        # Resize if needed (CIFAR is 32x32)
        if x.shape[-1] < 64:
            x = F.interpolate(x, size=64, mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)

        return [f1, f2, f3, f4]


class BasicBlock(nn.Module):
    """Basic residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.gelu(out)
        return out


class MultiScaleFeatureEncoder(nn.Module):
    """
    Multi-scale CNN feature encoder.

    4-stage architecture with progressive downsampling:
    - Stage 1: 32x32 features
    - Stage 2: 16x16 features
    - Stage 3: 8x8 features
    - Stage 4: 4x4 features

    Features from all stages are pooled and concatenated for multi-scale representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_width: int = 64,
        blocks_per_stage: int = 2,
        feature_dim: int = 512,
        multi_scale: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            base_width: Base number of channels (64 for MNIST, 128 for CIFAR)
            blocks_per_stage: Number of residual blocks per stage
            feature_dim: Output feature dimension
            multi_scale: Whether to use multi-scale features
        """
        super().__init__()
        self.multi_scale = multi_scale

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, base_width), base_width),
            nn.GELU(),
        )

        # Stage 1: 32x32 -> 32x32
        self.stage1 = self._make_stage(base_width, base_width, blocks_per_stage, stride=1)

        # Stage 2: 32x32 -> 16x16
        self.stage2 = self._make_stage(base_width, base_width * 2, blocks_per_stage, stride=2)

        # Stage 3: 16x16 -> 8x8
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, blocks_per_stage, stride=2)

        # Stage 4: 8x8 -> 4x4
        self.stage4 = self._make_stage(base_width * 4, base_width * 8, blocks_per_stage, stride=2)

        # Feature projection
        if multi_scale:
            total_channels = base_width + base_width * 2 + base_width * 4 + base_width * 8
        else:
            total_channels = base_width * 8

        self.proj = nn.Linear(total_channels, feature_dim)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            Features, shape (B, feature_dim)
        """
        x = self.stem(x)

        # Extract multi-scale features
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        if self.multi_scale:
            # Global average pooling at each scale
            p1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
            p2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
            p3 = F.adaptive_avg_pool2d(f3, 1).flatten(1)
            p4 = F.adaptive_avg_pool2d(f4, 1).flatten(1)

            # Concatenate multi-scale features
            features = torch.cat([p1, p2, p3, p4], dim=1)
        else:
            features = F.adaptive_avg_pool2d(f4, 1).flatten(1)

        # Project to output dimension
        features = self.proj(features)

        return features


class MAEEncoder(nn.Module):
    """
    Masked Autoencoder encoder for self-supervised pre-training.

    Pre-trains the feature encoder using reconstruction of masked patches.
    """

    def __init__(
        self,
        feature_encoder: MultiScaleFeatureEncoder,
        in_channels: int = 3,
        img_size: int = 32,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
    ):
        """
        Args:
            feature_encoder: The feature encoder to pre-train
            in_channels: Number of input channels
            img_size: Input image size
            patch_size: Patch size for masking
            mask_ratio: Ratio of patches to mask
        """
        super().__init__()
        self.encoder = feature_encoder
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(feature_encoder.proj.out_features, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, self.num_patches * patch_size * patch_size * in_channels),
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches."""
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, p, p)
        x = x.reshape(B, h * w, C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image."""
        B, N, D = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        C = self.in_channels
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, h * p, w * p)
        return x

    def random_masking(self, x: torch.Tensor) -> tuple:
        """
        Apply random masking to patches.

        Returns:
            x_masked: Masked image
            mask: Binary mask (1 = masked, 0 = visible)
            ids_restore: Indices to restore original order
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        N = h * w

        # Number of patches to mask
        num_mask = int(N * self.mask_ratio)

        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N - num_mask] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Mask the image (set masked patches to 0)
        mask_2d = mask.reshape(B, h, w, 1, 1).expand(-1, -1, -1, p, p)
        mask_2d = mask_2d.permute(0, 3, 1, 4, 2).reshape(B, 1, H, W)
        x_masked = x * (1 - mask_2d)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for MAE pre-training.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            loss: Reconstruction loss on masked patches
            pred: Reconstructed image
            mask: Binary mask
        """
        # Apply masking
        x_masked, mask, ids_restore = self.random_masking(x)

        # Encode
        features = self.encoder(x_masked)

        # Decode
        pred_patches = self.decoder(features)
        pred_patches = pred_patches.reshape(x.shape[0], self.num_patches, -1)

        # Get target patches
        target_patches = self.patchify(x)

        # Loss on masked patches only
        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()

        # Reconstruct image for visualization
        pred = self.unpatchify(pred_patches)

        return loss, pred, mask


def create_feature_encoder(
    dataset: str = "cifar10",
    feature_dim: int = 512,
    multi_scale: bool = True,
    use_pretrained: bool = True,
):
    """
    Create a feature encoder for the specified dataset.

    Args:
        dataset: "mnist" or "cifar10"
        feature_dim: Output feature dimension (ignored for pretrained)
        multi_scale: Whether to use multi-scale features
        use_pretrained: Whether to use ImageNet-pretrained ResNet (for CIFAR)

    Returns:
        Feature encoder
    """
    if dataset.lower() == "mnist":
        return MultiScaleFeatureEncoder(
            in_channels=1,
            base_width=64,
            blocks_per_stage=2,
            feature_dim=feature_dim,
            multi_scale=multi_scale,
        )
    elif dataset.lower() in ["cifar10", "cifar"]:
        if use_pretrained:
            # Use ImageNet-pretrained ResNet - returns multi-scale feature maps
            return PretrainedResNetEncoder(pretrained=True)
        else:
            return MultiScaleFeatureEncoder(
                in_channels=3,
                base_width=128,
                blocks_per_stage=2,
                feature_dim=feature_dim,
                multi_scale=multi_scale,
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def pretrain_mae(
    feature_encoder: MultiScaleFeatureEncoder,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> MultiScaleFeatureEncoder:
    """
    Pre-train feature encoder using MAE objective.

    Args:
        feature_encoder: The encoder to pre-train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Pre-trained feature encoder
    """
    in_channels = feature_encoder.stem[0].in_channels

    mae = MAEEncoder(
        feature_encoder,
        in_channels=in_channels,
        img_size=32,
        patch_size=4,
        mask_ratio=0.75,
    ).to(device)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    mae.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            loss, _, _ = mae(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"MAE Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}")

    # Return the pre-trained encoder (without decoder)
    return mae.encoder
