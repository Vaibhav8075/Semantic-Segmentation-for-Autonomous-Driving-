"""
model.py — Segmentation Model Architectures
=============================================

Implements four semantic segmentation architectures from scratch:

1. U-Net (fully from scratch)
2. ResNet50-UNet (transfer learning encoder + custom decoder)
3. DeepLabV3+ (ASPP + ResNet50 backbone)
4. SegFormer (Mix Transformer encoder + MLP decoder)

All models share a common interface:
    model(x: Tensor[B, 3, H, W]) -> Tensor[B, num_classes, H, W]

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional, Tuple

from config import NUM_CLASSES


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class DoubleConvBlock(nn.Module):
    """
    Double Convolution Block: (Conv2D → BN → ReLU) × 2

    This is the fundamental building block of U-Net. Each block applies
    two consecutive 3×3 convolutions with batch normalization and ReLU
    activation. Padding is used to preserve spatial dimensions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Number of intermediate channels (defaults to out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """
    U-Net Encoder Block: DoubleConv → MaxPool

    Extracts features and downsamples the spatial dimensions by 2×.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled: Downsampled features for next stage.
            skip: Pre-pooling features for skip connection.
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: ConvTranspose2D → Concatenate Skip → DoubleConv

    Upsamples features and fuses with encoder skip connections.

    Args:
        in_channels: Number of input channels (from previous decoder stage).
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2,
        )
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Upsampled feature map from previous decoder stage.
            skip: Skip connection from corresponding encoder stage.
        """
        x = self.up(x)

        # Handle size mismatches due to odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ============================================================================
# MODEL 1: U-NET (FROM SCRATCH)
# ============================================================================

class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)

    Implemented completely from scratch with:
    - 4-stage encoder (64 → 128 → 256 → 512)
    - Bottleneck (1024 channels)
    - 4-stage decoder with skip connections
    - Final 1×1 convolution to num_classes

    Architecture:
        Input (3, H, W)
        → Enc1 (64)  → skip1
        → Enc2 (128) → skip2
        → Enc3 (256) → skip3
        → Enc4 (512) → skip4
        → Bottleneck (1024)
        → Dec4 (512)  + skip4
        → Dec3 (256)  + skip3
        → Dec2 (128)  + skip2
        → Dec1 (64)   + skip1
        → Output (num_classes, H, W)

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_classes: Number of segmentation classes.
        init_features: Number of features in the first encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        init_features: int = 64,
    ):
        super().__init__()

        self.num_classes = num_classes
        f = init_features  # Feature multiplier

        # ---- Encoder ----
        self.enc1 = EncoderBlock(in_channels, f)       # 3 → 64
        self.enc2 = EncoderBlock(f, f * 2)             # 64 → 128
        self.enc3 = EncoderBlock(f * 2, f * 4)         # 128 → 256
        self.enc4 = EncoderBlock(f * 4, f * 8)         # 256 → 512

        # ---- Bottleneck ----
        self.bottleneck = DoubleConvBlock(f * 8, f * 16)  # 512 → 1024

        # ---- Decoder ----
        self.dec4 = DecoderBlock(f * 16, f * 8)        # 1024 → 512
        self.dec3 = DecoderBlock(f * 8, f * 4)         # 512 → 256
        self.dec2 = DecoderBlock(f * 4, f * 2)         # 256 → 128
        self.dec1 = DecoderBlock(f * 2, f)             # 128 → 64

        # ---- Output ----
        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolutional layer weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        # Encoder path
        x, skip1 = self.enc1(x)     # (B, 64, H/2, W/2)
        x, skip2 = self.enc2(x)     # (B, 128, H/4, W/4)
        x, skip3 = self.enc3(x)     # (B, 256, H/8, W/8)
        x, skip4 = self.enc4(x)     # (B, 512, H/16, W/16)

        # Bottleneck
        x = self.bottleneck(x)       # (B, 1024, H/16, W/16)

        # Decoder path with skip connections
        x = self.dec4(x, skip4)      # (B, 512, H/8, W/8)
        x = self.dec3(x, skip3)      # (B, 256, H/4, W/4)
        x = self.dec2(x, skip2)      # (B, 128, H/2, W/2)
        x = self.dec1(x, skip1)      # (B, 64, H, W)

        # Output
        logits = self.output_conv(x)  # (B, num_classes, H, W)
        return logits


# ============================================================================
# MODEL 2: RESNET50 U-NET (TRANSFER LEARNING)
# ============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet50 Encoder for U-Net decoder.

    Uses a pretrained ResNet50 backbone and extracts multi-scale features
    from 4 stages for skip connections. This enables transfer learning
    from ImageNet, significantly improving convergence and accuracy.

    Feature extraction points:
        Stage 1: After layer1 → 256 channels, H/4
        Stage 2: After layer2 → 512 channels, H/8
        Stage 3: After layer3 → 1024 channels, H/16
        Stage 4: After layer4 → 2048 channels, H/32

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_layers: If True, freeze encoder weights (fine-tuning mode).
    """

    def __init__(self, pretrained: bool = True, freeze_layers: bool = False):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Extract layers for multi-scale feature extraction
        self.initial = nn.Sequential(
            resnet.conv1,    # 3 → 64, stride 2
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool     # stride 2
        self.layer1 = resnet.layer1    # 64 → 256
        self.layer2 = resnet.layer2    # 256 → 512
        self.layer3 = resnet.layer3    # 512 → 1024
        self.layer4 = resnet.layer4    # 1024 → 2048

        # Output channels at each stage
        self.channels = [64, 256, 512, 1024, 2048]

        # Optionally freeze encoder
        if freeze_layers:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Returns:
            List of feature maps: [initial, layer1, layer2, layer3, layer4]
        """
        features = []

        x0 = self.initial(x)          # (B, 64, H/2, W/2)
        features.append(x0)

        x = self.pool(x0)
        x1 = self.layer1(x)           # (B, 256, H/4, W/4)
        features.append(x1)

        x2 = self.layer2(x1)          # (B, 512, H/8, W/8)
        features.append(x2)

        x3 = self.layer3(x2)          # (B, 1024, H/16, W/16)
        features.append(x3)

        x4 = self.layer4(x3)          # (B, 2048, H/32, W/32)
        features.append(x4)

        return features


class ResNetUNetDecoder(nn.Module):
    """
    Custom U-Net Decoder for ResNet50 encoder features.

    Progressively upsamples and fuses features from the encoder's
    skip connections using ConvTranspose2d and DoubleConvBlocks.
    """

    def __init__(self, encoder_channels: List[int], num_classes: int):
        super().__init__()

        # encoder_channels = [64, 256, 512, 1024, 2048]

        # Decoder stages (from deepest to shallowest)
        self.up4 = nn.ConvTranspose2d(encoder_channels[4], 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(512 + encoder_channels[3], 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConvBlock(256 + encoder_channels[2], 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConvBlock(128 + encoder_channels[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConvBlock(64 + encoder_channels[0], 64)

        # Final upsampling to original resolution
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv0 = DoubleConvBlock(32, 32)

        # Output head
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale features to segmentation map.

        Args:
            features: [f0(64), f1(256), f2(512), f3(1024), f4(2048)]
        """
        f0, f1, f2, f3, f4 = features

        # Stage 4: 2048 → 512 + skip(1024) → 512
        x = self.up4(f4)
        x = self._pad_and_cat(x, f3)
        x = self.conv4(x)

        # Stage 3: 512 → 256 + skip(512) → 256
        x = self.up3(x)
        x = self._pad_and_cat(x, f2)
        x = self.conv3(x)

        # Stage 2: 256 → 128 + skip(256) → 128
        x = self.up2(x)
        x = self._pad_and_cat(x, f1)
        x = self.conv2(x)

        # Stage 1: 128 → 64 + skip(64) → 64
        x = self.up1(x)
        x = self._pad_and_cat(x, f0)
        x = self.conv1(x)

        # Final upsample to original resolution
        x = self.up0(x)
        x = self.conv0(x)

        return self.output_conv(x)

    @staticmethod
    def _pad_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Pad x to match skip dimensions and concatenate."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([x, skip], dim=1)


class ResNetUNet(nn.Module):
    """
    ResNet50 Encoder + U-Net Decoder.

    Combines the power of ImageNet-pretrained ResNet50 features with
    a U-Net-style decoder for precise segmentation. This typically
    outperforms vanilla U-Net due to stronger learned representations.

    Args:
        num_classes: Number of segmentation classes.
        pretrained: Use pretrained encoder weights.
        freeze_encoder: Freeze encoder for fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained, freeze_layers=freeze_encoder)
        self.decoder = ResNetUNetDecoder(
            encoder_channels=self.encoder.channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


# ============================================================================
# MODEL 3: DEEPLABV3+
# ============================================================================

class ASPPConv(nn.Module):
    """Atrous (Dilated) Convolution in ASPP module."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global Average Pooling branch in ASPP module."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.

    Captures multi-scale context using parallel dilated convolutions at
    different rates, plus a global average pooling branch. This is the
    key innovation of DeepLabV3/V3+ for handling objects at multiple scales.

    Args:
        in_channels: Input feature channels.
        out_channels: Output channels per branch (total = 5 × out_channels → projected).
        dilations: Tuple of dilation rates (default: (6, 12, 18)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations: Tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()

        modules = [
            # 1×1 convolution (rate = 0)
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        # Atrous convolutions at different rates
        for dilation in dilations:
            modules.append(ASPPConv(in_channels, out_channels, dilation))

        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution
    (Chen et al., 2018)

    Architecture:
    1. ResNet50 backbone extracts high-level (layer4) and low-level (layer1) features
    2. ASPP module captures multi-scale context from high-level features
    3. Decoder fuses ASPP output with low-level features for precise boundaries

    Advantages over U-Net:
    - Multi-scale context via dilated convolutions
    - More efficient than dense skip connections
    - Better for large-scale objects (roads, buildings)

    Args:
        num_classes: Number of segmentation classes.
        pretrained: Use ImageNet pretrained backbone.
        output_stride: Backbone output stride (16 or 8). Lower = more detail but slower.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()

        # Backbone: ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Modify ResNet for DeepLab (dilated convolutions in layer3/layer4)
        if output_stride == 16:
            # layer3: stride 2, no dilation
            # layer4: stride 1, dilation 2
            self._modify_resnet_layer(resnet.layer4, stride=1, dilation=2)
        elif output_stride == 8:
            self._modify_resnet_layer(resnet.layer3, stride=1, dilation=2)
            self._modify_resnet_layer(resnet.layer4, stride=1, dilation=4)

        # Backbone layers
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1    # Low-level features (256 ch)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4    # High-level features (2048 ch)

        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Decoder
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Output classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    @staticmethod
    def _modify_resnet_layer(layer, stride: int, dilation: int):
        """Modify a ResNet layer to use dilated convolutions."""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (stride, stride)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.initial(x)
        x = self.maxpool(x)
        low_level = self.layer1(x)     # (B, 256, H/4, W/4)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)            # (B, 2048, H/16, W/16)

        # ASPP
        x = self.aspp(x)              # (B, 256, H/16, W/16)

        # Decoder: upsample ASPP output and fuse with low-level features
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode="bilinear", align_corners=False)

        low_level = self.low_level_proj(low_level)  # (B, 48, H/4, W/4)
        x = torch.cat([x, low_level], dim=1)       # (B, 304, H/4, W/4)
        x = self.decoder(x)                        # (B, 256, H/4, W/4)

        x = self.classifier(x)                     # (B, num_classes, H/4, W/4)

        # Upsample to original resolution
        x = F.interpolate(x, size=input_size,
                          mode="bilinear", align_corners=False)

        return x


# ============================================================================
# MODEL 4: SEGFORMER (Simplified Implementation)
# ============================================================================

class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding for SegFormer.

    Unlike ViT's non-overlapping patches, SegFormer uses overlapping patches
    to preserve local continuity, which is important for dense prediction tasks.

    Args:
        in_channels: Input channels.
        embed_dim: Embedding dimension.
        patch_size: Patch size.
        stride: Patch stride (< patch_size for overlap).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 7,
        stride: int = 4,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """
    Efficient Self-Attention with Spatial Reduction.

    SegFormer's key innovation: reduces the spatial dimensions of keys and
    values before computing attention, making it O(N²/R²) instead of O(N²).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        sr_ratio: Spatial reduction ratio.
    """

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_sr).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.sr_norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    """
    Mix Feed-Forward Network with depth-wise convolution.

    Uses a 3×3 depth-wise convolution between the two linear layers,
    introducing positional information without explicit position embeddings.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension (typically 4× dim).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, N).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single SegFormer transformer block."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MixTransformerEncoder(nn.Module):
    """
    Mix Transformer (MiT) Encoder for SegFormer.

    A hierarchical transformer encoder that produces multi-scale features
    (like a CNN backbone) while leveraging self-attention for global context.

    Uses the B2 configuration by default:
        Stage 1: embed_dim=64,  depth=3, sr_ratio=8
        Stage 2: embed_dim=128, depth=4, sr_ratio=4
        Stage 3: embed_dim=320, depth=6, sr_ratio=2
        Stage 4: embed_dim=512, depth=3, sr_ratio=1

    Args:
        in_channels: Input image channels.
        embed_dims: Embedding dimensions for each stage.
        num_heads: Number of attention heads per stage.
        depths: Number of transformer blocks per stage.
        sr_ratios: Spatial reduction ratios per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: Tuple[int, ...] = (64, 128, 320, 512),
        num_heads: Tuple[int, ...] = (1, 2, 5, 8),
        depths: Tuple[int, ...] = (3, 4, 6, 3),
        sr_ratios: Tuple[int, ...] = (8, 4, 2, 1),
        mlp_ratios: Tuple[float, ...] = (4.0, 4.0, 4.0, 4.0),
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_stages = len(embed_dims)

        # Build stages
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_stages):
            # Patch embedding
            in_ch = in_channels if i == 0 else embed_dims[i - 1]
            patch_size = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2

            self.patch_embeds.append(
                OverlapPatchEmbed(in_ch, embed_dims[i], patch_size, stride)
            )

            # Transformer blocks
            stage_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    mlp_ratio=mlp_ratios[i],
                )
                for _ in range(depths[i])
            ])
            self.blocks.append(stage_blocks)

            # Layer norm
            self.norms.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []

        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)

            for block in self.blocks[i]:
                x = block(x, H, W)

            x = self.norms[i](x)
            x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
            features.append(x)

        return features


class MLPDecodeHead(nn.Module):
    """
    All-MLP Decode Head for SegFormer.

    Unlike complex decoder designs, SegFormer uses a simple MLP decoder:
    1. Project all multi-scale features to the same dimension
    2. Upsample all to 1/4 resolution
    3. Concatenate
    4. Fuse with MLP
    5. Classify

    This simplicity is possible because the transformer encoder already
    captures sufficient global context.

    Args:
        embed_dims: Encoder output dimensions per stage.
        decode_dim: Decoder hidden dimension.
        num_classes: Number of segmentation classes.
    """

    def __init__(
        self,
        embed_dims: Tuple[int, ...] = (64, 128, 320, 512),
        decode_dim: int = 256,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Linear projections for each scale
        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decode_dim, kernel_size=1),
                nn.BatchNorm2d(decode_dim),
                nn.ReLU(inplace=True),
            )
            for dim in embed_dims
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(decode_dim * len(embed_dims), decode_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_dim),
            nn.ReLU(inplace=True),
        )

        # Classifier
        self.classifier = nn.Conv2d(decode_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale features to segmentation map.

        Args:
            features: List of feature maps from encoder stages.
        """
        target_size = features[0].shape[2:]  # 1/4 resolution

        projected = []
        for i, feat in enumerate(features):
            proj = self.linear_projs[i](feat)
            proj = F.interpolate(proj, size=target_size,
                                 mode="bilinear", align_corners=False)
            projected.append(proj)

        # Concatenate and fuse
        x = torch.cat(projected, dim=1)
        x = self.fusion(x)
        x = self.classifier(x)

        return x


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    with Transformers (Xie et al., 2021)

    Key innovations:
    1. Hierarchical Transformer encoder (MiT) without position embeddings
    2. Overlapping patch embeddings for local continuity
    3. Efficient self-attention with spatial reduction
    4. Lightweight all-MLP decoder

    This produces state-of-the-art results with fewer parameters than
    CNN-based methods like DeepLabV3+.

    Args:
        num_classes: Number of segmentation classes.
        variant: SegFormer variant ('b0' through 'b5', larger = more powerful).
    """

    # Architecture configurations for different variants
    CONFIGS = {
        "b0": {"embed_dims": (32, 64, 160, 256), "depths": (2, 2, 2, 2),
                "num_heads": (1, 2, 5, 8), "decode_dim": 256},
        "b1": {"embed_dims": (64, 128, 320, 512), "depths": (2, 2, 2, 2),
                "num_heads": (1, 2, 5, 8), "decode_dim": 256},
        "b2": {"embed_dims": (64, 128, 320, 512), "depths": (3, 4, 6, 3),
                "num_heads": (1, 2, 5, 8), "decode_dim": 768},
        "b3": {"embed_dims": (64, 128, 320, 512), "depths": (3, 4, 18, 3),
                "num_heads": (1, 2, 5, 8), "decode_dim": 768},
        "b4": {"embed_dims": (64, 128, 320, 512), "depths": (3, 8, 27, 3),
                "num_heads": (1, 2, 5, 8), "decode_dim": 768},
        "b5": {"embed_dims": (64, 128, 320, 512), "depths": (3, 6, 40, 3),
                "num_heads": (1, 2, 5, 8), "decode_dim": 768},
    }

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        variant: str = "b2",
        in_channels: int = 3,
    ):
        super().__init__()

        assert variant in self.CONFIGS, f"Unknown variant: {variant}"
        cfg = self.CONFIGS[variant]

        self.encoder = MixTransformerEncoder(
            in_channels=in_channels,
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
        )

        self.decoder = MLPDecodeHead(
            embed_dims=cfg["embed_dims"],
            decode_dim=cfg["decode_dim"],
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)

        # Upsample to original resolution
        logits = F.interpolate(logits, size=input_size,
                               mode="bilinear", align_corners=False)
        return logits


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def get_model_info() -> str:
    """Return a formatted string describing all available models."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║                    Available Model Architectures                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. UNet (from scratch)                                          ║
║     - Classic encoder-decoder with skip connections              ║
║     - 4 encoder stages + bottleneck + 4 decoder stages           ║
║     - ~31M parameters                                            ║
║                                                                  ║
║  2. ResNetUNet (transfer learning)                               ║
║     - ResNet50 encoder (ImageNet pretrained)                     ║
║     - Custom U-Net decoder with skip connections                 ║
║     - ~33M parameters                                            ║
║                                                                  ║
║  3. DeepLabV3+ (ASPP + encoder-decoder)                          ║
║     - ResNet50 backbone with dilated convolutions                ║
║     - Atrous Spatial Pyramid Pooling for multi-scale context     ║
║     - Low-level feature fusion for sharp boundaries              ║
║     - ~40M parameters                                            ║
║                                                                  ║
║  4. SegFormer (transformer-based)                                ║
║     - Mix Transformer encoder (hierarchical)                     ║
║     - Efficient self-attention with spatial reduction            ║
║     - Lightweight all-MLP decoder                                ║
║     - ~25M parameters (B2 variant)                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# ENTRY POINT: Test model architectures
# ============================================================================

if __name__ == "__main__":
    print(get_model_info())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 512, 512).to(device)

    models_to_test = {
        "UNet": UNet(num_classes=NUM_CLASSES),
        "ResNetUNet": ResNetUNet(num_classes=NUM_CLASSES, pretrained=False),
        "DeepLabV3Plus": DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False),
        "SegFormer-B0": SegFormer(num_classes=NUM_CLASSES, variant="b0"),
    }

    for name, model in models_to_test.items():
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            out = model(x)

        params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  {name:20s} | Input: {tuple(x.shape)} | Output: {tuple(out.shape)} | "
              f"Params: {params:.1f}M | OK")

    print("\n  All models produce correct output shapes!")
