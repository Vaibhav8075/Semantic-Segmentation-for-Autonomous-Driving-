"""
model_factory.py — Model Factory & Comparison Utility
======================================================

Provides a factory function to instantiate any supported segmentation
model by name, and utilities for comparing model architectures.

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from config import ModelConfig, NUM_CLASSES
from model import DeepLabV3Plus, ResNetUNet, SegFormer, UNet


# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_model(config: ModelConfig) -> nn.Module:
    """
    Instantiate a segmentation model based on configuration.

    Supported architectures:
        - 'unet': Standard U-Net (from scratch)
        - 'resnet_unet': ResNet50 encoder + U-Net decoder
        - 'deeplabv3plus': DeepLabV3+ with ASPP
        - 'segformer': SegFormer transformer architecture

    Args:
        config: Model configuration dataclass.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If architecture name is not supported.
    """
    architecture = config.architecture.lower()

    if architecture == "unet":
        model = UNet(
            in_channels=3,
            num_classes=config.num_classes,
            init_features=config.unet_init_features,
        )
        logging.info(f"Created UNet (init_features={config.unet_init_features})")

    elif architecture == "resnet_unet":
        model = ResNetUNet(
            num_classes=config.num_classes,
            pretrained=(config.encoder_weights == "imagenet"),
            freeze_encoder=config.freeze_encoder,
        )
        logging.info(
            f"Created ResNetUNet (pretrained={config.encoder_weights}, "
            f"freeze={config.freeze_encoder})"
        )

    elif architecture == "deeplabv3plus":
        model = DeepLabV3Plus(
            num_classes=config.num_classes,
            pretrained=(config.encoder_weights == "imagenet"),
            output_stride=config.output_stride,
        )
        logging.info(
            f"Created DeepLabV3+ (output_stride={config.output_stride})"
        )

    elif architecture == "segformer":
        model = SegFormer(
            num_classes=config.num_classes,
            variant=config.segformer_variant,
        )
        logging.info(f"Created SegFormer-{config.segformer_variant.upper()}")

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Supported: unet, resnet_unet, deeplabv3plus, segformer"
        )

    return model


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    num_classes: int = NUM_CLASSES,
    input_size: tuple = (2, 3, 512, 512),
    device: Optional[torch.device] = None,
) -> Dict[str, Dict]:
    """
    Compare all available model architectures.

    Computes parameter counts, output shapes, and inference speed
    for all supported models.

    Args:
        num_classes: Number of segmentation classes.
        input_size: Input tensor size (B, C, H, W).
        device: Device to run comparison on.

    Returns:
        Dictionary of model name → comparison metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(*input_size).to(device)
    results = {}

    models_to_compare = {
        "UNet": UNet(num_classes=num_classes),
        "ResNet50-UNet": ResNetUNet(num_classes=num_classes, pretrained=False),
        "DeepLabV3+": DeepLabV3Plus(num_classes=num_classes, pretrained=False),
        "SegFormer-B0": SegFormer(num_classes=num_classes, variant="b0"),
        "SegFormer-B2": SegFormer(num_classes=num_classes, variant="b2"),
    }

    print(f"\n{'=' * 75}")
    print(f"  {'Model':<20s} | {'Params (M)':>10s} | {'Output Shape':>20s} | {'Status':>8s}")
    print(f"{'=' * 75}")

    for name, model in models_to_compare.items():
        model = model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        try:
            with torch.no_grad():
                output = model(x)
            output_shape = tuple(output.shape)
            status = "OK"
        except Exception as e:
            output_shape = f"ERROR: {e}"
            status = "FAIL"

        results[name] = {
            "total_params_M": total_params,
            "trainable_params_M": trainable_params,
            "output_shape": output_shape,
            "status": status,
        }

        print(f"  {name:<20s} | {total_params:>10.1f} | {str(output_shape):>20s} | {status:>8s}")

        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{'=' * 75}\n")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    compare_models()
