"""
model_factory.py — Model Factory & Comparison Utility
======================================================

Provides a factory function to instantiate any supported segmentation
model by name, a loader that rebuilds the right model from a checkpoint,
and utilities for comparing model architectures.

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import dataclasses
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import CLASS_NAMES, NUM_CLASSES, ModelConfig
from model import (
    DeepLabV3Plus,
    HFSegFormer,
    ResNetUNet,
    SegFormer,
    UNet,
    transformers_available,
)


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

    For SegFormer, `config.segformer_impl == 'auto'` is resolved in place to
    'hf' or 'native', so the saved checkpoint records which one was used.

    Args:
        config: Model configuration dataclass.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If architecture name is not supported.
    """
    architecture = config.architecture.lower()
    pretrained = config.encoder_weights == "imagenet"

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
            pretrained=pretrained,
            freeze_encoder=config.freeze_encoder,
        )
        logging.info(
            f"Created ResNetUNet (pretrained={pretrained}, "
            f"freeze={config.freeze_encoder})"
        )

    elif architecture == "deeplabv3plus":
        model = DeepLabV3Plus(
            num_classes=config.num_classes,
            pretrained=pretrained,
            output_stride=config.output_stride,
        )
        logging.info(
            f"Created DeepLabV3+ (pretrained={pretrained}, "
            f"output_stride={config.output_stride})"
        )

    elif architecture == "segformer":
        if config.segformer_impl == "auto":
            config.segformer_impl = "hf" if transformers_available() else "native"

        if config.segformer_impl == "hf":
            model = HFSegFormer(
                num_classes=config.num_classes,
                variant=config.segformer_variant,
                pretrained=pretrained,
            )
        elif config.segformer_impl == "native":
            if pretrained:
                logging.warning(
                    "Native SegFormer has no pretrained weights and will train from "
                    "scratch, which needs far more data. Install `transformers` to "
                    "use the pretrained MiT encoder."
                )
            model = SegFormer(
                num_classes=config.num_classes,
                variant=config.segformer_variant,
            )
        else:
            raise ValueError(f"Unknown segformer_impl: '{config.segformer_impl}'")

        logging.info(
            f"Created SegFormer-{config.segformer_variant.upper()} "
            f"({config.segformer_impl}, pretrained={pretrained and config.segformer_impl == 'hf'})"
        )

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Supported: unet, resnet_unet, deeplabv3plus, segformer"
        )

    return model


# ============================================================================
# LOADING FROM A CHECKPOINT
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    architecture: Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Rebuild the model a checkpoint was trained with and load its weights.

    Checkpoints written by train.py store their model configuration, so the
    architecture never has to be passed by hand. `architecture` is only used
    for older checkpoints that lack this metadata; if both are present they
    must agree.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        device: Device to place the model on.
        architecture: Optional expected/fallback architecture name.

    Returns:
        (model in eval mode, metadata dict with keys such as 'epoch',
         'best_metric', 'image_size', 'class_names', 'model_config')
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_cfg = checkpoint.get("model_config")
    if saved_cfg is not None:
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        model_config = ModelConfig(**{k: v for k, v in saved_cfg.items() if k in valid_fields})
        if architecture and architecture != model_config.architecture:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' contains a '{model_config.architecture}' "
                f"model, but '{architecture}' was requested."
            )
    elif architecture:
        logging.warning(
            f"Checkpoint '{checkpoint_path}' has no model metadata (older format); "
            f"assuming architecture '{architecture}'."
        )
        model_config = ModelConfig(architecture=architecture)
    else:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has no model metadata (older format). "
            f"Pass the architecture explicitly (e.g. --model unet)."
        )

    class_names = checkpoint.get("class_names")
    if class_names is not None and list(class_names) != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' was trained for the classes {class_names}, "
            f"but this code uses {CLASS_NAMES}. Retrain the model."
        )

    # Weights come from the checkpoint, so never download pretrained ones here
    model_config.encoder_weights = None
    model_config.freeze_encoder = False
    model = get_model(model_config)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not load '{checkpoint_path}' into a '{model_config.architecture}' model "
            f"with {model_config.num_classes} classes. If this checkpoint predates the "
            f"{NUM_CLASSES}-class scheme, retrain it.\n{e}"
        ) from e

    model.to(device).eval()

    meta = {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "history": checkpoint.get("history", {}),
        "image_size": tuple(checkpoint["image_size"]) if checkpoint.get("image_size") else None,
        "class_names": class_names or CLASS_NAMES,
        "model_config": dataclasses.asdict(model_config),
    }
    logging.info(
        f"Loaded {model_config.architecture} from {checkpoint_path} "
        f"(epoch {meta['epoch']}, best mIoU {meta['best_metric']:.4f})"
    )
    return model, meta


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

    Computes parameter counts and output shapes for all supported models
    (randomly initialized — no weights are downloaded).

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
        "UNet": lambda: UNet(num_classes=num_classes),
        "ResNet50-UNet": lambda: ResNetUNet(num_classes=num_classes, pretrained=False),
        "DeepLabV3+": lambda: DeepLabV3Plus(num_classes=num_classes, pretrained=False),
        "SegFormer-B0": lambda: SegFormer(num_classes=num_classes, variant="b0"),
        "SegFormer-B2": lambda: SegFormer(num_classes=num_classes, variant="b2"),
    }

    print(f"\n{'=' * 75}")
    print(f"  {'Model':<20s} | {'Params (M)':>10s} | {'Output Shape':>20s} | {'Status':>8s}")
    print(f"{'=' * 75}")

    for name, build in models_to_compare.items():
        model = build().to(device)
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
