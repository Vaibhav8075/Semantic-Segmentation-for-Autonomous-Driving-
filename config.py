"""
config.py — Central Configuration for Semantic Segmentation Project
===================================================================

This module defines all configurable parameters for the semantic segmentation
pipeline using Python dataclasses. Configuration can be overridden via CLI
arguments, making it easy to experiment with different hyperparameters.

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CLASS DEFINITIONS & COLOR MAPPINGS
# ============================================================================

# 7-class semantic segmentation label scheme for autonomous driving
NUM_CLASSES: int = 7

CLASS_NAMES: List[str] = [
    "Road",
    "Lane Marking",
    "Vehicle",
    "Pedestrian",
    "Sidewalk",
    "Traffic Sign",
    "Background",
]

# RGB color map for visualization (each class → distinct color)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),    # Road — purple
    1: (255, 255, 0),     # Lane Marking — yellow
    2: (0, 0, 142),       # Vehicle — dark blue
    3: (220, 20, 60),     # Pedestrian — crimson
    4: (244, 35, 232),    # Sidewalk — pink
    5: (220, 220, 0),     # Traffic Sign — dark yellow
    6: (70, 70, 70),      # Background — dark gray
}

# Cityscapes label ID → our 7-class mapping
# Cityscapes uses specific label IDs (not sequential 0-33)
CITYSCAPES_LABEL_MAP: Dict[int, int] = {
    # Road (class 0)
    7: 0,     # road

    # Lane Marking (class 1) — derived from road markings
    6: 1,     # ground (includes road markings in many datasets)

    # Vehicle (class 2)
    26: 2,    # car
    27: 2,    # truck
    28: 2,    # bus
    31: 2,    # train
    32: 2,    # motorcycle
    33: 2,    # bicycle

    # Pedestrian (class 3)
    24: 3,    # person
    25: 3,    # rider

    # Sidewalk (class 4)
    8: 4,     # sidewalk

    # Traffic Sign (class 5)
    19: 5,    # traffic light
    20: 5,    # traffic sign

    # Background (class 6) — everything else
    0: 6,     # unlabeled
    1: 6,     # ego vehicle
    2: 6,     # rectification border
    3: 6,     # out of roi
    4: 6,     # static
    5: 6,     # dynamic
    9: 6,     # parking
    10: 6,    # rail track
    11: 6,    # building
    12: 6,    # wall
    13: 6,    # fence
    14: 6,    # guard rail
    15: 6,    # bridge
    16: 6,    # tunnel
    17: 6,    # pole
    18: 6,    # polegroup
    21: 6,    # vegetation
    22: 6,    # terrain
    23: 6,    # sky
    29: 6,    # caravan
    30: 6,    # trailer
    -1: 6,    # license plate
}


# ImageNet normalization constants
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    # Paths
    dataset_root: str = os.path.join(".", "dataset")
    cityscapes_root: str = os.path.join(".", "cityscapes")

    # Image settings
    image_height: int = 512
    image_width: int = 512

    # DataLoader settings
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation toggle
    use_augmentation: bool = True

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_height, self.image_width)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture: 'unet', 'resnet_unet', 'deeplabv3plus', 'segformer'
    architecture: str = "unet"

    # Number of segmentation classes
    num_classes: int = NUM_CLASSES

    # Encoder settings (for transfer learning models)
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    freeze_encoder: bool = False

    # U-Net specific
    unet_init_features: int = 64

    # DeepLabV3+ specific
    output_stride: int = 16
    aspp_dilations: Tuple[int, ...] = (6, 12, 18)

    # SegFormer specific
    segformer_variant: str = "b2"  # b0, b1, b2, b3, b4, b5


@dataclass
class TrainConfig:
    """Training pipeline configuration."""

    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD

    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'plateau', 'step', 'none'
    scheduler_patience: int = 5
    scheduler_t_max: int = 50
    min_lr: float = 1e-6

    # Training
    epochs: int = 100
    grad_clip_max_norm: float = 1.0
    use_mixed_precision: bool = True

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    save_every_n_epochs: int = 5
    resume_from: Optional[str] = None

    # Reproducibility
    seed: int = 42


@dataclass
class LossConfig:
    """Loss function configuration."""

    # Loss type: 'ce', 'dice', 'focal', 'combined'
    loss_type: str = "combined"

    # Focal Loss parameters
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # Combined Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0

    # Class weights for CE (None = uniform)
    class_weights: Optional[List[float]] = None

    # Label smoothing
    label_smoothing: float = 0.0


@dataclass
class PathConfig:
    """Output and checkpoint path configuration."""

    # Base output directory
    output_dir: str = os.path.join(".", "outputs")
    checkpoint_dir: str = os.path.join(".", "checkpoints")

    # Sub-directories (auto-created)
    eda_dir: str = os.path.join(".", "outputs", "eda")
    training_dir: str = os.path.join(".", "outputs", "training")
    evaluation_dir: str = os.path.join(".", "outputs", "evaluation")
    predictions_dir: str = os.path.join(".", "outputs", "predictions")
    explainability_dir: str = os.path.join(".", "outputs", "explainability")
    tensorboard_dir: str = os.path.join(".", "outputs", "training", "logs")

    # Best and latest model filenames
    best_model_name: str = "best_model.pth"
    latest_model_name: str = "latest_model.pth"

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.best_model_name)

    @property
    def latest_model_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.latest_model_name)

    def create_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            self.output_dir,
            self.checkpoint_dir,
            self.eda_dir,
            self.training_dir,
            self.evaluation_dir,
            self.predictions_dir,
            self.explainability_dir,
            self.tensorboard_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self):
        """Create output directories on initialization."""
        self.paths.create_dirs()


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training/evaluation/inference."""
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation for Autonomous Driving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset-root", type=str, default="./dataset",
                            help="Root directory of prepared dataset")
    data_group.add_argument("--cityscapes-root", type=str, default="./cityscapes",
                            help="Root directory of raw Cityscapes dataset")
    data_group.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                            help="Input image size (H W)")
    data_group.add_argument("--batch-size", type=int, default=8,
                            help="Batch size for training")
    data_group.add_argument("--num-workers", type=int, default=4,
                            help="Number of data loading workers")
    data_group.add_argument("--no-augmentation", action="store_true",
                            help="Disable training augmentations")

    # ---- Model ----
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="unet",
                             choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"],
                             help="Model architecture to use")
    model_group.add_argument("--freeze-encoder", action="store_true",
                             help="Freeze encoder weights (transfer learning)")

    # ---- Training ----
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=1e-4,
                             help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay for AdamW")
    train_group.add_argument("--scheduler", type=str, default="cosine",
                             choices=["cosine", "plateau", "step", "none"],
                             help="Learning rate scheduler")
    train_group.add_argument("--no-mixed-precision", action="store_true",
                             help="Disable mixed precision training")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                             help="Gradient clipping max norm")
    train_group.add_argument("--patience", type=int, default=15,
                             help="Early stopping patience")
    train_group.add_argument("--seed", type=int, default=42,
                             help="Random seed")
    train_group.add_argument("--resume", type=str, default=None,
                             help="Path to checkpoint to resume from")

    # ---- Loss ----
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--loss", type=str, default="combined",
                            choices=["ce", "dice", "focal", "combined"],
                            help="Loss function to use")
    loss_group.add_argument("--focal-gamma", type=float, default=2.0,
                            help="Focal loss gamma parameter")

    # ---- Paths ----
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--output-dir", type=str, default="./outputs",
                            help="Output directory")
    path_group.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                            help="Checkpoint directory")

    # ---- Inference ----
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--image", type=str, default=None,
                             help="Path to input image for inference")
    infer_group.add_argument("--image-dir", type=str, default=None,
                             help="Directory of images for batch inference")
    infer_group.add_argument("--checkpoint", type=str, default=None,
                             help="Path to model checkpoint for inference/evaluation")
    infer_group.add_argument("--webcam", action="store_true",
                             help="Run real-time webcam inference")

    # ---- Explainability ----
    explain_group = parser.add_argument_group("Explainability")
    explain_group.add_argument("--target-class", type=str, default=None,
                               help="Target class name for Grad-CAM")

    # ---- Export ----
    export_group = parser.add_argument_group("Export")
    export_group.add_argument("--export-format", type=str, default="onnx",
                              choices=["onnx"],
                              help="Export format")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    """Create a Config object from parsed CLI arguments."""
    config = Config(
        data=DataConfig(
            dataset_root=args.dataset_root,
            cityscapes_root=args.cityscapes_root,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_augmentation=not args.no_augmentation,
        ),
        model=ModelConfig(
            architecture=args.model,
            freeze_encoder=args.freeze_encoder,
        ),
        train=TrainConfig(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            scheduler=args.scheduler,
            use_mixed_precision=not args.no_mixed_precision,
            grad_clip_max_norm=args.grad_clip,
            early_stopping_patience=args.patience,
            seed=args.seed,
            resume_from=args.resume,
        ),
        loss=LossConfig(
            loss_type=args.loss,
            focal_gamma=args.focal_gamma,
        ),
        paths=PathConfig(
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            eda_dir=os.path.join(args.output_dir, "eda"),
            training_dir=os.path.join(args.output_dir, "training"),
            evaluation_dir=os.path.join(args.output_dir, "evaluation"),
            predictions_dir=os.path.join(args.output_dir, "predictions"),
            explainability_dir=os.path.join(args.output_dir, "explainability"),
            tensorboard_dir=os.path.join(args.output_dir, "training", "logs"),
        ),
    )
    return config


def get_config() -> Config:
    """Parse CLI args and return a fully initialized Config."""
    args = parse_args()
    return config_from_args(args)


# ============================================================================
# Convenience: if run directly, print the default config
# ============================================================================

if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("  Semantic Segmentation — Default Configuration")
    print("=" * 60)
    print(f"\n  Data Config:")
    print(f"    Dataset Root   : {cfg.data.dataset_root}")
    print(f"    Image Size     : {cfg.data.image_size}")
    print(f"    Batch Size     : {cfg.data.batch_size}")
    print(f"    Num Workers    : {cfg.data.num_workers}")
    print(f"    Augmentation   : {cfg.data.use_augmentation}")
    print(f"\n  Model Config:")
    print(f"    Architecture   : {cfg.model.architecture}")
    print(f"    Num Classes    : {cfg.model.num_classes}")
    print(f"    Encoder        : {cfg.model.encoder_name}")
    print(f"\n  Training Config:")
    print(f"    Epochs         : {cfg.train.epochs}")
    print(f"    Learning Rate  : {cfg.train.learning_rate}")
    print(f"    Optimizer      : {cfg.train.optimizer}")
    print(f"    Scheduler      : {cfg.train.scheduler}")
    print(f"    Mixed Precision: {cfg.train.use_mixed_precision}")
    print(f"    Early Stop     : {cfg.train.early_stopping_patience}")
    print(f"\n  Loss Config:")
    print(f"    Loss Type      : {cfg.loss.loss_type}")
    print(f"    Focal Gamma    : {cfg.loss.focal_gamma}")
    print(f"\n  Path Config:")
    print(f"    Output Dir     : {cfg.paths.output_dir}")
    print(f"    Checkpoint Dir : {cfg.paths.checkpoint_dir}")
    print(f"    Best Model     : {cfg.paths.best_model_path}")
    print("=" * 60)
