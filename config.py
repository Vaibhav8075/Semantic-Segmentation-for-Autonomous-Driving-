"""
config.py — Central Configuration for Semantic Segmentation Project
===================================================================

This module defines all configurable parameters for the semantic segmentation
pipeline using Python dataclasses. Configuration can be overridden via CLI
arguments, making it easy to experiment with different hyperparameters.

It is also the single source of truth for the class scheme: the class names,
colors, and the Cityscapes label-ID → class mapping all live here.

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

# 6-class semantic segmentation label scheme for autonomous driving.
# (An earlier version had a "Lane Marking" class, but Cityscapes has no
# lane-marking annotations, so that class could never be learned from it.)
CLASS_NAMES: List[str] = [
    "Road",
    "Vehicle",
    "Pedestrian",
    "Sidewalk",
    "Traffic Sign",
    "Background",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# Label value for pixels excluded from the loss and the metrics
IGNORE_INDEX: int = 255

# Index of the catch-all class (always the last one)
BACKGROUND_CLASS: int = NUM_CLASSES - 1

# RGB color map for visualization (each class → distinct color)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),    # Road — purple
    1: (0, 0, 142),       # Vehicle — dark blue
    2: (220, 20, 60),     # Pedestrian — crimson
    3: (244, 35, 232),    # Sidewalk — pink
    4: (220, 220, 0),     # Traffic Sign — yellow
    5: (70, 70, 70),      # Background — dark gray
}

# Cityscapes label ID (from *_gtFine_labelIds.png) → our class index.
# IDs not listed here fall back to BACKGROUND_CLASS.
CITYSCAPES_LABEL_MAP: Dict[int, int] = {
    # Ignored: pixels that carry no scene information
    0: IGNORE_INDEX,    # unlabeled
    1: IGNORE_INDEX,    # ego vehicle
    2: IGNORE_INDEX,    # rectification border
    3: IGNORE_INDEX,    # out of roi

    # Road (class 0)
    7: 0,               # road

    # Vehicle (class 1)
    26: 1,              # car
    27: 1,              # truck
    28: 1,              # bus
    29: 1,              # caravan
    30: 1,              # trailer
    31: 1,              # train
    32: 1,              # motorcycle
    33: 1,              # bicycle

    # Pedestrian (class 2)
    24: 2,              # person
    25: 2,              # rider

    # Sidewalk (class 3)
    8: 3,               # sidewalk

    # Traffic Sign (class 4)
    19: 4,              # traffic light
    20: 4,              # traffic sign

    # Background (class 5) — listed for clarity; unknown IDs also map here
    4: 5,               # static
    5: 5,               # dynamic
    6: 5,               # ground
    9: 5,               # parking
    10: 5,              # rail track
    11: 5,              # building
    12: 5,              # wall
    13: 5,              # fence
    14: 5,              # guard rail
    15: 5,              # bridge
    16: 5,              # tunnel
    17: 5,              # pole
    18: 5,              # polegroup
    21: 5,              # vegetation
    22: 5,              # terrain
    23: 5,              # sky
}

# Cities held out of the Cityscapes *train* split to form our validation set.
# The official Cityscapes *val* split becomes our test set, because the
# official test split has no public labels.
CITYSCAPES_VAL_CITIES: Tuple[str, ...] = ("darmstadt", "krefeld", "ulm", "weimar")

# ImageNet normalization constants
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

ARCHITECTURES: Tuple[str, ...] = ("unet", "resnet_unet", "deeplabv3plus", "segformer")


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    # Paths
    dataset_root: str = os.path.join(".", "dataset")
    cityscapes_root: str = os.path.join(".", "cityscapes")

    # Working resolution: every image is resized to this before training,
    # validation and inference. 512×1024 keeps Cityscapes' 1:2 aspect ratio.
    image_height: int = 512
    image_width: int = 1024

    # Random crop taken from the resized image during training only
    crop_height: int = 512
    crop_width: int = 512

    # DataLoader settings
    batch_size: int = 4  # fits an 8 GB GPU at 512x512 crops with mixed precision
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation toggle
    use_augmentation: bool = True

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_height, self.image_width)

    @property
    def crop_size(self) -> Tuple[int, int]:
        return (self.crop_height, self.crop_width)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture: 'unet', 'resnet_unet', 'deeplabv3plus', 'segformer'
    architecture: str = "unet"

    # Number of segmentation classes
    num_classes: int = NUM_CLASSES

    # Pretrained weights for the encoder ('imagenet' or None)
    encoder_name: str = "resnet50"
    encoder_weights: Optional[str] = "imagenet"
    freeze_encoder: bool = False

    # U-Net specific
    unet_init_features: int = 64

    # DeepLabV3+ specific
    output_stride: int = 16
    aspp_dilations: Tuple[int, ...] = (6, 12, 18)

    # SegFormer specific
    segformer_variant: str = "b2"  # b0, b1, b2, b3, b4, b5
    # 'hf' = Hugging Face implementation (can load pretrained MiT weights),
    # 'native' = from-scratch implementation in model.py,
    # 'auto' = 'hf' if the transformers package is installed, else 'native'
    segformer_impl: str = "auto"


@dataclass
class TrainConfig:
    """Training pipeline configuration."""

    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD

    # Scheduler: linear warmup followed by the chosen schedule
    scheduler: str = "cosine"  # 'cosine', 'poly', 'plateau', 'none'
    warmup_epochs: int = 3
    scheduler_patience: int = 5
    min_lr: float = 1e-6

    # Training
    epochs: int = 100
    grad_clip_max_norm: float = 1.0
    use_mixed_precision: bool = True
    deterministic: bool = False

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

    @classmethod
    def for_run(cls, output_dir: str, checkpoint_dir: str, run_name: str) -> "PathConfig":
        """Paths for one training run, so different models don't overwrite each other."""
        return cls(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            eda_dir=os.path.join(output_dir, "eda"),
            training_dir=os.path.join(output_dir, "training", run_name),
            evaluation_dir=os.path.join(output_dir, "evaluation", run_name),
            predictions_dir=os.path.join(output_dir, "predictions"),
            explainability_dir=os.path.join(output_dir, "explainability"),
            tensorboard_dir=os.path.join(output_dir, "training", run_name, "logs"),
            best_model_name=f"{run_name}_best.pth",
            latest_model_name=f"{run_name}_latest.pth",
        )

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
# CLI ARGUMENT PARSING (training)
# ============================================================================

def build_train_parser() -> argparse.ArgumentParser:
    """Command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a semantic segmentation model for autonomous driving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset-root", type=str, default="./dataset",
                            help="Root directory of prepared dataset")
    data_group.add_argument("--use-mock", action="store_true",
                            help="Generate synthetic samples on the fly instead of reading --dataset-root")
    data_group.add_argument("--image-size", type=int, nargs=2, default=[512, 1024],
                            metavar=("H", "W"), help="Working resolution images are resized to")
    data_group.add_argument("--crop-size", type=int, nargs=2, default=[512, 512],
                            metavar=("H", "W"), help="Random training crop taken after resizing")
    data_group.add_argument("--batch-size", type=int, default=4,
                            help="Batch size for training")
    data_group.add_argument("--num-workers", type=int, default=4,
                            help="Number of data loading workers")
    data_group.add_argument("--no-augmentation", action="store_true",
                            help="Disable training augmentations")

    # ---- Model ----
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="unet", choices=ARCHITECTURES,
                             help="Model architecture to use")
    model_group.add_argument("--no-pretrained", action="store_true",
                             help="Train the encoder from scratch instead of from ImageNet weights")
    model_group.add_argument("--freeze-encoder", action="store_true",
                             help="Freeze encoder weights (resnet_unet only)")
    model_group.add_argument("--segformer-variant", type=str, default="b2",
                             choices=["b0", "b1", "b2", "b3", "b4", "b5"])

    # ---- Training ----
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=1e-4,
                             help="Peak learning rate (reached after warmup)")
    train_group.add_argument("--min-lr", type=float, default=1e-6,
                             help="Final learning rate of the cosine/poly schedule")
    train_group.add_argument("--warmup-epochs", type=int, default=3,
                             help="Linear learning-rate warmup epochs")
    train_group.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay for AdamW")
    train_group.add_argument("--scheduler", type=str, default="cosine",
                             choices=["cosine", "poly", "plateau", "none"],
                             help="Learning rate schedule after warmup")
    train_group.add_argument("--no-mixed-precision", action="store_true",
                             help="Disable mixed precision training")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                             help="Gradient clipping max norm (0 disables)")
    train_group.add_argument("--patience", type=int, default=15,
                             help="Early stopping patience")
    train_group.add_argument("--seed", type=int, default=42,
                             help="Random seed")
    train_group.add_argument("--deterministic", action="store_true",
                             help="Force deterministic cuDNN kernels (slower)")
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
    path_group.add_argument("--run-name", type=str, default=None,
                            help="Name used for checkpoint files and output folders "
                                 "(default: the model architecture)")

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Create a Config object from parsed training arguments."""
    run_name = args.run_name or args.model
    return Config(
        data=DataConfig(
            dataset_root=args.dataset_root,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
            crop_height=args.crop_size[0],
            crop_width=args.crop_size[1],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_augmentation=not args.no_augmentation,
        ),
        model=ModelConfig(
            architecture=args.model,
            encoder_weights=None if args.no_pretrained else "imagenet",
            freeze_encoder=args.freeze_encoder,
            segformer_variant=args.segformer_variant,
        ),
        train=TrainConfig(
            learning_rate=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            scheduler=args.scheduler,
            use_mixed_precision=not args.no_mixed_precision,
            deterministic=args.deterministic,
            grad_clip_max_norm=args.grad_clip,
            early_stopping_patience=args.patience,
            seed=args.seed,
            resume_from=args.resume,
        ),
        loss=LossConfig(
            loss_type=args.loss,
            focal_gamma=args.focal_gamma,
        ),
        paths=PathConfig.for_run(args.output_dir, args.checkpoint_dir, run_name),
    )


# ============================================================================
# Convenience: if run directly, print the default config
# ============================================================================

if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("  Semantic Segmentation — Default Configuration")
    print("=" * 60)
    print(f"\n  Classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print(f"\n  Data Config:")
    print(f"    Dataset Root   : {cfg.data.dataset_root}")
    print(f"    Image Size     : {cfg.data.image_size}")
    print(f"    Crop Size      : {cfg.data.crop_size}")
    print(f"    Batch Size     : {cfg.data.batch_size}")
    print(f"    Num Workers    : {cfg.data.num_workers}")
    print(f"    Augmentation   : {cfg.data.use_augmentation}")
    print(f"\n  Model Config:")
    print(f"    Architecture   : {cfg.model.architecture}")
    print(f"    Num Classes    : {cfg.model.num_classes}")
    print(f"    Encoder        : {cfg.model.encoder_name} ({cfg.model.encoder_weights})")
    print(f"\n  Training Config:")
    print(f"    Epochs         : {cfg.train.epochs}")
    print(f"    Learning Rate  : {cfg.train.learning_rate}")
    print(f"    Optimizer      : {cfg.train.optimizer}")
    print(f"    Scheduler      : {cfg.train.scheduler} (warmup {cfg.train.warmup_epochs})")
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
