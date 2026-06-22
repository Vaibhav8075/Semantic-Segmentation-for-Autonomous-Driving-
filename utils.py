"""
utils.py — Visualization & Utility Functions
=============================================

Provides helper functions for:
- Mask colorization and overlay creation
- Training curve plotting
- Prediction visualization
- Confusion matrix plotting
- Metric table export
- Logging setup
- Reproducibility seeding
- Model parameter counting

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap

from config import CLASS_COLORS, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (GPU not available)")
    return device


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configure Python logging with console and optional file output.

    Args:
        log_file: Optional path to log file.
        level: Logging level.
    """
    fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a formatted model parameter summary."""
    params = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f"  {model_name} - Parameter Summary")
    print(f"{'=' * 50}")
    print(f"  Total Parameters     : {params['total']:>12,}")
    print(f"  Trainable Parameters : {params['trainable']:>12,}")
    print(f"  Frozen Parameters    : {params['frozen']:>12,}")
    print(f"  Total (Millions)     : {params['total_millions']:>12.2f} M")
    print(f"{'=' * 50}\n")


# ============================================================================
# MASK & IMAGE UTILITIES
# ============================================================================

def colorize_mask(
    mask: np.ndarray,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Convert a class-index mask to an RGB color image.

    Args:
        mask: 2D array of shape (H, W) with integer class indices.
        color_map: Mapping from class index to (R, G, B) tuple.
                   Defaults to CLASS_COLORS.

    Returns:
        RGB image of shape (H, W, 3) as uint8.
    """
    if color_map is None:
        color_map = CLASS_COLORS

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color

    return color_mask


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Overlay a colored segmentation mask on the original image.

    Args:
        image: Original RGB image (H, W, 3), uint8.
        mask: Class-index mask (H, W).
        alpha: Transparency of overlay (0 = image only, 1 = mask only).
        color_map: Color mapping for classes.

    Returns:
        Blended overlay image (H, W, 3) as uint8.
    """
    colored_mask = colorize_mask(mask, color_map)
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverse ImageNet normalization on a tensor-converted image.

    Args:
        image: Normalized image (H, W, 3) or (3, H, W) as float.
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized image in uint8 format (H, W, 3).
    """
    if image.ndim == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = (image * std + mean) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# ============================================================================
# VISUALIZATION — TRAINING CURVES
# ============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Plot training and validation curves (loss, IoU, accuracy).

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                 'train_iou', 'val_iou', 'train_acc', 'val_acc', 'lr'.
        save_path: Directory to save the plots.
    """
    os.makedirs(save_path, exist_ok=True)

    # Define what to plot
    metric_pairs = [
        ("train_loss", "val_loss", "Loss", "Training & Validation Loss"),
        ("train_iou", "val_iou", "Mean IoU", "Training & Validation IoU"),
        ("train_acc", "val_acc", "Pixel Accuracy", "Training & Validation Accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (train_key, val_key, ylabel, title) in enumerate(metric_pairs):
        ax = axes[idx]

        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], "b-", linewidth=2, label="Train")
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], "r-", linewidth=2, label="Validation")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Learning rate curve (separate)
    if "lr" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(history["lr"]) + 1)
        ax.plot(epochs, history["lr"], "g-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "lr_schedule.png"), dpi=150, bbox_inches="tight")
        plt.close()

    logging.info(f"Training curves saved to {save_path}")


# ============================================================================
# VISUALIZATION — PREDICTIONS
# ============================================================================

def visualize_predictions(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    save_path: str,
    num_samples: int = 4,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> None:
    """
    Visualize predictions in a 4-column grid: Image | GT | Prediction | Overlay.

    Args:
        images: List of original images (H, W, 3).
        gt_masks: List of ground truth masks (H, W).
        pred_masks: List of predicted masks (H, W).
        save_path: Path to save the visualization.
        num_samples: Number of samples to display.
        color_map: Color mapping for classes.
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    column_titles = ["Original Image", "Ground Truth", "Prediction", "Overlay"]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight="bold")

    for i in range(num_samples):
        img = images[i]
        gt = gt_masks[i]
        pred = pred_masks[i]

        # Column 1: Original image
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        # Column 2: Ground truth mask
        gt_colored = colorize_mask(gt, color_map)
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].axis("off")

        # Column 3: Predicted mask
        pred_colored = colorize_mask(pred, color_map)
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].axis("off")

        # Column 4: Overlay
        overlay = create_overlay(img, pred, alpha=0.4, color_map=color_map)
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis("off")

    # Add legend
    legend_elements = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        color = np.array(CLASS_COLORS[class_id]) / 255.0
        legend_elements.append(
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                       markersize=10, label=class_name)
        )

    fig.legend(handles=legend_elements, loc="lower center", ncol=len(CLASS_NAMES),
               fontsize=11, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Prediction visualization saved to {save_path}")


# ============================================================================
# VISUALIZATION — CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a confusion matrix as a seaborn heatmap.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
        class_names: List of class names.
        save_path: Path to save the plot.
        normalize: If True, normalize rows to show percentages.
        title: Plot title.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm.astype(float) / row_sums * 100
        fmt = ".1f"
        label = "Percentage (%)"
    else:
        cm_normalized = cm
        fmt = "d"
        label = "Count"

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": label},
    )

    ax.set_xlabel("Predicted Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix saved to {save_path}")


# ============================================================================
# METRIC TABLE EXPORT
# ============================================================================

def save_metric_tables(
    metrics: Dict[str, Union[float, List[float]]],
    save_dir: str,
    prefix: str = "",
) -> None:
    """
    Save metric results as CSV and print to console.

    Args:
        metrics: Dictionary of metric name → value(s).
        save_dir: Directory to save CSV files.
        prefix: Optional prefix for filenames.
    """
    import pandas as pd
    from tabulate import tabulate

    os.makedirs(save_dir, exist_ok=True)

    # Per-class metrics table
    per_class_data = {}

    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)) and len(value) == NUM_CLASSES:
            per_class_data[key] = value

    if per_class_data:
        df = pd.DataFrame(per_class_data, index=CLASS_NAMES)
        csv_path = os.path.join(save_dir, f"{prefix}per_class_metrics.csv")
        df.to_csv(csv_path, float_format="%.4f")

        print(f"\n{'=' * 70}")
        print(f"  Per-Class Metrics {f'({prefix})' if prefix else ''}")
        print(f"{'=' * 70}")
        print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".4f"))
        print()

    # Scalar metrics
    scalar_data = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            scalar_data[key] = [value]

    if scalar_data:
        df_scalar = pd.DataFrame(scalar_data, index=["Value"])
        csv_path = os.path.join(save_dir, f"{prefix}summary_metrics.csv")
        df_scalar.to_csv(csv_path, float_format="%.4f")

        print(f"\n{'=' * 70}")
        print(f"  Summary Metrics {f'({prefix})' if prefix else ''}")
        print(f"{'=' * 70}")
        print(tabulate(df_scalar, headers="keys", tablefmt="pretty", floatfmt=".4f"))
        print()


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    history: Dict,
    save_path: str,
) -> None:
    """
    Save a training checkpoint with all state needed to resume.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: GradScaler state for mixed precision.
        epoch: Current epoch number.
        best_metric: Best validation metric so far.
        history: Training history dictionary.
        save_path: Path to save checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_metric": best_metric,
        "history": history,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path} (epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        scaler: Optional GradScaler to restore state.
        device: Device to map checkpoint tensors to.

    Returns:
        Dictionary with checkpoint metadata (epoch, best_metric, history).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logging.info(
        f"Checkpoint loaded from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "history": checkpoint.get("history", {}),
    }


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping monitor that tracks a validation metric and triggers
    when improvement stalls beyond a configurable patience window.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'max' for metrics like IoU, 'min' for losses.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current epoch's metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logging.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best: {self.best_value:.4f}"
                )

        return self.should_stop


# ============================================================================
# MISC
# ============================================================================

def create_class_legend_image(
    save_path: str,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Create a standalone color legend image for the segmentation classes.

    Args:
        save_path: Path to save the legend image.
        color_map: Color mapping (defaults to CLASS_COLORS).
        class_names: Class name list (defaults to CLASS_NAMES).
    """
    if color_map is None:
        color_map = CLASS_COLORS
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(8, 1.5))

    for i, (name) in enumerate(class_names):
        color = np.array(color_map[i]) / 255.0
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
        ax.text(i + 0.5, 0.5, name, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white" if np.mean(color) < 0.5 else "black")

    ax.set_xlim(0, len(class_names))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Segmentation Class Legend", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
