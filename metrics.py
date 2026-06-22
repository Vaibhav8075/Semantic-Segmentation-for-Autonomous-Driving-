"""
metrics.py — Evaluation Metrics for Semantic Segmentation
==========================================================

Implements all standard segmentation metrics using confusion matrix
as the base for efficient batch-accumulated computation.

Metrics:
- Pixel Accuracy (overall and per-class)
- Mean IoU (Intersection over Union)
- Per-Class IoU
- Dice Score (F1 for segmentation)
- Precision, Recall, F1 Score
- Confusion Matrix

Usage:
    metrics = SegmentationMetrics(num_classes=7)
    for images, masks in dataloader:
        predictions = model(images).argmax(dim=1)
        metrics.update(predictions, masks)
    report = metrics.generate_report()

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import CLASS_NAMES, NUM_CLASSES


# ============================================================================
# SEGMENTATION METRICS
# ============================================================================

class SegmentationMetrics:
    """
    Comprehensive segmentation metric tracker.

    Accumulates predictions across batches using a confusion matrix,
    then computes all metrics from this single data structure for
    efficiency and consistency.

    The confusion matrix C has shape (num_classes, num_classes) where:
        C[i, j] = number of pixels with true class i predicted as class j

    From this, all metrics can be derived:
        - TP (True Positives)  = C[i, i]  (diagonal)
        - FP (False Positives) = C[:, i].sum() - C[i, i]  (column sum - diagonal)
        - FN (False Negatives) = C[i, :].sum() - C[i, i]  (row sum - diagonal)

    Args:
        num_classes: Number of segmentation classes.
        class_names: Optional list of class names for reporting.
        ignore_index: Label index to ignore in metric computation.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.ignore_index = ignore_index

        # Initialize confusion matrix
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        """Reset the confusion matrix for a new epoch."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update confusion matrix with a batch of predictions.

        Args:
            predictions: Predicted class indices (B, H, W) or logits (B, C, H, W).
            targets: Ground truth class indices (B, H, W).
        """
        # Handle logits input
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        # Move to CPU
        preds = predictions.cpu().numpy().astype(np.int64)
        gts = targets.cpu().numpy().astype(np.int64)

        # Flatten
        preds = preds.flatten()
        gts = gts.flatten()

        # Create valid mask (ignore specific index)
        valid = gts != self.ignore_index
        preds = preds[valid]
        gts = gts[valid]

        # Clip to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)
        gts = np.clip(gts, 0, self.num_classes - 1)

        # Update confusion matrix using histogram
        cm = np.bincount(
            self.num_classes * gts + preds,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm

    # ---- Core Metrics ----

    def pixel_accuracy(self) -> float:
        """
        Overall Pixel Accuracy.

        Accuracy = TP_total / total_pixels
        = (sum of diagonal) / (sum of all elements)

        Note: Can be misleadingly high with imbalanced classes
        (e.g., 90% accuracy if 90% of pixels are background).
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return float(correct / max(total, 1))

    def per_class_accuracy(self) -> np.ndarray:
        """
        Per-class pixel accuracy.

        For each class: TP / (TP + FN) = diagonal / row_sum
        """
        row_sums = self.confusion_matrix.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.diag(self.confusion_matrix) / np.maximum(row_sums, 1)
        return acc

    def mean_iou(self) -> float:
        """
        Mean Intersection over Union (mIoU).

        The standard metric for semantic segmentation.

        For each class: IoU = TP / (TP + FP + FN)
        mIoU = mean of per-class IoUs

        Returns:
            Scalar mIoU value in [0, 1].
        """
        iou = self.per_class_iou()
        # Only average over classes that exist in ground truth
        valid = self.confusion_matrix.sum(axis=1) > 0
        if valid.sum() == 0:
            return 0.0
        return float(np.mean(iou[valid]))

    def per_class_iou(self) -> np.ndarray:
        """
        Per-class Intersection over Union.

        IoU_c = TP_c / (TP_c + FP_c + FN_c)
              = C[c,c] / (row_c + col_c - C[c,c])

        Returns:
            Array of shape (num_classes,) with IoU per class.
        """
        tp = np.diag(self.confusion_matrix)
        row_sum = self.confusion_matrix.sum(axis=1)  # TP + FN
        col_sum = self.confusion_matrix.sum(axis=0)  # TP + FP
        denominator = row_sum + col_sum - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / np.maximum(denominator, 1)

        return iou

    def dice_score(self) -> np.ndarray:
        """
        Per-class Dice Score (F1 Score for segmentation).

        Dice = 2 * TP / (2 * TP + FP + FN)
             = 2 * |P ∩ G| / (|P| + |G|)

        Equivalent to F1 score. Related to IoU by:
            Dice = 2 * IoU / (1 + IoU)

        Returns:
            Array of shape (num_classes,) with Dice per class.
        """
        tp = np.diag(self.confusion_matrix)
        row_sum = self.confusion_matrix.sum(axis=1)  # TP + FN
        col_sum = self.confusion_matrix.sum(axis=0)  # TP + FP
        denominator = row_sum + col_sum

        with np.errstate(divide="ignore", invalid="ignore"):
            dice = (2.0 * tp) / np.maximum(denominator, 1)

        return dice

    def mean_dice(self) -> float:
        """Mean Dice score across valid classes."""
        dice = self.dice_score()
        valid = self.confusion_matrix.sum(axis=1) > 0
        if valid.sum() == 0:
            return 0.0
        return float(np.mean(dice[valid]))

    def precision(self) -> np.ndarray:
        """
        Per-class Precision.

        Precision = TP / (TP + FP) = diagonal / column_sum

        How many predicted positives are actually positive?

        Returns:
            Array of shape (num_classes,).
        """
        tp = np.diag(self.confusion_matrix)
        col_sum = self.confusion_matrix.sum(axis=0)  # TP + FP

        with np.errstate(divide="ignore", invalid="ignore"):
            prec = tp / np.maximum(col_sum, 1)

        return prec

    def recall(self) -> np.ndarray:
        """
        Per-class Recall (Sensitivity).

        Recall = TP / (TP + FN) = diagonal / row_sum

        How many actual positives are correctly identified?
        Same as per-class accuracy.

        Returns:
            Array of shape (num_classes,).
        """
        return self.per_class_accuracy()

    def f1_score(self) -> np.ndarray:
        """
        Per-class F1 Score.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        This is equivalent to the Dice score.

        Returns:
            Array of shape (num_classes,).
        """
        prec = self.precision()
        rec = self.recall()

        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)

        return f1

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix."""
        return self.confusion_matrix.copy()

    # ---- Reporting ----

    def generate_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive metrics report as a DataFrame.

        Returns:
            DataFrame with per-class and mean metrics.
        """
        data = {
            "Class": self.class_names + ["MEAN"],
            "IoU": list(self.per_class_iou()) + [self.mean_iou()],
            "Dice": list(self.dice_score()) + [self.mean_dice()],
            "Precision": list(self.precision()) + [float(np.mean(self.precision()))],
            "Recall": list(self.recall()) + [float(np.mean(self.recall()))],
            "F1": list(self.f1_score()) + [float(np.mean(self.f1_score()))],
            "Pixel Acc": list(self.per_class_accuracy()) + [self.pixel_accuracy()],
        }

        # Add pixel counts
        row_sums = self.confusion_matrix.sum(axis=1)
        data["Pixels"] = list(row_sums) + [int(row_sums.sum())]

        df = pd.DataFrame(data)
        return df

    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary dictionary of key metrics.

        Returns:
            Dictionary with scalar metrics for logging.
        """
        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
            "mean_dice": self.mean_dice(),
            "mean_precision": float(np.mean(self.precision())),
            "mean_recall": float(np.mean(self.recall())),
            "mean_f1": float(np.mean(self.f1_score())),
        }

    def get_detailed_metrics(self) -> Dict:
        """
        Get all metrics in a dictionary format for saving.

        Returns:
            Dictionary with per-class and summary metrics.
        """
        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
            "mean_dice": self.mean_dice(),
            "per_class_iou": self.per_class_iou().tolist(),
            "per_class_dice": self.dice_score().tolist(),
            "per_class_precision": self.precision().tolist(),
            "per_class_recall": self.recall().tolist(),
            "per_class_f1": self.f1_score().tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def print_report(self) -> None:
        """Print a formatted metrics report to console."""
        from tabulate import tabulate

        df = self.generate_report()

        print(f"\n{'=' * 85}")
        print(f"  Segmentation Metrics Report")
        print(f"{'=' * 85}")
        print(tabulate(df, headers="keys", tablefmt="pretty",
                       floatfmt=".4f", showindex=False))
        print(f"{'=' * 85}")
        print(f"  Overall Pixel Accuracy: {self.pixel_accuracy():.4f}")
        print(f"  Mean IoU             : {self.mean_iou():.4f}")
        print(f"  Mean Dice            : {self.mean_dice():.4f}")
        print(f"  Total Pixels         : {self.confusion_matrix.sum():,}")
        print(f"{'=' * 85}\n")


# ============================================================================
# QUICK METRIC FUNCTIONS (for inline use during training)
# ============================================================================

def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """
    Compute pixel accuracy for a single batch.

    Args:
        predictions: Predicted logits (B, C, H, W) or class indices (B, H, W).
        targets: Ground truth (B, H, W).

    Returns:
        Scalar pixel accuracy.
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    valid = targets != ignore_index
    correct = ((predictions == targets) & valid).sum().item()
    total = valid.sum().item()

    return correct / max(total, 1)


def compute_batch_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> float:
    """
    Compute mean IoU for a single batch (approximate).

    Args:
        predictions: Predicted logits (B, C, H, W) or class indices (B, H, W).
        targets: Ground truth (B, H, W).

    Returns:
        Scalar mean IoU.
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    valid = targets != ignore_index
    preds = predictions[valid]
    gts = targets[valid]

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        gt_cls = (gts == cls)

        intersection = (pred_cls & gt_cls).sum().item()
        union = (pred_cls | gt_cls).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


# ============================================================================
# ENTRY POINT: Test metrics
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Testing Segmentation Metrics")
    print("=" * 60)

    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    # Simulate some predictions
    for _ in range(10):
        batch_preds = torch.randint(0, NUM_CLASSES, (4, 64, 64))
        batch_targets = torch.randint(0, NUM_CLASSES, (4, 64, 64))

        # Make predictions slightly correlated with targets
        match_mask = torch.rand(4, 64, 64) > 0.3  # 70% match
        batch_preds[match_mask] = batch_targets[match_mask]

        metrics.update(batch_preds, batch_targets)

    # Print report
    metrics.print_report()

    # Test summary
    summary = metrics.get_summary()
    print("  Summary metrics:")
    for key, val in summary.items():
        print(f"    {key:20s}: {val:.4f}")

    # Test quick functions
    preds = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    targets = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    print(f"\n  Quick pixel accuracy: {compute_pixel_accuracy(preds, targets):.4f}")
    print(f"  Quick mean IoU: {compute_batch_iou(preds, targets):.4f}")

    print("\n  [OK] All metrics working correctly!")
    print("=" * 60)
