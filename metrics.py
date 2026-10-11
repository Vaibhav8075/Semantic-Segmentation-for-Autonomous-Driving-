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

Averaging rule (applies to every "mean" metric):
    A class is included when it appears in the ground truth OR in the
    predictions. Classes that appear in neither are undefined (NaN) and
    excluded. So a class that is predicted but absent from the ground truth
    counts with IoU 0 — false positives are never hidden.

Usage:
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    for images, masks in dataloader:
        predictions = model(images).argmax(dim=1)
        metrics.update(predictions, masks)
    report = metrics.generate_report()

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that returns NaN where the denominator is 0."""
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    out = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _nanmean(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else 0.0


def _to_jsonable(values: np.ndarray) -> List[Optional[float]]:
    """Convert to a list, with NaN (undefined) as None so the JSON stays valid."""
    return [None if np.isnan(v) else float(v) for v in values]


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
        ignore_index: int = IGNORE_INDEX,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the confusion matrix for a new epoch."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update confusion matrix with a batch of predictions.

        Args:
            predictions: Predicted class indices (B, H, W) or logits (B, C, H, W).
            targets: Ground truth class indices (B, H, W).
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )

        # Count on the device the tensors live on (fast on GPU), then move
        # only the small (C×C) matrix to the CPU.
        preds = predictions.reshape(-1).long()
        gts = targets.reshape(-1).to(preds.device).long()

        valid = (gts != self.ignore_index) & (gts >= 0) & (gts < self.num_classes)
        preds = preds[valid].clamp_(0, self.num_classes - 1)
        gts = gts[valid]

        cm = torch.bincount(
            self.num_classes * gts + preds,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm.cpu().numpy()

    # ---- Core Metrics ----

    @property
    def _tp(self) -> np.ndarray:
        return np.diag(self.confusion_matrix)

    @property
    def _gt_count(self) -> np.ndarray:
        return self.confusion_matrix.sum(axis=1)  # TP + FN

    @property
    def _pred_count(self) -> np.ndarray:
        return self.confusion_matrix.sum(axis=0)  # TP + FP

    def valid_classes(self) -> np.ndarray:
        """Boolean mask of classes present in the ground truth or the predictions."""
        return (self._gt_count + self._pred_count) > 0

    def pixel_accuracy(self) -> float:
        """
        Overall Pixel Accuracy = correct pixels / labeled pixels.

        Note: Can be misleadingly high with imbalanced classes
        (e.g., 90% accuracy if 90% of pixels are background).
        """
        total = self.confusion_matrix.sum()
        return float(self._tp.sum() / total) if total > 0 else 0.0

    def per_class_iou(self) -> np.ndarray:
        """
        Per-class IoU = TP / (TP + FP + FN). NaN for classes that appear in
        neither the ground truth nor the predictions.
        """
        union = self._gt_count + self._pred_count - self._tp
        return _safe_divide(self._tp, union)

    def mean_iou(self) -> float:
        """Mean IoU over classes present in the ground truth or predictions."""
        return _nanmean(self.per_class_iou())

    def dice_score(self) -> np.ndarray:
        """
        Per-class Dice = 2·TP / (2·TP + FP + FN). Equivalent to F1.
        NaN for classes that appear in neither the ground truth nor the predictions.
        """
        return _safe_divide(2 * self._tp, self._gt_count + self._pred_count)

    def mean_dice(self) -> float:
        """Mean Dice over classes present in the ground truth or predictions."""
        return _nanmean(self.dice_score())

    def precision(self) -> np.ndarray:
        """
        Per-class Precision = TP / (TP + FP).

        A class that exists in the ground truth but is never predicted gets 0
        (it is missed entirely); a class absent from both gets NaN.
        """
        prec = _safe_divide(self._tp, self._pred_count)
        missed = (self._pred_count == 0) & (self._gt_count > 0)
        prec[missed] = 0.0
        return prec

    def recall(self) -> np.ndarray:
        """
        Per-class Recall = TP / (TP + FN).

        A class that is predicted but absent from the ground truth gets 0;
        a class absent from both gets NaN.
        """
        rec = _safe_divide(self._tp, self._gt_count)
        spurious = (self._gt_count == 0) & (self._pred_count > 0)
        rec[spurious] = 0.0
        return rec

    def per_class_accuracy(self) -> np.ndarray:
        """Per-class pixel accuracy (same as recall)."""
        return self.recall()

    def f1_score(self) -> np.ndarray:
        """Per-class F1 = 2·P·R / (P + R), identical to the Dice score."""
        return self.dice_score()

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix."""
        return self.confusion_matrix.copy()

    # ---- Reporting ----

    def generate_report(self) -> pd.DataFrame:
        """
        Comprehensive per-class report. The MEAN row averages over valid
        classes only (see module docstring); undefined entries are NaN.
        """
        iou, dice = self.per_class_iou(), self.dice_score()
        prec, rec = self.precision(), self.recall()

        data = {
            "Class": list(self.class_names) + ["MEAN"],
            "IoU": list(iou) + [_nanmean(iou)],
            "Dice": list(dice) + [_nanmean(dice)],
            "Precision": list(prec) + [_nanmean(prec)],
            "Recall": list(rec) + [_nanmean(rec)],
            "GT Pixels": list(self._gt_count) + [int(self._gt_count.sum())],
            "Pred Pixels": list(self._pred_count) + [int(self._pred_count.sum())],
        }
        return pd.DataFrame(data)

    def get_summary(self) -> Dict[str, float]:
        """Scalar metrics for logging."""
        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
            "mean_dice": self.mean_dice(),
            "mean_precision": _nanmean(self.precision()),
            "mean_recall": _nanmean(self.recall()),
            "mean_f1": _nanmean(self.f1_score()),
        }

    def get_detailed_metrics(self) -> Dict:
        """All metrics in a JSON-serializable dictionary (undefined → None)."""
        return {
            **self.get_summary(),
            "num_valid_classes": int(self.valid_classes().sum()),
            "valid_classes": [
                name for name, ok in zip(self.class_names, self.valid_classes()) if ok
            ],
            "per_class_iou": _to_jsonable(self.per_class_iou()),
            "per_class_dice": _to_jsonable(self.dice_score()),
            "per_class_precision": _to_jsonable(self.precision()),
            "per_class_recall": _to_jsonable(self.recall()),
            "per_class_f1": _to_jsonable(self.f1_score()),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def print_report(self) -> None:
        """Print a formatted metrics report to console."""
        from tabulate import tabulate

        df = self.generate_report()
        num_valid = int(self.valid_classes().sum())

        print(f"\n{'=' * 85}")
        print(f"  Segmentation Metrics Report")
        print(f"{'=' * 85}")
        print(tabulate(df, headers="keys", tablefmt="pretty",
                       floatfmt=".4f", showindex=False, missingval="n/a"))
        print(f"{'=' * 85}")
        print(f"  Overall Pixel Accuracy: {self.pixel_accuracy():.4f}")
        print(f"  Mean IoU             : {self.mean_iou():.4f}  "
              f"(over {num_valid}/{self.num_classes} classes)")
        print(f"  Mean Dice            : {self.mean_dice():.4f}")
        print(f"  Total Pixels         : {self.confusion_matrix.sum():,}")
        print(f"{'=' * 85}\n")


# ============================================================================
# QUICK METRIC FUNCTIONS (for inline use during training)
# ============================================================================

def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
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
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """
    Compute mean IoU for a single batch (same averaging rule as above).

    Args:
        predictions: Predicted logits (B, C, H, W) or class indices (B, H, W).
        targets: Ground truth (B, H, W).

    Returns:
        Scalar mean IoU.
    """
    metrics = SegmentationMetrics(num_classes, ignore_index=ignore_index)
    metrics.update(predictions, targets)
    return metrics.mean_iou()


# ============================================================================
# ENTRY POINT: Test metrics
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Testing Segmentation Metrics")
    print("=" * 60)

    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    for _ in range(10):
        batch_preds = torch.randint(0, NUM_CLASSES, (4, 64, 64))
        batch_targets = torch.randint(0, NUM_CLASSES, (4, 64, 64))
        match_mask = torch.rand(4, 64, 64) > 0.3  # 70% match
        batch_preds[match_mask] = batch_targets[match_mask]
        metrics.update(batch_preds, batch_targets)

    metrics.print_report()

    summary = metrics.get_summary()
    print("  Summary metrics:")
    for key, val in summary.items():
        print(f"    {key:20s}: {val:.4f}")

    preds = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    targets = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    print(f"\n  Quick pixel accuracy: {compute_pixel_accuracy(preds, targets):.4f}")
    print(f"  Quick mean IoU: {compute_batch_iou(preds, targets):.4f}")

    print("\n  [OK] All metrics working correctly!")
    print("=" * 60)
