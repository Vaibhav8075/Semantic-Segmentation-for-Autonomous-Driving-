"""
losses.py — Loss Functions for Semantic Segmentation
=====================================================

Implements specialized loss functions that address the severe class imbalance
inherent in autonomous driving datasets.

Available losses:
1. CrossEntropyLoss — Standard pixel-wise classification loss
2. DiceLoss — Region-based loss robust to class imbalance
3. FocalLoss — Down-weights easy examples, focuses on hard ones
4. CombinedLoss — Weighted sum of CE + Dice + Focal

WHY CLASS IMBALANCE IS A MAJOR ISSUE IN ROAD DATASETS:
=======================================================
In typical driving scenes, the pixel distribution is extremely skewed:
  - Road + Background can occupy 70-85% of all pixels
  - Lane markings may be only 0.5-2% of pixels (thin lines)
  - Traffic signs may be < 1% (small objects)
  - Pedestrians may be 1-3% (distant, small)

This imbalance causes naive cross-entropy to:
  1. Bias predictions toward majority classes (road, background)
  2. Ignore rare but safety-critical classes (pedestrians, signs)
  3. Produce poor boundary delineation

Our loss strategy addresses this:
  - Dice Loss: Treats each class equally regardless of pixel count
  - Focal Loss: Reduces contribution of easy/majority-class pixels
  - Combined Loss: Leverages all three for robust training

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LossConfig, NUM_CLASSES


# ============================================================================
# CROSS ENTROPY LOSS
# ============================================================================

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss for pixel-wise classification.

    Computes the negative log-likelihood of the correct class for each pixel.
    Supports optional class weights to address imbalance, and label smoothing
    to prevent overconfident predictions.

    Loss = -1/N * Σ w_c * log(p_c)

    where w_c are class weights and p_c is the predicted probability for
    the true class.

    Args:
        weight: Optional class weights tensor of shape (num_classes,).
        ignore_index: Label index to ignore (e.g., 255 for unlabeled pixels).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model logits of shape (B, C, H, W).
            targets: Ground truth labels of shape (B, H, W), dtype=long.

        Returns:
            Scalar loss value.
        """
        return self.ce(predictions, targets)


# ============================================================================
# DICE LOSS
# ============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for semantic segmentation.

    Based on the Dice coefficient (F1 score), this loss directly optimizes
    the overlap between predicted and ground truth segmentation masks.

    Key advantage: Treats each class equally regardless of pixel count.
    A class with 1% of pixels contributes the same as one with 50%.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Dice Loss = 1 - Dice

    Uses soft (differentiable) formulation with softmax probabilities.

    Args:
        smooth: Smoothing constant to avoid division by zero.
        ignore_index: Label index to ignore.
        reduction: 'mean' or 'sum' over classes.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model logits (B, C, H, W).
            targets: Ground truth (B, H, W), dtype=long.

        Returns:
            Scalar Dice loss.
        """
        num_classes = predictions.shape[1]

        # Softmax to get probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)

        # Create valid mask (ignore unlabeled pixels)
        valid_mask = (targets != self.ignore_index)  # (B, H, W)

        # Clamp targets to valid range
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_one_hot = F.one_hot(
            targets_clamped, num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute Dice per class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score  # (C,)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection / segmentation.
    (Lin et al., "Focal Loss for Dense Object Detection", 2017)

    Down-weights the loss contribution from easy, well-classified pixels
    and focuses training on hard, misclassified examples. Particularly
    effective for extreme class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    When gamma > 0:
    - Easy examples (p_t > 0.5): loss is significantly reduced
    - Hard examples (p_t < 0.5): loss remains similar to CE
    - gamma = 0: equivalent to standard cross-entropy
    - gamma = 2: recommended value (reduces easy example loss by 100×)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
        alpha: Optional class-specific weighting factor.
        ignore_index: Label index to ignore.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model logits (B, C, H, W).
            targets: Ground truth (B, H, W), dtype=long.

        Returns:
            Scalar focal loss.
        """
        num_classes = predictions.shape[1]

        # Create valid mask
        valid_mask = (targets != self.ignore_index)

        # Clamp targets
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0

        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(predictions, dim=1)  # (B, C, H, W)
        probs = torch.exp(log_probs)

        # Gather the log-probability for the true class
        targets_expanded = targets_clamped.unsqueeze(1)  # (B, 1, H, W)
        log_pt = log_probs.gather(1, targets_expanded).squeeze(1)  # (B, H, W)
        pt = probs.gather(1, targets_expanded).squeeze(1)          # (B, H, W)

        # Compute focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        # Compute loss
        loss = -focal_weight * log_pt  # (B, H, W)

        # Apply valid mask
        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            return loss.sum() / valid_mask.float().sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ============================================================================
# COMBINED LOSS
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted sum of Cross-Entropy + Dice + Focal Loss.

    Total Loss = w_ce * CE + w_dice * Dice + w_focal * Focal

    This combination leverages the strengths of each loss:
    - CE: Strong gradient signal for overall pixel classification
    - Dice: Class-balanced optimization, good for small objects
    - Focal: Focuses on hard-to-classify pixels

    The result is robust training that handles class imbalance while
    maintaining sharp boundaries and accurate predictions for rare classes.

    Args:
        ce_weight: Weight for Cross-Entropy loss.
        dice_weight: Weight for Dice loss.
        focal_weight: Weight for Focal loss.
        focal_gamma: Focal loss gamma parameter.
        class_weights: Optional per-class weights for CE.
        ignore_index: Label index to ignore.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce_loss = CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
        )
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Returns individual losses as well for logging.
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        total = (
            self.ce_weight * ce +
            self.dice_weight * dice +
            self.focal_weight * focal
        )

        return total

    def forward_with_components(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss and return individual components for logging.

        Returns:
            Dictionary with 'total', 'ce', 'dice', 'focal' loss values.
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        total = (
            self.ce_weight * ce +
            self.dice_weight * dice +
            self.focal_weight * focal
        )

        return {
            "total": total,
            "ce": ce.item(),
            "dice": dice.item(),
            "focal": focal.item(),
        }


# ============================================================================
# LOSS FACTORY
# ============================================================================

def get_loss_function(config: LossConfig) -> nn.Module:
    """
    Create a loss function based on configuration.

    Args:
        config: Loss configuration.

    Returns:
        Loss function module.
    """
    # Prepare class weights if specified
    class_weights = None
    if config.class_weights is not None:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32)

    loss_type = config.loss_type.lower()

    if loss_type == "ce":
        loss_fn = CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )

    elif loss_type == "dice":
        loss_fn = DiceLoss()

    elif loss_type == "focal":
        loss_fn = FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
        )

    elif loss_type == "combined":
        loss_fn = CombinedLoss(
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )

    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Supported: ce, dice, focal, combined"
        )

    logging.info(f"Loss function: {loss_type}")
    return loss_fn


# ============================================================================
# ENTRY POINT: Test all loss functions
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Testing Loss Functions")
    print("=" * 60)

    # Create dummy predictions and targets
    B, C, H, W = 2, NUM_CLASSES, 64, 64
    predictions = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    losses = {
        "CrossEntropy": CrossEntropyLoss(),
        "Dice": DiceLoss(),
        "Focal (γ=2)": FocalLoss(gamma=2.0),
        "Combined": CombinedLoss(),
    }

    for name, loss_fn in losses.items():
        loss_val = loss_fn(predictions, targets)
        print(f"  {name:20s}: {loss_val.item():.4f}")

    # Test combined loss with components
    combined = CombinedLoss()
    components = combined.forward_with_components(predictions, targets)
    print(f"\n  Combined Loss Breakdown:")
    for key, val in components.items():
        val_display = val.item() if torch.is_tensor(val) else val
        print(f"    {key:8s}: {val_display:.4f}")

    print("\n  [OK] All loss functions working correctly!")
    print("=" * 60)
