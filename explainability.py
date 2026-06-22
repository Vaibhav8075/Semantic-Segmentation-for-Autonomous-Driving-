"""
explainability.py — Grad-CAM Visualization for Semantic Segmentation
=====================================================================

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) adapted
for semantic segmentation models. Shows which regions of the image
influence the model's predictions for specific classes.

Unlike classification Grad-CAM (which targets a single logit), segmentation
Grad-CAM computes gradients w.r.t. the spatial average of a target class's
logit map, revealing regions that contribute most to predicting that class.

Usage:
    python explainability.py --image sample.jpg --model unet \\
        --checkpoint best_model.pth --target-class road

    python explainability.py --image sample.jpg --model unet \\
        --checkpoint best_model.pth --all-classes

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    ModelConfig,
)
from model_factory import get_model
from utils import (
    colorize_mask,
    get_device,
    load_checkpoint,
    setup_logging,
)


# ============================================================================
# GRAD-CAM FOR SEGMENTATION
# ============================================================================

class SegmentationGradCAM:
    """
    Grad-CAM adapted for semantic segmentation models.

    For a target class c, this computes:
    1. Forward pass to get logits
    2. Compute spatial mean of class c's logit map as the scalar target
    3. Backward pass to get gradients w.r.t. a chosen intermediate layer
    4. Weight feature maps by global-average-pooled gradients
    5. ReLU + normalize to get the attention heatmap

    This reveals which spatial regions most strongly activate for
    the target class, helping visualize what the model "looks at."

    Args:
        model: Segmentation model.
        target_layer: Layer to compute Grad-CAM for. If None, auto-detects.
        device: Computation device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.model.eval()

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer

        # Hooks for capturing activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _find_target_layer(self) -> torch.nn.Module:
        """
        Auto-detect a suitable target layer for Grad-CAM.

        Searches for the last convolutional layer in the encoder/bottleneck
        which typically has the richest semantic information.
        """
        last_conv = None

        # Look for named layers that are typically good Grad-CAM targets
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                if "bottleneck" in name or "enc4" in name or "layer4" in name:
                    last_conv = module
                    logging.info(f"Grad-CAM target layer: {name}")
                    return module

        # Fallback: find last Conv2d
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module

        if last_conv is not None:
            return last_conv

        raise ValueError("Could not find a suitable target layer for Grad-CAM")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a specific target class.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W).
            target_class: Class index to generate Grad-CAM for.

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()

        # Forward pass
        image_tensor = image_tensor.to(self.device).requires_grad_(True)
        logits = self.model(image_tensor)  # (1, C, H, W)

        # Target: spatial mean of the target class logits
        target = logits[0, target_class, :, :].mean()

        # Backward pass
        target.backward()

        # Get activations and gradients
        activations = self.activations  # (1, C_feat, H_feat, W_feat)
        gradients = self.gradients      # (1, C_feat, H_feat, W_feat)

        if activations is None or gradients is None:
            logging.warning("No activations/gradients captured. Check target layer.")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

        # Global average pool gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C_feat, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam, size=image_tensor.shape[2:],
            mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def generate_all_classes(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for all classes.

        Args:
            image_tensor: Preprocessed image tensor.

        Returns:
            Dictionary of class_name → heatmap.
        """
        heatmaps = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            heatmaps[cls_name] = self.generate(image_tensor, cls_idx)
        return heatmaps


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    prediction_mask: Optional[np.ndarray] = None,
    class_name: str = "",
    save_path: Optional[str] = None,
    alpha: float = 0.4,
) -> None:
    """
    Visualize Grad-CAM heatmap overlaid on the original image.

    Args:
        image: Original RGB image (H, W, 3).
        heatmap: Grad-CAM heatmap (H, W), values in [0, 1].
        prediction_mask: Optional predicted segmentation mask.
        class_name: Target class name for title.
        save_path: Path to save visualization.
        alpha: Heatmap overlay transparency.
    """
    # Resize heatmap to match image
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Create colored heatmap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    # Plot
    num_cols = 4 if prediction_mask is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM: {class_name}", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    if prediction_mask is not None:
        colored_mask = colorize_mask(prediction_mask)
        axes[3].imshow(colored_mask)
        axes[3].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[3].axis("off")

    plt.suptitle(
        f"Grad-CAM Explainability - {class_name}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Grad-CAM saved: {save_path}")

    plt.close()


def visualize_all_class_gradcams(
    image: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    prediction_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize Grad-CAM heatmaps for all classes in a grid.

    Args:
        image: Original image.
        heatmaps: Dict of class_name → heatmap.
        prediction_mask: Optional prediction mask.
        save_path: Path to save the grid visualization.
    """
    num_classes = len(heatmaps)
    cols = 4
    rows = (num_classes + 2 + cols - 1) // cols  # +2 for original + prediction

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Plot prediction
    if prediction_mask is not None:
        axes[1].imshow(colorize_mask(prediction_mask))
        axes[1].set_title("Prediction", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        start_idx = 2
    else:
        start_idx = 1

    # Plot each class heatmap
    h, w = image.shape[:2]
    for idx, (cls_name, heatmap) in enumerate(heatmaps.items()):
        ax = axes[start_idx + idx]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Overlay on image
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        ax.imshow(overlay)
        ax.set_title(f"{cls_name}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for idx in range(start_idx + num_classes, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Grad-CAM Attention Maps - All Classes",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"All-class Grad-CAM saved: {save_path}")

    plt.close()


# ============================================================================
# PREPROCESSING HELPER
# ============================================================================

def preprocess_image(
    image: np.ndarray,
    size: Tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """Preprocess an image for model input."""
    resized = cv2.resize(image, (size[1], size[0]))
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (normalized - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for segmentation models"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--target-class", type=str, default=None,
                        help="Target class name (e.g., 'road', 'vehicle')")
    parser.add_argument("--all-classes", action="store_true",
                        help="Generate Grad-CAM for all classes")
    parser.add_argument("--output-dir", type=str, default="./outputs/explainability",
                        help="Output directory")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])

    args = parser.parse_args()

    setup_logging()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model_config = ModelConfig(architecture=args.model)
    model = get_model(model_config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = preprocess_image(image, tuple(args.image_size)).to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(tensor)
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

    # Initialize Grad-CAM
    gradcam = SegmentationGradCAM(model, device=device)

    base_name = os.path.splitext(os.path.basename(args.image))[0]

    if args.all_classes:
        # Generate for all classes
        heatmaps = gradcam.generate_all_classes(tensor)
        save_path = os.path.join(args.output_dir, f"{base_name}_gradcam_all.png")
        visualize_all_class_gradcams(image, heatmaps, pred_mask, save_path)

    elif args.target_class:
        # Generate for specific class
        class_lower = args.target_class.lower()
        class_idx = None
        for i, name in enumerate(CLASS_NAMES):
            if name.lower() == class_lower:
                class_idx = i
                break

        if class_idx is None:
            print(f"Unknown class: {args.target_class}")
            print(f"Available: {CLASS_NAMES}")
            return

        heatmap = gradcam.generate(tensor, class_idx)
        save_path = os.path.join(
            args.output_dir,
            f"{base_name}_gradcam_{class_lower.replace(' ', '_')}.png"
        )
        visualize_gradcam(image, heatmap, pred_mask, CLASS_NAMES[class_idx], save_path)

    else:
        # Default: generate for the most frequent predicted class
        unique, counts = np.unique(pred_mask, return_counts=True)
        top_class = unique[np.argmax(counts)]
        heatmap = gradcam.generate(tensor, int(top_class))
        save_path = os.path.join(args.output_dir, f"{base_name}_gradcam.png")
        visualize_gradcam(image, heatmap, pred_mask, CLASS_NAMES[top_class], save_path)

    print(f"\n[OK] Grad-CAM visualization saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
