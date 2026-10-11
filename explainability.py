"""
explainability.py — Grad-CAM Visualization for Semantic Segmentation
=====================================================================

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) adapted
for semantic segmentation models. Shows which regions of the image
influence the model's predictions for specific classes.

Unlike classification Grad-CAM (which targets a single logit), segmentation
Grad-CAM computes gradients w.r.t. the spatial average of a target class's
logit map over the pixels predicted as that class, revealing regions that
contribute most to predicting it.

Usage:
    python explainability.py --image sample.jpg \\
        --checkpoint checkpoints/unet_best.pth --target-class road

    python explainability.py --image sample.jpg \\
        --checkpoint checkpoints/unet_best.pth --all-classes

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

from config import ARCHITECTURES, CLASS_NAMES, NUM_CLASSES
from dataset import preprocess_image
from model import DeepLabV3Plus, HFSegFormer, ResNetUNet, SegFormer, UNet
from model_factory import load_model_from_checkpoint
from utils import colorize_mask, get_device, setup_logging


# ============================================================================
# GRAD-CAM FOR SEGMENTATION
# ============================================================================

def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Pick the Grad-CAM layer for a model: the deepest feature map that still
    has spatial structure and feeds the decoder.
    """
    if isinstance(model, UNet):
        return model.bottleneck
    if isinstance(model, ResNetUNet):
        return model.encoder.layer4
    if isinstance(model, DeepLabV3Plus):
        return model.aspp
    if isinstance(model, SegFormer):
        return model.decoder.fusion
    if isinstance(model, HFSegFormer):
        return model.net.decode_head.linear_fuse

    # Fallback: last convolution that is not the final classifier
    candidates = [
        m for m in model.modules()
        if isinstance(m, torch.nn.Conv2d) and m.out_channels != NUM_CLASSES
    ]
    if not candidates:
        raise ValueError("Could not find a suitable target layer for Grad-CAM")
    return candidates[-1]


class SegmentationGradCAM:
    """
    Grad-CAM adapted for semantic segmentation models (Seg-Grad-CAM).

    For a target class c, this computes:
    1. Forward pass to get logits
    2. Sum class c's logits over the pixels predicted as c (falls back to the
       whole image if no pixel is predicted as c)
    3. Backward pass to get gradients w.r.t. a chosen intermediate layer
    4. Weight feature maps by global-average-pooled gradients
    5. ReLU + normalize to get the attention heatmap

    Hooks are registered once and removed by `remove()`; use it as a context
    manager so a long-lived model (e.g. cached in the dashboard) never
    accumulates hooks:

        with SegmentationGradCAM(model, device=device) as cam:
            heatmap = cam.generate(tensor, class_idx)

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

        self.target_layer = target_layer or find_target_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            # A tensor hook (rather than a module backward hook) works even
            # when the layer ends in an in-place ReLU.
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def remove(self) -> None:
        """Detach the hook from the model."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "SegmentationGradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

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
        self.model.zero_grad(set_to_none=True)
        self.activations = self.gradients = None

        with torch.enable_grad():
            logits = self.model(image_tensor.to(self.device))  # (1, C, H, W)

            class_logits = logits[0, target_class]
            predicted = logits[0].argmax(dim=0) == target_class
            target = class_logits[predicted].sum() if predicted.any() else class_logits.mean()
            target.backward()

        if self.activations is None or self.gradients is None:
            logging.warning("No activations/gradients captured. Check target layer.")
            return np.zeros(image_tensor.shape[2:], dtype=np.float32)

        # Global average pool gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C_feat, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().float().cpu().numpy()
        span = cam.max() - cam.min()
        if span > 0:
            cam = (cam - cam.min()) / span
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)

    def generate_all_classes(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for all classes.

        Args:
            image_tensor: Preprocessed image tensor.

        Returns:
            Dictionary of class_name -> heatmap.
        """
        return {
            cls_name: self.generate(image_tensor, cls_idx)
            for cls_idx, cls_name in enumerate(CLASS_NAMES)
        }


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
    aspect = image.shape[0] / image.shape[1]
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5 * aspect + 1))

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

    aspect = image.shape[0] / image.shape[1]
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * aspect * rows + 1))
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
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for segmentation models"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model", type=str, default=None, choices=ARCHITECTURES,
                        help="Model architecture (only needed for old checkpoints "
                             "without metadata)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--target-class", type=str, default=None,
                        help="Target class name (e.g., 'road', 'vehicle')")
    parser.add_argument("--all-classes", action="store_true",
                        help="Generate Grad-CAM for all classes")
    parser.add_argument("--output-dir", type=str, default="./outputs/explainability",
                        help="Output directory")
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        help="Model input size H W (default: the size the model was trained at)")

    args = parser.parse_args()

    setup_logging()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model (architecture and input size come from the checkpoint)
    model, meta = load_model_from_checkpoint(args.checkpoint, device, args.model)
    image_size = tuple(args.image_size or meta["image_size"] or (512, 1024))

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = preprocess_image(image, image_size).to(device)

    # Get prediction at the original resolution
    with torch.no_grad():
        logits = F.interpolate(model(tensor), size=image.shape[:2],
                               mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

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

    gradcam.remove()
    print(f"\n[OK] Grad-CAM visualization saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
