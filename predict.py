"""
predict.py — Inference Pipeline for Semantic Segmentation
===========================================================

Supports:
1. Single image inference
2. Batch inference (directory of images)
3. Real-time webcam inference with live overlay

Outputs per image:
- Segmentation mask (color-coded)
- Overlay image (semi-transparent mask on original)
- Confidence map (per-pixel prediction confidence heatmap)

Usage:
    # Single image
    python predict.py --image sample.jpg --model unet --checkpoint best_model.pth

    # Batch inference
    python predict.py --image-dir ./test_images/ --model unet --checkpoint best_model.pth

    # Webcam
    python predict.py --webcam --model unet --checkpoint best_model.pth

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    Config,
    ModelConfig,
    PathConfig,
)
from model_factory import get_model
from utils import (
    colorize_mask,
    create_overlay,
    get_device,
    load_checkpoint,
    setup_logging,
)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class SegmentationPredictor:
    """
    Production inference engine for semantic segmentation.

    Handles image preprocessing, model inference, and post-processing
    for single images, batch directories, and real-time video streams.

    Args:
        model_name: Architecture name.
        checkpoint_path: Path to trained model checkpoint.
        image_size: (height, width) for input preprocessing.
        device: Inference device.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (512, 512),
        device: Optional[torch.device] = None,
    ):
        setup_logging()
        self.image_size = image_size
        self.device = device or get_device()

        # Load model
        model_config = ModelConfig(architecture=model_name)
        self.model = get_model(model_config).to(self.device)

        # Load weights
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        logging.info(f"Predictor ready: {model_name} on {self.device}")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess an image for model input.

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            tensor: Preprocessed tensor (1, 3, H, W).
            original_size: (H, W) of the original image.
        """
        original_size = image.shape[:2]

        # Resize
        resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        # Normalize (ImageNet)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        normalized = (normalized - mean) / std

        # To tensor: (H, W, 3) → (1, 3, H, W)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

        return tensor.to(self.device), original_size

    @torch.no_grad()
    def predict(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            Dictionary with:
                'mask': Class prediction mask (H, W), int
                'colored_mask': RGB colored mask (H, W, 3), uint8
                'overlay': Image with mask overlay (H, W, 3), uint8
                'confidence': Per-pixel confidence map (H, W), float32
                'probabilities': Full probability map (H, W, C), float32
        """
        tensor, original_size = self.preprocess(image)

        # Model inference
        logits = self.model(tensor)  # (1, C, H, W)

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (1, C, H, W)

        # Class predictions
        confidence, predictions = probs.max(dim=1)  # (1, H, W)

        # Convert to numpy
        mask = predictions[0].cpu().numpy().astype(np.int32)
        confidence_map = confidence[0].cpu().numpy()
        prob_map = probs[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

        # Resize back to original size
        h, w = original_size
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(confidence_map, (w, h))
        mask = mask.astype(np.int32)

        # Generate visualizations
        colored_mask = colorize_mask(mask)
        overlay = create_overlay(image, mask, alpha=0.4)

        return {
            "mask": mask,
            "colored_mask": colored_mask,
            "overlay": overlay,
            "confidence": confidence_map,
            "probabilities": prob_map,
        }

    def predict_and_save(
        self,
        image_path: str,
        save_dir: str,
    ) -> Dict[str, str]:
        """
        Run inference on an image file and save all outputs.

        Args:
            image_path: Path to input image.
            save_dir: Directory to save outputs.

        Returns:
            Dictionary of output file paths.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict
        results = self.predict(image)

        # Generate filename base
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save outputs
        paths = {}

        # 1. Segmentation mask
        mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, cv2.cvtColor(results["colored_mask"], cv2.COLOR_RGB2BGR))
        paths["mask"] = mask_path

        # 2. Overlay
        overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(results["overlay"], cv2.COLOR_RGB2BGR))
        paths["overlay"] = overlay_path

        # 3. Confidence map
        confidence_path = os.path.join(save_dir, f"{base_name}_confidence.png")
        confidence_vis = (results["confidence"] * 255).astype(np.uint8)
        confidence_colored = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_JET)
        cv2.imwrite(confidence_path, confidence_colored)
        paths["confidence"] = confidence_path

        # 4. Combined visualization
        combined_path = os.path.join(save_dir, f"{base_name}_combined.png")
        self._save_combined_visualization(
            image, results, combined_path, base_name
        )
        paths["combined"] = combined_path

        logging.info(f"Predictions saved for: {base_name}")
        return paths

    def _save_combined_visualization(
        self,
        image: np.ndarray,
        results: Dict,
        save_path: str,
        title: str = "",
    ) -> None:
        """Save a combined 4-panel visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(results["colored_mask"])
        axes[0, 1].set_title("Predicted Segmentation", fontsize=13, fontweight="bold")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(results["overlay"])
        axes[1, 0].set_title("Overlay", fontsize=13, fontweight="bold")
        axes[1, 0].axis("off")

        confidence_vis = axes[1, 1].imshow(
            results["confidence"], cmap="jet", vmin=0, vmax=1
        )
        axes[1, 1].set_title("Confidence Map", fontsize=13, fontweight="bold")
        axes[1, 1].axis("off")
        plt.colorbar(confidence_vis, ax=axes[1, 1], fraction=0.046)

        # Add legend
        legend_elements = []
        for i, name in enumerate(CLASS_NAMES):
            color = np.array(CLASS_COLORS[i]) / 255.0
            legend_elements.append(
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                           markersize=10, label=name)
            )
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=len(CLASS_NAMES), fontsize=9)

        plt.suptitle(
            f"Semantic Segmentation - {title}",
            fontsize=15, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ========================================================================
    # BATCH INFERENCE
    # ========================================================================

    def predict_directory(
        self,
        image_dir: str,
        save_dir: str,
    ) -> None:
        """
        Run inference on all images in a directory.

        Args:
            image_dir: Directory containing input images.
            save_dir: Directory to save outputs.
        """
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        image_files = [
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ]

        if not image_files:
            logging.warning(f"No images found in: {image_dir}")
            return

        logging.info(f"Batch inference: {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            self.predict_and_save(img_path, save_dir)

        logging.info(f"Batch inference complete. Results in: {save_dir}")

    # ========================================================================
    # REAL-TIME WEBCAM INFERENCE
    # ========================================================================

    def run_webcam(self, camera_id: int = 0) -> None:
        """
        Run real-time semantic segmentation on webcam feed.

        Shows live overlay with FPS counter. Press 'q' to quit.

        Args:
            camera_id: OpenCV camera device ID.
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logging.error(f"Cannot open camera {camera_id}")
            return

        print("\n" + "=" * 50)
        print("  [Webcam] Real-Time Webcam Segmentation")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
        print("=" * 50 + "\n")

        frame_count = 0
        fps_start = time.time()
        fps = 0.0

        save_dir = "./outputs/predictions/webcam"
        os.makedirs(save_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict
            results = self.predict(frame_rgb)

            # Create display
            overlay = results["overlay"]
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # Compute FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Add FPS overlay
            cv2.putText(
                overlay_bgr, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

            # Add class legend
            y_offset = 60
            for i, name in enumerate(CLASS_NAMES):
                color = CLASS_COLORS[i]
                cv2.rectangle(
                    overlay_bgr,
                    (10, y_offset + i * 25),
                    (30, y_offset + i * 25 + 15),
                    color[::-1],  # RGB → BGR
                    -1,
                )
                cv2.putText(
                    overlay_bgr, name,
                    (35, y_offset + i * 25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

            # Display
            cv2.imshow("Semantic Segmentation - Real-Time", overlay_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_path = os.path.join(save_dir, f"webcam_frame_{int(time.time())}.png")
                cv2.imwrite(save_path, overlay_bgr)
                print(f"  Saved: {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        print("\n  Webcam inference stopped.")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation inference"
    )

    # Input
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image for single inference")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of images for batch inference")
    parser.add_argument("--webcam", action="store_true",
                        help="Run real-time webcam inference")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Webcam device ID")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Settings
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                        help="Input image size (H W)")
    parser.add_argument("--output-dir", type=str, default="./outputs/predictions",
                        help="Output directory for results")

    args = parser.parse_args()

    # Initialize predictor
    predictor = SegmentationPredictor(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        image_size=tuple(args.image_size),
    )

    # Run inference
    if args.webcam:
        predictor.run_webcam(camera_id=args.camera_id)

    elif args.image:
        paths = predictor.predict_and_save(args.image, args.output_dir)
        print(f"\n[OK] Inference complete!")
        for name, path in paths.items():
            print(f"   {name:12s}: {path}")

    elif args.image_dir:
        predictor.predict_directory(args.image_dir, args.output_dir)
        print(f"\n[OK] Batch inference complete! Results in: {args.output_dir}")

    else:
        print("Error: Specify --image, --image-dir, or --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
