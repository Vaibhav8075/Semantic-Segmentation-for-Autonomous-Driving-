"""
evaluate.py — Model Evaluation on Test Set
============================================

Comprehensive evaluation pipeline that generates:
1. Confusion matrix (heatmap visualization)
2. Per-class IoU table
3. Precision/Recall table
4. Dice score table
5. Model comparison table (if multiple checkpoints provided)

All results are exported as CSV and visualized as PNG.

Usage:
    # Evaluate a single model
    python evaluate.py --model unet --checkpoint checkpoints/best_model.pth

    # Evaluate with mock data
    python evaluate.py --model unet --checkpoint checkpoints/best_model.pth --use-mock

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    NUM_CLASSES,
    Config,
    DataConfig,
    ModelConfig,
    PathConfig,
)
from dataset import get_dataloaders
from metrics import SegmentationMetrics
from model_factory import get_model
from utils import (
    colorize_mask,
    create_overlay,
    denormalize_image,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
    save_metric_tables,
    setup_logging,
    visualize_predictions,
)


# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    """
    Comprehensive model evaluator for semantic segmentation.

    Runs inference on the full test set, computes all metrics, and
    generates visualizations and CSV exports.

    Args:
        model_name: Architecture name (for loading).
        checkpoint_path: Path to trained model checkpoint.
        config: Project configuration.
        use_mock: Use mock dataset for testing.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        config: Config,
        use_mock: bool = False,
    ):
        setup_logging()
        self.config = config
        self.device = get_device()
        self.save_dir = config.paths.evaluation_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Load model
        self.config.model.architecture = model_name
        self.model = get_model(self.config.model).to(self.device)

        # Load checkpoint
        meta = load_checkpoint(
            checkpoint_path, self.model, device=self.device
        )
        self.model.eval()

        logging.info(
            f"Model loaded: {model_name} (epoch {meta['epoch']}, "
            f"best mIoU: {meta['best_metric']:.4f})"
        )

        # Load test data
        _, _, self.test_loader = get_dataloaders(config, use_mock=use_mock)
        logging.info(f"Test set: {len(self.test_loader)} batches")

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run full evaluation on the test set.

        Returns:
            Dictionary of all computed metrics.
        """
        logging.info("\nStarting evaluation on test set...")

        metrics = SegmentationMetrics(NUM_CLASSES)

        # Collect samples for visualization
        vis_images = []
        vis_gt = []
        vis_pred = []

        pbar = tqdm(self.test_loader, desc="  Evaluating", ncols=100)

        for images, masks in pbar:
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)

            metrics.update(predictions, masks)

            # Collect visualization samples
            if len(vis_images) < 8:
                for i in range(min(2, images.size(0))):
                    img = denormalize_image(images[i].cpu().numpy())
                    vis_images.append(img)
                    vis_gt.append(masks[i].cpu().numpy())
                    vis_pred.append(predictions[i].cpu().numpy())

            pbar.set_postfix({"mIoU": f"{metrics.mean_iou():.4f}"})

        # ---- Generate Reports ----
        self._save_confusion_matrix(metrics)
        self._save_metric_tables(metrics)
        self._save_predictions_visualization(vis_images, vis_gt, vis_pred)
        self._save_json_results(metrics)

        # Print report
        metrics.print_report()

        return metrics.get_detailed_metrics()

    def _save_confusion_matrix(self, metrics: SegmentationMetrics) -> None:
        """Generate and save confusion matrix heatmap."""
        cm = metrics.get_confusion_matrix()

        # Normalized version
        plot_confusion_matrix(
            cm, CLASS_NAMES,
            save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
            normalize=True,
            title="Confusion Matrix (Normalized %)",
        )

        # Raw counts version
        plot_confusion_matrix(
            cm, CLASS_NAMES,
            save_path=os.path.join(self.save_dir, "confusion_matrix_raw.png"),
            normalize=False,
            title="Confusion Matrix (Pixel Counts)",
        )

    def _save_metric_tables(self, metrics: SegmentationMetrics) -> None:
        """Save per-class metric tables as CSV."""
        # IoU table
        iou_data = {
            "Class": CLASS_NAMES,
            "IoU": metrics.per_class_iou().tolist(),
        }
        pd.DataFrame(iou_data).to_csv(
            os.path.join(self.save_dir, "per_class_iou.csv"),
            index=False, float_format="%.4f"
        )

        # Precision/Recall table
        pr_data = {
            "Class": CLASS_NAMES,
            "Precision": metrics.precision().tolist(),
            "Recall": metrics.recall().tolist(),
            "F1": metrics.f1_score().tolist(),
        }
        pd.DataFrame(pr_data).to_csv(
            os.path.join(self.save_dir, "precision_recall.csv"),
            index=False, float_format="%.4f"
        )

        # Dice table
        dice_data = {
            "Class": CLASS_NAMES,
            "Dice": metrics.dice_score().tolist(),
        }
        pd.DataFrame(dice_data).to_csv(
            os.path.join(self.save_dir, "dice_scores.csv"),
            index=False, float_format="%.4f"
        )

        # Full report
        report = metrics.generate_report()
        report.to_csv(
            os.path.join(self.save_dir, "full_evaluation_report.csv"),
            index=False, float_format="%.4f"
        )

        logging.info(f"Metric tables saved to {self.save_dir}")

    def _save_predictions_visualization(
        self,
        images: List[np.ndarray],
        gt_masks: List[np.ndarray],
        pred_masks: List[np.ndarray],
    ) -> None:
        """Save prediction visualizations."""
        if images:
            visualize_predictions(
                images, gt_masks, pred_masks,
                save_path=os.path.join(self.save_dir, "test_predictions.png"),
                num_samples=min(8, len(images)),
            )

    def _save_json_results(self, metrics: SegmentationMetrics) -> None:
        """Save all metrics as JSON."""
        results = metrics.get_detailed_metrics()
        results["class_names"] = CLASS_NAMES

        json_path = os.path.join(self.save_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"JSON results saved to {json_path}")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    model_configs: List[Dict],
    config: Config,
    use_mock: bool = False,
) -> pd.DataFrame:
    """
    Compare multiple trained models on the test set.

    Args:
        model_configs: List of {"name": str, "architecture": str, "checkpoint": str}
        config: Project configuration.
        use_mock: Use mock dataset.

    Returns:
        Comparison DataFrame.
    """
    results = []

    for mc in model_configs:
        print(f"\n  Evaluating: {mc['name']}...")

        evaluator = Evaluator(
            model_name=mc["architecture"],
            checkpoint_path=mc["checkpoint"],
            config=config,
            use_mock=use_mock,
        )

        metrics = evaluator.evaluate()

        results.append({
            "Model": mc["name"],
            "mIoU": metrics["mean_iou"],
            "mDice": metrics["mean_dice"],
            "Pixel Acc": metrics["pixel_accuracy"],
        })

    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(config.paths.evaluation_dir, "model_comparison.csv"),
        index=False, float_format="%.4f"
    )

    print(f"\n{'=' * 60}")
    print(f"  Model Comparison Results")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"{'=' * 60}")

    return df


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained segmentation model on test set"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str, default="./dataset",
                        help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock dataset")

    args = parser.parse_args()

    config = Config(
        data=DataConfig(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
        ),
        model=ModelConfig(architecture=args.model),
        paths=PathConfig(
            output_dir=args.output_dir,
            evaluation_dir=os.path.join(args.output_dir, "evaluation"),
        ),
    )

    evaluator = Evaluator(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        config=config,
        use_mock=args.use_mock,
    )

    results = evaluator.evaluate()

    print(f"\n[OK] Evaluation complete!")
    print(f"   Results saved to: {config.paths.evaluation_dir}")


if __name__ == "__main__":
    main()
