"""
evaluate.py — Model Evaluation on the Test Set
================================================

Comprehensive evaluation pipeline that generates:
1. Confusion matrix (heatmap visualization)
2. Per-class IoU table
3. Precision/Recall table
4. Dice score table
5. Model comparison table (if multiple checkpoints provided)

Predictions are upsampled to the ground-truth resolution before scoring, so
metrics are computed on the original label maps, not on downscaled ones.

All results are exported as CSV/JSON and visualized as PNG.

Usage:
    # Evaluate a single model (architecture is read from the checkpoint)
    python evaluate.py --checkpoint checkpoints/unet_best.pth

    # Compare several trained models
    python evaluate.py --checkpoint checkpoints/unet_best.pth checkpoints/deeplabv3plus_best.pth

    # Evaluate on synthetic data generated on the fly
    python evaluate.py --checkpoint checkpoints/unet_best.pth --use-mock

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ARCHITECTURES, CLASS_NAMES, NUM_CLASSES
from dataset import CityscapesSegDataset, MockSegDataset, get_validation_augmentations
from metrics import SegmentationMetrics
from model_factory import load_model_from_checkpoint
from utils import (
    denormalize_image,
    get_device,
    plot_confusion_matrix,
    setup_logging,
    visualize_predictions,
)


# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    """
    Comprehensive model evaluator for semantic segmentation.

    Runs inference on a full split, computes all metrics, and generates
    visualizations and CSV exports.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        save_dir: Directory for the reports.
        dataset_root: Prepared dataset root (ignored with use_mock).
        split: 'test' (default) or 'val'.
        model_name: Only needed for old checkpoints without metadata.
        image_size: Model input size; defaults to the training size.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
        use_mock: Use synthetic data generated on the fly.
    """

    def __init__(
        self,
        checkpoint_path: str,
        save_dir: str,
        dataset_root: str = "./dataset",
        split: str = "test",
        model_name: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        use_mock: bool = False,
    ):
        setup_logging()
        self.device = get_device()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model, self.meta = load_model_from_checkpoint(
            checkpoint_path, self.device, model_name
        )
        self.architecture = self.meta["model_config"]["architecture"]
        self.image_size = tuple(image_size or self.meta["image_size"] or (512, 1024))
        self.split = split

        transform = get_validation_augmentations(*self.image_size)
        if use_mock:
            # Same seeds as dataset.get_dataloaders, so this is the same split
            seed = 200_000 if split == "test" else 100_000
            dataset = MockSegDataset(100, self.image_size, transform, seed=seed)
        else:
            dataset = CityscapesSegDataset(dataset_root, split, transform, full_res_mask=True)

        if len(dataset) == 0:
            raise RuntimeError(f"No '{split}' samples found in '{dataset_root}'.")

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        logging.info(
            f"Evaluating {self.architecture} on {len(dataset)} '{split}' samples "
            f"(input {self.image_size[0]}x{self.image_size[1]})"
        )

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run full evaluation on the split.

        Returns:
            Dictionary of all computed metrics.
        """
        metrics = SegmentationMetrics(NUM_CLASSES)
        vis_images, vis_gt, vis_pred = [], [], []
        use_amp = self.device.type == "cuda"
        inference_seconds = 0.0
        num_images = 0

        for images, masks in tqdm(self.loader, desc="  Evaluating", ncols=100):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with autocast(device_type=self.device.type, enabled=use_amp):
                logits = self.model(images)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            num_images += images.size(0)

            # Score at ground-truth resolution
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits.float(), size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1)
            metrics.update(predictions, masks)

            if len(vis_images) < 8:
                for i in range(min(2, images.size(0))):
                    img = denormalize_image(images[i].float().cpu().numpy())
                    size = (img.shape[1], img.shape[0])
                    vis_images.append(img)
                    vis_gt.append(_resize_mask(masks[i].cpu().numpy(), size))
                    vis_pred.append(_resize_mask(predictions[i].cpu().numpy(), size))

        self._save_confusion_matrix(metrics)
        self._save_metric_tables(metrics)
        if vis_images:
            visualize_predictions(
                vis_images, vis_gt, vis_pred,
                save_path=os.path.join(self.save_dir, "test_predictions.png"),
                num_samples=min(8, len(vis_images)),
            )

        results = metrics.get_detailed_metrics()
        results.update({
            "class_names": CLASS_NAMES,
            "architecture": self.architecture,
            "split": self.split,
            "num_images": num_images,
            "input_size": list(self.image_size),
            "model_ms_per_image": 1000 * inference_seconds / max(num_images, 1),
            "checkpoint_epoch": self.meta["epoch"],
        })
        json_path = os.path.join(self.save_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"JSON results saved to {json_path}")

        metrics.print_report()
        return results

    def _save_confusion_matrix(self, metrics: SegmentationMetrics) -> None:
        """Generate and save confusion matrix heatmaps."""
        cm = metrics.get_confusion_matrix()
        plot_confusion_matrix(
            cm, CLASS_NAMES,
            save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
            normalize=True,
            title="Confusion Matrix (Normalized %)",
        )
        plot_confusion_matrix(
            cm, CLASS_NAMES,
            save_path=os.path.join(self.save_dir, "confusion_matrix_raw.png"),
            normalize=False,
            title="Confusion Matrix (Pixel Counts)",
        )

    def _save_metric_tables(self, metrics: SegmentationMetrics) -> None:
        """Save per-class metric tables as CSV (empty cell = undefined)."""
        def save(name: str, data: Dict) -> None:
            pd.DataFrame(data).to_csv(
                os.path.join(self.save_dir, name), index=False, float_format="%.4f"
            )

        save("per_class_iou.csv", {"Class": CLASS_NAMES, "IoU": metrics.per_class_iou()})
        save("precision_recall.csv", {
            "Class": CLASS_NAMES,
            "Precision": metrics.precision(),
            "Recall": metrics.recall(),
            "F1": metrics.f1_score(),
        })
        save("dice_scores.csv", {"Class": CLASS_NAMES, "Dice": metrics.dice_score()})
        metrics.generate_report().to_csv(
            os.path.join(self.save_dir, "full_evaluation_report.csv"),
            index=False, float_format="%.4f",
        )
        logging.info(f"Metric tables saved to {self.save_dir}")


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map to (width, height)."""
    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(results: List[Dict], save_path: str) -> pd.DataFrame:
    """
    Build a comparison table from several evaluation results.

    Args:
        results: List of Evaluator.evaluate() outputs.
        save_path: CSV path for the table.

    Returns:
        Comparison DataFrame.
    """
    rows = []
    for r in results:
        row = {
            "Model": r["architecture"],
            "mIoU": r["mean_iou"],
            "mDice": r["mean_dice"],
            "Pixel Acc": r["pixel_accuracy"],
            "ms/img": r["model_ms_per_image"],
        }
        for name, iou in zip(CLASS_NAMES, r["per_class_iou"]):
            row[f"IoU {name}"] = iou
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format="%.4f")

    print(f"\n{'=' * 60}")
    print(f"  Model Comparison Results")
    print(f"{'=' * 60}")
    print(df[["Model", "mIoU", "mDice", "Pixel Acc", "ms/img"]].to_string(index=False))
    print(f"{'=' * 60}")
    return df


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained segmentation model(s)"
    )
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="Path(s) to model checkpoint(s); several are compared")
    parser.add_argument("--model", type=str, default=None, choices=ARCHITECTURES,
                        help="Model architecture (only needed for old checkpoints "
                             "without metadata)")
    parser.add_argument("--dataset-root", type=str, default="./dataset",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        help="Model input size H W (default: the size the model was trained at)")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use synthetic data generated on the fly")

    args = parser.parse_args()

    all_results = []
    for ckpt in args.checkpoint:
        run_name = os.path.splitext(os.path.basename(ckpt))[0].removesuffix("_best")
        evaluator = Evaluator(
            checkpoint_path=ckpt,
            save_dir=os.path.join(args.output_dir, "evaluation", run_name),
            dataset_root=args.dataset_root,
            split=args.split,
            model_name=args.model,
            image_size=tuple(args.image_size) if args.image_size else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_mock=args.use_mock,
        )
        all_results.append(evaluator.evaluate())
        print(f"\n[OK] {run_name}: results saved to {evaluator.save_dir}")

    if len(all_results) > 1:
        compare_models(
            all_results,
            os.path.join(args.output_dir, "evaluation", "model_comparison.csv"),
        )


if __name__ == "__main__":
    main()
