"""
train.py — Training Pipeline for Semantic Segmentation
========================================================

Full-featured training pipeline with:
- Mixed Precision Training (FP16 via torch.cuda.amp)
- Gradient Clipping
- Learning Rate Scheduling (Cosine, Plateau, Step)
- Early Stopping
- Checkpoint Saving (best + latest)
- Resume Training from checkpoint
- TensorBoard Logging
- Per-epoch metric tables

Usage:
    # Train U-Net with combined loss
    python train.py --model unet --loss combined --epochs 100

    # Train ResNet-UNet with frozen encoder
    python train.py --model resnet_unet --freeze-encoder --lr 3e-4

    # Resume training
    python train.py --resume checkpoints/latest_model.pth

    # Use mock dataset for development
    python train.py --model unet --epochs 5 --use-mock

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    NUM_CLASSES,
    CLASS_NAMES,
    Config,
    DataConfig,
    LossConfig,
    ModelConfig,
    PathConfig,
    TrainConfig,
    parse_args,
    config_from_args,
)
from dataset import get_dataloaders
from losses import CombinedLoss, get_loss_function
from metrics import SegmentationMetrics, compute_batch_iou, compute_pixel_accuracy
from model_factory import get_model
from utils import (
    EarlyStopping,
    get_device,
    load_checkpoint,
    plot_training_curves,
    print_model_summary,
    save_checkpoint,
    seed_everything,
    setup_logging,
    visualize_predictions,
    denormalize_image,
    colorize_mask,
)


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """
    Full-featured training pipeline for semantic segmentation models.

    Handles the complete training lifecycle including:
    - Model initialization and device placement
    - Optimizer and scheduler setup
    - Mixed precision training with gradient scaling
    - Epoch-level training and validation loops
    - Metric computation and logging
    - Checkpoint management
    - Early stopping
    - TensorBoard integration

    Args:
        config: Master configuration object.
        use_mock: Use mock dataset for development.
    """

    def __init__(self, config: Config, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock

        # Setup
        setup_logging(
            log_file=os.path.join(config.paths.training_dir, "training.log")
        )
        seed_everything(config.train.seed)
        self.device = get_device()

        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_amp()
        self._setup_early_stopping()
        self._setup_tensorboard()

        # Training state
        self.start_epoch = 0
        self.best_metric = 0.0
        self.history = defaultdict(list)

        # Resume if specified
        if config.train.resume_from:
            self._resume_training(config.train.resume_from)

        logging.info("=" * 60)
        logging.info("  Trainer initialized successfully")
        logging.info(f"  Model      : {config.model.architecture}")
        logging.info(f"  Loss       : {config.loss.loss_type}")
        logging.info(f"  Device     : {self.device}")
        logging.info(f"  Epochs     : {config.train.epochs}")
        logging.info(f"  Batch Size : {config.data.batch_size}")
        logging.info(f"  LR         : {config.train.learning_rate}")
        logging.info(f"  Mixed Prec : {config.train.use_mixed_precision}")
        logging.info("=" * 60)

    def _setup_model(self) -> None:
        """Initialize and configure the model."""
        self.model = get_model(self.config.model).to(self.device)
        print_model_summary(self.model, self.config.model.architecture)

    def _setup_data(self) -> None:
        """Initialize data loaders."""
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.config, use_mock=self.use_mock
        )
        logging.info(
            f"Data loaders ready — "
            f"Train: {len(self.train_loader)} batches, "
            f"Val: {len(self.val_loader)} batches"
        )

    def _setup_loss(self) -> None:
        """Initialize loss function."""
        self.criterion = get_loss_function(self.config.loss).to(self.device)

    def _setup_optimizer(self) -> None:
        """Initialize optimizer."""
        cfg = self.config.train

        if cfg.optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer.lower() == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        logging.info(f"Optimizer: {cfg.optimizer} (lr={cfg.learning_rate})")

    def _setup_scheduler(self) -> None:
        """Initialize learning rate scheduler."""
        cfg = self.config.train

        if cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.scheduler_t_max, T_mult=2
            )
        elif cfg.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=cfg.scheduler_patience,
                min_lr=cfg.min_lr,
            )
        elif cfg.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif cfg.scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        logging.info(f"Scheduler: {cfg.scheduler}")

    def _setup_amp(self) -> None:
        """Initialize Automatic Mixed Precision components."""
        self.use_amp = (
            self.config.train.use_mixed_precision
            and torch.cuda.is_available()
        )
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        if self.use_amp:
            logging.info("Mixed precision training enabled (FP16)")
        else:
            logging.info("Mixed precision disabled (FP32)")

    def _setup_early_stopping(self) -> None:
        """Initialize early stopping monitor."""
        self.early_stopping = EarlyStopping(
            patience=self.config.train.early_stopping_patience,
            min_delta=self.config.train.early_stopping_min_delta,
            mode="max",  # Monitoring mIoU (higher is better)
        )

    def _setup_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(
            log_dir=self.config.paths.tensorboard_dir
        )
        logging.info(f"TensorBoard: {self.config.paths.tensorboard_dir}")

    def _resume_training(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        meta = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            device=self.device,
        )
        self.start_epoch = meta["epoch"] + 1
        self.best_metric = meta["best_metric"]
        self.history = defaultdict(list, meta.get("history", {}))
        logging.info(
            f"Resumed from epoch {self.start_epoch}, "
            f"best mIoU: {self.best_metric:.4f}"
        )

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dictionary.
        """
        logging.info("\nStarting training...\n")

        total_epochs = self.config.train.epochs

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()

            # ---- Train one epoch ----
            train_metrics = self._train_one_epoch(epoch)

            # ---- Validate ----
            val_metrics = self._validate(epoch)

            # ---- Learning rate scheduling ----
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mean_iou"])
                else:
                    self.scheduler.step()

            # ---- Record history ----
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_iou"].append(train_metrics["mean_iou"])
            self.history["val_iou"].append(val_metrics["mean_iou"])
            self.history["train_acc"].append(train_metrics["pixel_accuracy"])
            self.history["val_acc"].append(val_metrics["pixel_accuracy"])
            self.history["lr"].append(current_lr)

            # ---- TensorBoard logging ----
            self._log_tensorboard(epoch, train_metrics, val_metrics, current_lr)

            # ---- Checkpoint saving ----
            is_best = val_metrics["mean_iou"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["mean_iou"]
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    history=dict(self.history),
                    save_path=self.config.paths.best_model_path,
                )
                logging.info(f"  [NEW BEST] New best model! mIoU: {self.best_metric:.4f}")

            # Always save latest
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_metric=self.best_metric,
                history=dict(self.history),
                save_path=self.config.paths.latest_model_path,
            )

            # ---- Epoch summary ----
            elapsed = time.time() - epoch_start
            self._print_epoch_summary(
                epoch, total_epochs, train_metrics, val_metrics,
                current_lr, elapsed, is_best
            )

            # ---- Early stopping ----
            if self.early_stopping(val_metrics["mean_iou"]):
                logging.info(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best mIoU: {self.best_metric:.4f}"
                )
                break

            # ---- Periodic visualization ----
            if (epoch + 1) % self.config.train.save_every_n_epochs == 0:
                self._save_sample_predictions(epoch)

        # ---- Training complete ----
        self.writer.close()

        # Save training curves
        plot_training_curves(
            history=dict(self.history),
            save_path=self.config.paths.training_dir,
        )

        logging.info(f"\n[OK] Training complete! Best mIoU: {self.best_metric:.4f}")
        logging.info(f"   Best model: {self.config.paths.best_model_path}")

        return dict(self.history)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for this epoch.
        """
        self.model.train()
        metrics = SegmentationMetrics(NUM_CLASSES)

        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False,
            ncols=100,
        )

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # ---- Forward pass with mixed precision ----
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # ---- Backward pass with gradient scaling ----
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # ---- Gradient clipping ----
            if self.config.train.grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.grad_clip_max_norm,
                )

            # ---- Optimizer step ----
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ---- Metrics ----
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                metrics.update(predictions, masks)

            running_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mIoU": f"{metrics.mean_iou():.4f}",
            })

        avg_loss = running_loss / max(num_batches, 1)
        summary = metrics.get_summary()
        summary["loss"] = avg_loss

        return summary

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on the validation set.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        metrics = SegmentationMetrics(NUM_CLASSES)

        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]  ",
            leave=False,
            ncols=100,
        )

        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            predictions = outputs.argmax(dim=1)
            metrics.update(predictions, masks)

            running_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mIoU": f"{metrics.mean_iou():.4f}",
            })

        avg_loss = running_loss / max(num_batches, 1)
        summary = metrics.get_summary()
        summary["loss"] = avg_loss

        # Print per-class metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics.print_report()

        return summary

    # ========================================================================
    # LOGGING & VISUALIZATION
    # ========================================================================

    def _log_tensorboard(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
    ) -> None:
        """Log metrics to TensorBoard."""
        self.writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)

        self.writer.add_scalars("mIoU", {
            "train": train_metrics["mean_iou"],
            "val": val_metrics["mean_iou"],
        }, epoch)

        self.writer.add_scalars("Pixel Accuracy", {
            "train": train_metrics["pixel_accuracy"],
            "val": val_metrics["pixel_accuracy"],
        }, epoch)

        self.writer.add_scalar("Learning Rate", lr, epoch)

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train: Dict,
        val: Dict,
        lr: float,
        elapsed: float,
        is_best: bool,
    ) -> None:
        """Print formatted epoch summary."""
        best_marker = " [BEST]" if is_best else ""

        print(
            f"\n  Epoch [{epoch + 1}/{total_epochs}] - {elapsed:.1f}s"
            f"\n  |-- Train: Loss={train['loss']:.4f} | "
            f"mIoU={train['mean_iou']:.4f} | "
            f"Acc={train['pixel_accuracy']:.4f}"
            f"\n  |-- Val  : Loss={val['loss']:.4f} | "
            f"mIoU={val['mean_iou']:.4f} | "
            f"Acc={val['pixel_accuracy']:.4f}"
            f"\n  \\-- LR={lr:.2e} | "
            f"Best mIoU={self.best_metric:.4f}{best_marker}"
        )

    def _save_sample_predictions(self, epoch: int) -> None:
        """Save sample prediction visualizations."""
        self.model.eval()
        images_list = []
        gt_list = []
        pred_list = []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                for i in range(min(4, images.size(0))):
                    img = denormalize_image(images[i].cpu().numpy())
                    gt = masks[i].cpu().numpy()
                    pred = predictions[i].cpu().numpy()

                    images_list.append(img)
                    gt_list.append(gt)
                    pred_list.append(pred)

                if len(images_list) >= 4:
                    break

        if images_list:
            save_path = os.path.join(
                self.config.paths.training_dir,
                f"predictions_epoch_{epoch + 1:03d}.png"
            )
            visualize_predictions(
                images_list, gt_list, pred_list, save_path, num_samples=4
            )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments (reuse from config)
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"])
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["ce", "dice", "focal", "combined"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step", "none"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="./dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock dataset for development/testing")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    args = parser.parse_args()

    # Build config from arguments
    config = Config(
        data=DataConfig(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
        ),
        model=ModelConfig(
            architecture=args.model,
            freeze_encoder=args.freeze_encoder,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            early_stopping_patience=args.patience,
            grad_clip_max_norm=args.grad_clip,
            seed=args.seed,
            use_mixed_precision=not args.no_mixed_precision,
            resume_from=args.resume,
        ),
        loss=LossConfig(
            loss_type=args.loss,
            focal_gamma=args.focal_gamma,
        ),
        paths=PathConfig(
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            eda_dir=os.path.join(args.output_dir, "eda"),
            training_dir=os.path.join(args.output_dir, "training"),
            evaluation_dir=os.path.join(args.output_dir, "evaluation"),
            predictions_dir=os.path.join(args.output_dir, "predictions"),
            explainability_dir=os.path.join(args.output_dir, "explainability"),
            tensorboard_dir=os.path.join(args.output_dir, "training", "logs"),
        ),
    )

    # Create trainer and run
    trainer = Trainer(config, use_mock=args.use_mock)
    history = trainer.train()

    print("\n[OK] Training completed successfully!")
    print(f"   Best checkpoint: {config.paths.best_model_path}")
    print(f"   Training curves: {config.paths.training_dir}")
    print(f"   TensorBoard:     tensorboard --logdir {config.paths.tensorboard_dir}")


if __name__ == "__main__":
    main()
