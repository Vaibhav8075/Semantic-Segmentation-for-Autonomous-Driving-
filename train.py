"""
train.py — Training Pipeline for Semantic Segmentation
========================================================

Full-featured training pipeline with:
- Mixed Precision Training (FP16 via torch.amp)
- Gradient Clipping
- Learning Rate Schedule (linear warmup + cosine/poly decay, or plateau)
- Early Stopping
- Checkpoint Saving (best + latest), self-describing (architecture, classes,
  image size) so evaluation/inference rebuild the right model automatically
- Resume Training from checkpoint (incl. early-stopping and AMP state)
- TensorBoard Logging
- Per-epoch metric tables

Usage:
    # Train U-Net with combined loss
    python train.py --model unet --loss combined --epochs 100

    # Train ResNet-UNet with frozen encoder
    python train.py --model resnet_unet --freeze-encoder --lr 3e-4

    # Resume training
    python train.py --model unet --resume checkpoints/unet_latest.pth

    # Use synthetic data generated on the fly
    python train.py --model unet --epochs 5 --use-mock

Checkpoints are written to checkpoints/<run-name>_best.pth and
checkpoints/<run-name>_latest.pth (run name defaults to the architecture).

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import dataclasses
import json
import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict

import torch
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    NUM_CLASSES,
    Config,
    build_train_parser,
    config_from_args,
)
from dataset import get_dataloaders
from losses import get_loss_function
from metrics import SegmentationMetrics
from model_factory import get_model
from utils import (
    EarlyStopping,
    denormalize_image,
    get_device,
    load_checkpoint,
    plot_training_curves,
    print_model_summary,
    save_checkpoint,
    seed_everything,
    setup_logging,
    visualize_predictions,
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
        use_mock: Use synthetic data generated on the fly.
    """

    def __init__(self, config: Config, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock

        # Setup
        setup_logging(
            log_file=os.path.join(config.paths.training_dir, "training.log")
        )
        seed_everything(config.train.seed, deterministic=config.train.deterministic)
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
        self.best_epoch = -1
        self.history = defaultdict(list)

        # Resume if specified
        if config.train.resume_from:
            self._resume_training(config.train.resume_from)

        logging.info("=" * 60)
        logging.info("  Trainer initialized successfully")
        logging.info(f"  Model      : {config.model.architecture}")
        logging.info(f"  Classes    : {NUM_CLASSES} {CLASS_NAMES}")
        logging.info(f"  Loss       : {config.loss.loss_type}")
        logging.info(f"  Device     : {self.device}")
        logging.info(f"  Epochs     : {config.train.epochs}")
        logging.info(f"  Batch Size : {config.data.batch_size}")
        logging.info(f"  Image/Crop : {config.data.image_size} / {config.data.crop_size}")
        logging.info(f"  LR         : {config.train.learning_rate} "
                     f"({config.train.scheduler}, warmup {self.warmup_epochs})")
        logging.info(f"  Mixed Prec : {self.use_amp}")
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
        """Initialize optimizer (only over trainable parameters)."""
        cfg = self.config.train
        params = [p for p in self.model.parameters() if p.requires_grad]

        if cfg.optimizer.lower() == "adamw":
            self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == "sgd":
            self.optimizer = SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                 weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        logging.info(f"Optimizer: {cfg.optimizer} (lr={cfg.learning_rate})")

    def _setup_scheduler(self) -> None:
        """
        Learning-rate schedule, stepped once per epoch.

        cosine/poly/none: linear warmup to the peak LR, then cosine or
        polynomial decay down to min_lr (or constant for 'none').
        plateau: halve the LR when validation mIoU stops improving.
        """
        cfg = self.config.train
        total = cfg.epochs
        self.warmup_epochs = max(0, min(cfg.warmup_epochs, total - 1))
        min_ratio = cfg.min_lr / cfg.learning_rate
        warmup = self.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            if cfg.scheduler == "none":
                return 1.0
            progress = min(1.0, (epoch - warmup) / max(1, total - warmup - 1))
            if cfg.scheduler == "cosine":
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:  # poly
                decay = (1.0 - progress) ** 0.9
            return min_ratio + (1.0 - min_ratio) * decay

        if cfg.scheduler == "plateau":
            self.warmup_epochs = 0
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=cfg.scheduler_patience, min_lr=cfg.min_lr,
            )
        elif cfg.scheduler in ("cosine", "poly", "none"):
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        logging.info(f"Scheduler: {cfg.scheduler} (warmup {self.warmup_epochs} epochs)")

    def _setup_amp(self) -> None:
        """Initialize Automatic Mixed Precision components."""
        self.use_amp = (
            self.config.train.use_mixed_precision
            and self.device.type == "cuda"
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
        self.writer = SummaryWriter(log_dir=self.config.paths.tensorboard_dir)
        logging.info(f"TensorBoard: {self.config.paths.tensorboard_dir}")

    def _resume_training(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        meta = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )
        self.start_epoch = meta["epoch"] + 1
        self.best_metric = meta["best_metric"]
        self.history = defaultdict(list, meta.get("history", {}))
        if meta.get("early_stopping"):
            self.early_stopping.load_state_dict(meta["early_stopping"])
        logging.info(
            f"Resumed from epoch {self.start_epoch}, "
            f"best mIoU: {self.best_metric:.4f}"
        )

    def _checkpoint_extra(self) -> Dict:
        """Metadata stored in every checkpoint."""
        return {
            "model_config": dataclasses.asdict(self.config.model),
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "image_size": list(self.config.data.image_size),
            "crop_size": list(self.config.data.crop_size),
            "train_config": dataclasses.asdict(self.config.train),
            "loss_config": dataclasses.asdict(self.config.loss),
            "early_stopping": self.early_stopping.state_dict(),
            "best_epoch": self.best_epoch,
        }

    def _save(self, epoch: int, path: str, include_training_state: bool = True) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_metric=self.best_metric,
            history=dict(self.history),
            save_path=path,
            extra=self._checkpoint_extra(),
            include_training_state=include_training_state,
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
        train_start = time.time()

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- Train one epoch ----
            train_metrics = self._train_one_epoch(epoch)

            # ---- Validate ----
            val_metrics = self._validate(epoch)

            # ---- Learning rate scheduling ----
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

            # ---- Early stopping (updated before saving so its state is current) ----
            should_stop = self.early_stopping(val_metrics["mean_iou"])

            # ---- Checkpoint saving ----
            is_best = val_metrics["mean_iou"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["mean_iou"]
                self.best_epoch = epoch
                # Best = weights for inference; latest = full state for resuming
                self._save(epoch, self.config.paths.best_model_path, include_training_state=False)
                logging.info(f"  [NEW BEST] New best model! mIoU: {self.best_metric:.4f}")

            self._save(epoch, self.config.paths.latest_model_path)

            # ---- Epoch summary ----
            elapsed = time.time() - epoch_start
            self._print_epoch_summary(
                epoch, total_epochs, train_metrics, val_metrics,
                current_lr, elapsed, is_best
            )

            # ---- Periodic visualization ----
            if (epoch + 1) % self.config.train.save_every_n_epochs == 0:
                self._save_sample_predictions(epoch)

            if should_stop:
                logging.info(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best mIoU: {self.best_metric:.4f}"
                )
                break

        # ---- Training complete ----
        self.writer.close()

        plot_training_curves(
            history=dict(self.history),
            save_path=self.config.paths.training_dir,
        )
        self._save_summary(time.time() - train_start)

        logging.info(f"\n[OK] Training complete! Best val mIoU: {self.best_metric:.4f}")
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

        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # ---- Forward pass with mixed precision ----
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # ---- Backward pass with gradient scaling ----
            self.optimizer.zero_grad(set_to_none=True)
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
                metrics.update(outputs.argmax(dim=1), masks)

            running_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        summary = metrics.get_summary()
        summary["loss"] = running_loss / max(num_batches, 1)
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

            metrics.update(outputs.argmax(dim=1), masks)

            running_loss += loss.item()
            num_batches += 1

        summary = metrics.get_summary()
        summary["loss"] = running_loss / max(num_batches, 1)

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
        images_list, gt_list, pred_list = [], [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    predictions = self.model(images).argmax(dim=1)

                for i in range(min(4, images.size(0))):
                    images_list.append(denormalize_image(images[i].float().cpu().numpy()))
                    gt_list.append(masks[i].cpu().numpy())
                    pred_list.append(predictions[i].cpu().numpy())

                if len(images_list) >= 4:
                    break

        if images_list:
            save_path = os.path.join(
                self.config.paths.training_dir,
                f"predictions_epoch_{epoch + 1:03d}.png"
            )
            visualize_predictions(images_list, gt_list, pred_list, save_path, num_samples=4)

    def _data_source(self) -> str:
        """'mock' or 'cityscapes', from the prepared dataset's stats file."""
        if self.use_mock:
            return "mock (generated on the fly)"
        stats_path = os.path.join(self.config.data.dataset_root, "dataset_stats.json")
        try:
            with open(stats_path) as f:
                return json.load(f).get("source", "unknown")
        except (OSError, ValueError):
            return "unknown"

    def _save_summary(self, total_seconds: float) -> None:
        """Write a JSON summary of the run next to the training curves."""
        summary = {
            "architecture": self.config.model.architecture,
            "best_val_miou": self.best_metric,
            "best_epoch": self.best_epoch + 1,
            "epochs_run": len(self.history["val_iou"]),
            "train_minutes": round(total_seconds / 60, 2),
            "device": str(self.device),
            "data_source": self._data_source(),
            "dataset_root": self.config.data.dataset_root,
            "image_size": list(self.config.data.image_size),
            "crop_size": list(self.config.data.crop_size),
            "batch_size": self.config.data.batch_size,
            "best_checkpoint": self.config.paths.best_model_path,
        }
        path = os.path.join(self.config.paths.training_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for training."""
    args = build_train_parser().parse_args()
    config = config_from_args(args)

    trainer = Trainer(config, use_mock=args.use_mock)
    trainer.train()

    print("\n[OK] Training completed successfully!")
    print(f"   Best checkpoint: {config.paths.best_model_path}")
    print(f"   Training curves: {config.paths.training_dir}")
    print(f"   TensorBoard:     tensorboard --logdir {config.paths.tensorboard_dir}")


if __name__ == "__main__":
    main()
