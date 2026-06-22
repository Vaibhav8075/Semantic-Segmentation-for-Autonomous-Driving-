"""
eda.py — Exploratory Data Analysis for Semantic Segmentation
==============================================================

Generates comprehensive visualizations for understanding the dataset:
1. Class distribution (bar chart of pixel counts per class)
2. Pixel intensity distribution (per-channel histograms)
3. Sample image visualization (grid of training images)
4. Sample mask visualization (color-coded class masks)
5. Overlay visualization (semi-transparent mask on image)
6. Class imbalance analysis (pie chart of proportions)

All plots are saved automatically to outputs/eda/

Usage:
    python eda.py --dataset-root ./dataset
    python eda.py --dataset-root ./dataset --use-mock

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
import seaborn as sns
from tqdm import tqdm

from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
    Config,
    DataConfig,
    PathConfig,
)
from dataset import CityscapesSegDataset, MockSegDataset
from utils import colorize_mask, create_overlay, setup_logging


# ============================================================================
# EDA FUNCTIONS
# ============================================================================

def analyze_class_distribution(
    dataset,
    save_dir: str,
    max_samples: int = 500,
) -> Dict[str, int]:
    """
    Compute and plot class pixel distribution across the dataset.

    Args:
        dataset: Segmentation dataset with get_raw_sample method.
        save_dir: Directory to save the plot.
        max_samples: Maximum number of samples to analyze.

    Returns:
        Dictionary of class name → total pixel count.
    """
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    num_samples = min(max_samples, len(dataset))

    print(f"\nAnalyzing class distribution ({num_samples} samples)...")

    for idx in tqdm(range(num_samples), desc="  Counting pixels"):
        _, mask = dataset.get_raw_sample(idx)
        for c in range(NUM_CLASSES):
            class_counts[c] += np.sum(mask == c)

    total_pixels = class_counts.sum()

    # ---- Bar Chart ----
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(NUM_CLASSES)]
    bars = ax.bar(CLASS_NAMES, class_counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add percentage labels on bars
    for bar, count in zip(bars, class_counts):
        pct = count / total_pixels * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_pixels * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Pixel Count", fontsize=13)
    ax.set_title("Class Pixel Distribution (Training Set)", fontsize=15, fontweight="bold")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150)
    plt.close()

    # ---- Log Scale Bar Chart ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(CLASS_NAMES, class_counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Pixel Count (log scale)", fontsize=13)
    ax.set_title("Class Distribution — Log Scale (Reveals Imbalance)", fontsize=15, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution_log.png"), dpi=150)
    plt.close()

    # Print stats
    print(f"\n  {'Class':15s} | {'Pixels':>12s} | {'Proportion':>10s}")
    print(f"  {'-' * 15}-+-{'-' * 12}-+-{'-' * 10}")
    results = {}
    for i, name in enumerate(CLASS_NAMES):
        pct = class_counts[i] / total_pixels * 100
        print(f"  {name:15s} | {class_counts[i]:>12,} | {pct:>9.2f}%")
        results[name] = int(class_counts[i])

    return results


def analyze_pixel_distribution(
    dataset,
    save_dir: str,
    max_samples: int = 100,
) -> None:
    """
    Plot pixel intensity distributions per RGB channel.

    Args:
        dataset: Segmentation dataset.
        save_dir: Directory to save the plot.
        max_samples: Number of samples to analyze.
    """
    print(f"\nAnalyzing pixel intensity distribution...")

    all_r, all_g, all_b = [], [], []
    num_samples = min(max_samples, len(dataset))

    for idx in tqdm(range(num_samples), desc="  Collecting pixels"):
        image, _ = dataset.get_raw_sample(idx)
        # Subsample pixels for efficiency
        h, w = image.shape[:2]
        step = max(1, h * w // 10000)
        pixels = image.reshape(-1, 3)[::step]
        all_r.extend(pixels[:, 0].tolist())
        all_g.extend(pixels[:, 1].tolist())
        all_b.extend(pixels[:, 2].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channels = [("Red", all_r, "red"), ("Green", all_g, "green"), ("Blue", all_b, "blue")]

    for ax, (name, data, color) in zip(axes, channels):
        ax.hist(data, bins=64, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Pixel Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{name} Channel", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 255)
        ax.axvline(np.mean(data), color="black", linestyle="--", label=f"Mean: {np.mean(data):.1f}")
        ax.legend()

    plt.suptitle("Pixel Intensity Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pixel_distribution.png"), dpi=150)
    plt.close()


def visualize_samples(
    dataset,
    save_dir: str,
    num_samples: int = 16,
    grid_cols: int = 4,
) -> None:
    """
    Visualize sample images in a grid.

    Args:
        dataset: Segmentation dataset.
        save_dir: Directory to save the plot.
        num_samples: Number of samples to show.
        grid_cols: Number of columns in the grid.
    """
    print(f"\nVisualizing {num_samples} sample images...")

    num_samples = min(num_samples, len(dataset))
    grid_rows = (num_samples + grid_cols - 1) // grid_cols

    # ---- Sample Images ----
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for ax_idx, data_idx in enumerate(indices):
        image, _ = dataset.get_raw_sample(data_idx)
        axes[ax_idx].imshow(image)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(f"Sample {data_idx}", fontsize=10)

    # Hide unused axes
    for ax in axes[num_samples:]:
        ax.axis("off")

    plt.suptitle("Sample Training Images", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_images.png"), dpi=150)
    plt.close()

    # ---- Sample Masks ----
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    for ax_idx, data_idx in enumerate(indices):
        _, mask = dataset.get_raw_sample(data_idx)
        colored_mask = colorize_mask(mask)
        axes[ax_idx].imshow(colored_mask)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(f"Mask {data_idx}", fontsize=10)

    for ax in axes[num_samples:]:
        ax.axis("off")

    # Add legend
    legend_elements = []
    for i, name in enumerate(CLASS_NAMES):
        color = np.array(CLASS_COLORS[i]) / 255.0
        legend_elements.append(
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                       markersize=10, label=name)
        )
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(CLASS_NAMES),
               fontsize=9, frameon=True)

    plt.suptitle("Sample Segmentation Masks", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "sample_masks.png"), dpi=150)
    plt.close()


def visualize_overlays(
    dataset,
    save_dir: str,
    num_samples: int = 8,
) -> None:
    """
    Visualize image-mask overlays in a side-by-side layout.

    Shows: Original Image | Colored Mask | Overlay

    Args:
        dataset: Segmentation dataset.
        save_dir: Directory to save the plot.
        num_samples: Number of samples to show.
    """
    print(f"\nGenerating overlay visualizations...")

    num_samples = min(num_samples, len(dataset))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("Segmentation Mask", fontsize=14, fontweight="bold")
    axes[0, 2].set_title("Overlay", fontsize=14, fontweight="bold")

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for row, data_idx in enumerate(indices):
        image, mask = dataset.get_raw_sample(data_idx)

        # Resize for consistent display
        display_size = (512, 512)
        image_resized = cv2.resize(image, display_size)
        mask_resized = cv2.resize(mask.astype(np.uint8), display_size, interpolation=cv2.INTER_NEAREST)

        colored_mask = colorize_mask(mask_resized)
        overlay = create_overlay(image_resized, mask_resized, alpha=0.4)

        axes[row, 0].imshow(image_resized)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colored_mask)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

    plt.suptitle("Image, Mask, and Overlay Visualization", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overlay_visualization.png"), dpi=150)
    plt.close()


def plot_class_imbalance(class_counts: Dict[str, int], save_dir: str) -> None:
    """
    Create a pie chart showing class proportions.

    Args:
        class_counts: Dictionary of class name → pixel count.
        save_dir: Directory to save the plot.
    """
    print(f"\nPlotting class imbalance analysis...")

    names = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(len(names))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart
    explode = [0.05] * len(names)
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=names,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax1.set_title("Class Proportions", fontsize=14, fontweight="bold")

    # Imbalance ratio bar chart
    total = sum(counts)
    proportions = [c / total for c in counts]
    ideal = 1.0 / len(names)
    imbalance_ratios = [p / ideal for p in proportions]

    bars = ax2.barh(names, imbalance_ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Balanced (1.0)")
    ax2.set_xlabel("Imbalance Ratio (vs. Uniform)", fontsize=12)
    ax2.set_title("Class Imbalance Analysis", fontsize=14, fontweight="bold")
    ax2.legend()

    for bar, ratio in zip(bars, imbalance_ratios):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{ratio:.2f}×", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_imbalance.png"), dpi=150)
    plt.close()


# ============================================================================
# MAIN EDA PIPELINE
# ============================================================================

def run_eda(
    dataset_root: str = "./dataset",
    save_dir: str = "./outputs/eda",
    use_mock: bool = False,
    max_samples: int = 500,
) -> None:
    """
    Run the complete EDA pipeline.

    Args:
        dataset_root: Root directory of the dataset.
        save_dir: Directory to save all EDA outputs.
        use_mock: Use mock dataset.
        max_samples: Maximum samples for statistics computation.
    """
    os.makedirs(save_dir, exist_ok=True)
    setup_logging()

    print("\n" + "=" * 60)
    print("  Exploratory Data Analysis")
    print("=" * 60)

    # Load dataset (raw, no augmentation)
    if use_mock:
        dataset = MockSegDataset(num_samples=200, image_size=(512, 512))
        print("  Using mock dataset (200 samples)")
    else:
        dataset = CityscapesSegDataset(
            root_dir=dataset_root,
            split="train",
            transform=None,
        )
        print(f"  Using Cityscapes from: {dataset_root}")
        print(f"  Total training samples: {len(dataset)}")

    if len(dataset) == 0:
        print("\n  [WARNING] No data found! Use --use-mock flag for development.")
        return

    # Run analyses
    class_counts = analyze_class_distribution(dataset, save_dir, max_samples)
    analyze_pixel_distribution(dataset, save_dir, max_samples=min(100, len(dataset)))
    visualize_samples(dataset, save_dir, num_samples=16)
    visualize_overlays(dataset, save_dir, num_samples=8)
    plot_class_imbalance(class_counts, save_dir)

    print(f"\n[OK] EDA complete! All plots saved to: {save_dir}")
    print("=" * 60)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Exploratory Data Analysis on the segmentation dataset"
    )
    parser.add_argument("--dataset-root", type=str, default="./dataset",
                        help="Root directory of the prepared dataset")
    parser.add_argument("--save-dir", type=str, default="./outputs/eda",
                        help="Directory to save EDA outputs")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock dataset for development")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples for statistics")

    args = parser.parse_args()
    run_eda(args.dataset_root, args.save_dir, args.use_mock, args.max_samples)
