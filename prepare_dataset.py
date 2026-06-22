"""
prepare_dataset.py — Cityscapes Dataset Preparation Script
==========================================================

This script handles:
1. Instructions for downloading Cityscapes
2. Reorganizing Cityscapes into the expected folder structure
3. Remapping label masks to our 7-class scheme
4. Computing dataset statistics
5. Generating a mock dataset for development

Usage:
    # Prepare Cityscapes (after download)
    python prepare_dataset.py --cityscapes-root ./cityscapes --output-root ./dataset

    # Generate mock dataset for development
    python prepare_dataset.py --mock --output-root ./dataset --num-samples 500

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    CITYSCAPES_LABEL_MAP,
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
)


# ============================================================================
# DOWNLOAD INSTRUCTIONS
# ============================================================================

DOWNLOAD_INSTRUCTIONS = """
================================================================================
  CITYSCAPES DATASET — DOWNLOAD INSTRUCTIONS
================================================================================

Cityscapes is a large-scale driving dataset for semantic segmentation.
It contains 5,000 finely annotated images from 50 European cities.

  Splits:
    - Train : 2,975 images
    - Val   :   500 images
    - Test  : 1,525 images (labels not publicly available)

  Steps to Download:

  1. Create a free account at:
     https://www.cityscapes-dataset.com/register/

  2. After email verification, download the following packages:
     - leftImg8bit_trainvaltest.zip  (~11GB) — RGB images
     - gtFine_trainvaltest.zip       (~241MB) — Fine annotations

  3. Extract both archives into a single directory:

     cityscapes/
     ├── leftImg8bit/
     │   ├── train/
     │   │   ├── aachen/
     │   │   │   ├── aachen_000000_000019_leftImg8bit.png
     │   │   │   └── ...
     │   │   ├── bochum/
     │   │   └── ...
     │   ├── val/
     │   └── test/
     └── gtFine/
         ├── train/
         │   ├── aachen/
         │   │   ├── aachen_000000_000019_gtFine_labelIds.png
         │   │   ├── aachen_000000_000019_gtFine_color.png
         │   │   └── ...
         │   └── ...
         ├── val/
         └── test/

  4. Run this script:
     python prepare_dataset.py --cityscapes-root ./cityscapes --output-root ./dataset

  Alternative (using cityscapesScripts):
     pip install cityscapesscripts
     export CITYSCAPES_DATASET=./cityscapes
     csDownload -d ./cityscapes leftImg8bit_trainvaltest.zip
     csDownload -d ./cityscapes gtFine_trainvaltest.zip

================================================================================
"""


# ============================================================================
# CITYSCAPES PREPARATION
# ============================================================================

def prepare_cityscapes(
    cityscapes_root: str,
    output_root: str,
    remap_labels: bool = True,
) -> Dict:
    """
    Reorganize Cityscapes dataset into a flat structure and remap labels.

    Converts:
        cityscapes/leftImg8bit/{split}/{city}/{name}_leftImg8bit.png
        cityscapes/gtFine/{split}/{city}/{name}_gtFine_labelIds.png

    To:
        dataset/images/{split}/{name}.png
        dataset/masks/{split}/{name}.png

    Args:
        cityscapes_root: Root directory of extracted Cityscapes.
        output_root: Output directory for prepared dataset.
        remap_labels: If True, remap Cityscapes labels to 7-class scheme.

    Returns:
        Dictionary with dataset statistics.
    """
    print(DOWNLOAD_INSTRUCTIONS)

    # Validate Cityscapes structure
    img_root = os.path.join(cityscapes_root, "leftImg8bit")
    gt_root = os.path.join(cityscapes_root, "gtFine")

    if not os.path.exists(img_root):
        raise FileNotFoundError(
            f"Cityscapes images not found at: {img_root}\n"
            f"Please download and extract leftImg8bit_trainvaltest.zip"
        )

    if not os.path.exists(gt_root):
        raise FileNotFoundError(
            f"Cityscapes annotations not found at: {gt_root}\n"
            f"Please download and extract gtFine_trainvaltest.zip"
        )

    stats = {"total": 0, "splits": {}}

    for split in ["train", "val", "test"]:
        split_img_dir = os.path.join(img_root, split)
        split_gt_dir = os.path.join(gt_root, split)

        if not os.path.exists(split_img_dir):
            logging.warning(f"Split '{split}' not found, skipping.")
            continue

        # Create output directories
        out_img_dir = os.path.join(output_root, "images", split)
        out_mask_dir = os.path.join(output_root, "masks", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        count = 0

        # Iterate through city subdirectories
        cities = sorted(os.listdir(split_img_dir))
        for city in tqdm(cities, desc=f"Processing {split}"):
            city_img_dir = os.path.join(split_img_dir, city)
            city_gt_dir = os.path.join(split_gt_dir, city)

            if not os.path.isdir(city_img_dir):
                continue

            # Find all images in this city
            for img_file in sorted(os.listdir(city_img_dir)):
                if not img_file.endswith("_leftImg8bit.png"):
                    continue

                # Construct corresponding mask filename
                base = img_file.replace("_leftImg8bit.png", "")
                gt_file = f"{base}_gtFine_labelIds.png"
                gt_path = os.path.join(city_gt_dir, gt_file)

                if not os.path.exists(gt_path):
                    logging.warning(f"Missing mask for: {img_file}")
                    continue

                # Copy image
                src_img = os.path.join(city_img_dir, img_file)
                dst_img = os.path.join(out_img_dir, f"{base}.png")
                shutil.copy2(src_img, dst_img)

                # Process mask
                mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                if remap_labels:
                    mask = remap_cityscapes_mask(mask)

                dst_mask = os.path.join(out_mask_dir, f"{base}.png")
                cv2.imwrite(dst_mask, mask.astype(np.uint8))

                count += 1

        stats["splits"][split] = count
        stats["total"] += count
        print(f"  {split}: {count} image-mask pairs prepared")

    # Save statistics
    stats_path = os.path.join(output_root, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[OK] Dataset prepared: {stats['total']} total samples")
    print(f"  Statistics saved to: {stats_path}")

    return stats


def remap_cityscapes_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap a Cityscapes label mask to our 7-class scheme.

    Args:
        mask: Original Cityscapes label mask (H, W).

    Returns:
        Remapped mask with values in [0, 6].
    """
    remapped = np.full_like(mask, fill_value=NUM_CLASSES - 1, dtype=np.uint8)

    for src_id, tgt_class in CITYSCAPES_LABEL_MAP.items():
        if src_id >= 0:  # Skip negative IDs for uint8 masks
            remapped[mask == src_id] = tgt_class

    return remapped


# ============================================================================
# MOCK DATASET GENERATION
# ============================================================================

def generate_mock_dataset(
    output_root: str,
    num_train: int = 300,
    num_val: int = 50,
    num_test: int = 50,
    image_size: Tuple[int, int] = (1024, 2048),
) -> None:
    """
    Generate a mock dataset for development without Cityscapes access.

    Creates synthetic road-scene images with plausible segmentation masks.

    Args:
        output_root: Output directory for mock dataset.
        num_train: Number of training samples.
        num_val: Number of validation samples.
        num_test: Number of test samples.
        image_size: (height, width) of generated images.
    """
    splits = {"train": num_train, "val": num_val, "test": num_test}

    for split, num_samples in splits.items():
        img_dir = os.path.join(output_root, "images", split)
        mask_dir = os.path.join(output_root, "masks", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        print(f"Generating {num_samples} mock {split} samples...")

        for i in tqdm(range(num_samples), desc=f"  {split}"):
            h, w = image_size
            mask = _generate_road_mask(h, w)

            # Create corresponding image from mask + noise
            image = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id, color in CLASS_COLORS.items():
                image[mask == class_id] = color

            # Add variation: noise, slight blur, brightness variation
            noise = np.random.randint(0, 40, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)

            brightness = np.random.uniform(0.7, 1.3)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)

            if np.random.random() > 0.5:
                image = cv2.GaussianBlur(image, (3, 3), 0)

            # Save
            name = f"mock_{split}_{i:06d}"
            cv2.imwrite(os.path.join(img_dir, f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, f"{name}.png"), mask.astype(np.uint8))

    # Save statistics
    total = sum(splits.values())
    stats = {"total": total, "splits": splits, "mock": True, "image_size": list(image_size)}
    with open(os.path.join(output_root, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[OK] Mock dataset generated: {total} total samples at {output_root}")


def _generate_road_mask(h: int, w: int) -> np.ndarray:
    """Generate a synthetic road scene mask with realistic layout."""
    mask = np.full((h, w), fill_value=6, dtype=np.int64)  # Background

    # Horizon line
    horizon = int(h * np.random.uniform(0.35, 0.45))

    # Road (trapezoidal shape)
    road_top_width = np.random.uniform(0.15, 0.25)
    for y in range(horizon, h):
        progress = (y - horizon) / max(1, h - horizon)
        road_width = int(w * (road_top_width + (1.0 - road_top_width) * progress ** 0.8))
        offset = np.random.randint(-10, 10)
        left = (w - road_width) // 2 + offset
        right = left + road_width
        left = max(0, left)
        right = min(w, right)
        mask[y, left:right] = 0  # Road

    # Lane markings (dashed center line)
    center = w // 2
    lane_width = max(2, int(w * 0.003))
    dash_len = max(5, int(h * 0.02))
    gap_len = max(3, int(h * 0.015))

    for y in range(horizon, h, dash_len + gap_len):
        y_end = min(y + dash_len, h)
        lw = int(lane_width * (1 + (y - horizon) / max(1, h - horizon)))
        mask[y:y_end, center - lw:center + lw] = 1  # Lane marking

    # Side lane markings
    for y in range(horizon, h):
        progress = (y - horizon) / max(1, h - horizon)
        road_width = int(w * (road_top_width + (1.0 - road_top_width) * progress ** 0.8))
        left = (w - road_width) // 2
        right = left + road_width
        lw = max(1, int(lane_width * (0.5 + progress)))
        mask[y, max(0, left):left + lw] = 1
        mask[y, right - lw:min(w, right)] = 1

    # Sidewalks
    sidewalk_width_ratio = np.random.uniform(0.03, 0.06)
    for y in range(horizon, h):
        progress = (y - horizon) / max(1, h - horizon)
        road_width = int(w * (road_top_width + (1.0 - road_top_width) * progress ** 0.8))
        left = (w - road_width) // 2
        right = left + road_width
        sw = int(w * sidewalk_width_ratio * progress) + 3
        mask[y, max(0, left - sw):left] = 4
        mask[y, right:min(w, right + sw)] = 4

    # Vehicles
    num_vehicles = np.random.randint(0, 6)
    for _ in range(num_vehicles):
        vy = np.random.randint(horizon + 20, h - 60)
        progress = (vy - horizon) / max(1, h - horizon)
        scale = 0.3 + 0.7 * progress
        vh = int(np.random.randint(40, 80) * scale)
        vw = int(np.random.randint(30, 50) * scale)
        road_width = int(w * (road_top_width + (1.0 - road_top_width) * progress ** 0.8))
        left_road = (w - road_width) // 2
        vx = np.random.randint(left_road + 10, left_road + road_width - vw - 10)
        vx = max(0, min(vx, w - vw))
        mask[vy:min(vy + vh, h), vx:min(vx + vw, w)] = 2

    # Pedestrians
    num_peds = np.random.randint(0, 3)
    for _ in range(num_peds):
        py = np.random.randint(horizon + 30, h - 40)
        progress = (py - horizon) / max(1, h - horizon)
        ph = int(np.random.randint(30, 50) * (0.4 + 0.6 * progress))
        pw = int(ph * 0.4)
        px = np.random.randint(10, w - pw - 10)
        mask[py:min(py + ph, h), px:min(px + pw, w)] = 3

    # Traffic signs
    num_signs = np.random.randint(0, 3)
    for _ in range(num_signs):
        ty = np.random.randint(int(horizon * 0.5), horizon + 30)
        tx = np.random.randint(10, w - 30)
        ts = np.random.randint(10, 30)
        mask[ty:min(ty + ts, h), tx:min(tx + ts, w)] = 5

    return mask


# ============================================================================
# DATASET STATISTICS
# ============================================================================

def compute_dataset_statistics(dataset_root: str) -> Dict:
    """
    Compute pixel-level statistics for the prepared dataset.

    Args:
        dataset_root: Root directory of prepared dataset.

    Returns:
        Dictionary with class pixel counts and proportions.
    """
    stats = {
        "class_pixel_counts": Counter(),
        "total_pixels": 0,
    }

    mask_dir = os.path.join(dataset_root, "masks", "train")
    if not os.path.exists(mask_dir):
        logging.warning(f"Training mask directory not found: {mask_dir}")
        return stats

    mask_files = sorted([
        f for f in os.listdir(mask_dir) if f.endswith(".png")
    ])

    print(f"Computing statistics over {len(mask_files)} training masks...")

    for mask_file in tqdm(mask_files, desc="  Computing stats"):
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, counts):
            stats["class_pixel_counts"][int(val)] += int(count)
        stats["total_pixels"] += mask.size

    # Compute proportions
    stats["class_proportions"] = {
        CLASS_NAMES[k]: v / stats["total_pixels"] * 100
        for k, v in stats["class_pixel_counts"].items()
        if k < NUM_CLASSES
    }

    print("\n  Class Distribution:")
    for name, prop in stats["class_proportions"].items():
        print(f"    {name:15s}: {prop:6.2f}%")

    return stats


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for semantic segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DOWNLOAD_INSTRUCTIONS,
    )

    parser.add_argument("--cityscapes-root", type=str, default="./cityscapes",
                        help="Root directory of raw Cityscapes dataset")
    parser.add_argument("--output-root", type=str, default="./dataset",
                        help="Output directory for prepared dataset")
    parser.add_argument("--mock", action="store_true",
                        help="Generate mock dataset instead of processing Cityscapes")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of mock samples per split (mock mode only)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 1024],
                        help="Image size for mock dataset (H W)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics for existing dataset")
    parser.add_argument("--show-instructions", action="store_true",
                        help="Print download instructions and exit")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.show_instructions:
        print(DOWNLOAD_INSTRUCTIONS)
        return

    if args.stats_only:
        compute_dataset_statistics(args.output_root)
        return

    if args.mock:
        # Calculate split sizes (60/20/20)
        n = args.num_samples
        generate_mock_dataset(
            output_root=args.output_root,
            num_train=int(n * 0.6),
            num_val=int(n * 0.2),
            num_test=int(n * 0.2),
            image_size=tuple(args.image_size),
        )
    else:
        prepare_cityscapes(
            cityscapes_root=args.cityscapes_root,
            output_root=args.output_root,
        )

    # Compute statistics after preparation
    compute_dataset_statistics(args.output_root)


if __name__ == "__main__":
    main()
