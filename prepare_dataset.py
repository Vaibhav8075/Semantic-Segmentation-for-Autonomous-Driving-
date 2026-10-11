"""
prepare_dataset.py — Dataset Preparation Script
================================================

This script handles:
1. Instructions for downloading Cityscapes
2. Reorganizing Cityscapes into the expected folder structure
3. Converting Cityscapes label IDs to our class scheme (the ONLY place
   this conversion happens — the Dataset class reads the result as-is)
4. Computing dataset statistics
5. Generating a synthetic mock dataset for development

Split strategy for Cityscapes (the official test split has no public labels):
    our train = official train minus CITYSCAPES_VAL_CITIES
    our val   = official train, CITYSCAPES_VAL_CITIES only
    our test  = official val

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
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    BACKGROUND_CLASS,
    CITYSCAPES_LABEL_MAP,
    CITYSCAPES_VAL_CITIES,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
)
from mock_data import generate_mock_sample


# ============================================================================
# DOWNLOAD INSTRUCTIONS
# ============================================================================

DOWNLOAD_INSTRUCTIONS = """
================================================================================
  CITYSCAPES DATASET — DOWNLOAD INSTRUCTIONS
================================================================================

Cityscapes is a large-scale driving dataset for semantic segmentation.
It contains 5,000 finely annotated images from 50 European cities.

  Official splits:
    - Train : 2,975 images
    - Val   :   500 images
    - Test  : 1,525 images (labels not publicly available — not used here)

  Steps to Download:

  1. Create a free account at:
     https://www.cityscapes-dataset.com/register/

  2. After email verification, download the following packages:
     - leftImg8bit_trainvaltest.zip  (~11GB) — RGB images
     - gtFine_trainvaltest.zip       (~241MB) — Fine annotations

  3. Extract both archives into a single directory:

     cityscapes/
     ├── leftImg8bit/{train,val,test}/<city>/<name>_leftImg8bit.png
     └── gtFine/{train,val,test}/<city>/<name>_gtFine_labelIds.png

  4. Run this script:
     python prepare_dataset.py --cityscapes-root ./cityscapes --output-root ./dataset

  Alternative (using cityscapesScripts):
     pip install cityscapesscripts
     csDownload -d ./cityscapes leftImg8bit_trainvaltest.zip
     csDownload -d ./cityscapes gtFine_trainvaltest.zip

================================================================================
"""


# ============================================================================
# LABEL CONVERSION
# ============================================================================

def build_cityscapes_lut() -> np.ndarray:
    """
    Lookup table mapping every possible uint8 Cityscapes label ID to our
    class index. IDs not in CITYSCAPES_LABEL_MAP become Background.
    """
    lut = np.full(256, BACKGROUND_CLASS, dtype=np.uint8)
    for src_id, tgt_class in CITYSCAPES_LABEL_MAP.items():
        if 0 <= src_id < 256:
            lut[src_id] = tgt_class
    return lut


_CITYSCAPES_LUT = build_cityscapes_lut()


def remap_cityscapes_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a Cityscapes labelIds mask to our class scheme.

    Args:
        mask: Cityscapes label-ID mask (H, W), uint8.

    Returns:
        Mask (H, W), uint8, values in [0, NUM_CLASSES) or IGNORE_INDEX.
    """
    return _CITYSCAPES_LUT[mask]


# ============================================================================
# CITYSCAPES PREPARATION
# ============================================================================

def _locate(root: str, name: str, max_depth: int = 3) -> Optional[str]:
    """
    Find the `leftImg8bit` / `gtFine` folder under `root`.

    The official zips extract to root/leftImg8bit and root/gtFine, but mirrors
    (e.g. on Kaggle) often nest them, as in root/gtFine_trainvaltest/gtFine.
    """
    for dirpath, dirnames, _ in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if os.path.basename(dirpath) == name and "train" in dirnames:
            return dirpath
        if depth >= max_depth:
            dirnames.clear()
    return None


def _cityscapes_pairs(img_root: str, gt_root: str, split: str,
                      cities: Iterable[str]) -> List[Tuple[str, str, str]]:
    """(image_path, label_path, base_name) for every image in the given cities."""
    pairs = []
    for city in sorted(cities):
        city_img_dir = os.path.join(img_root, split, city)
        city_gt_dir = os.path.join(gt_root, split, city)
        for img_file in sorted(os.listdir(city_img_dir)):
            if not img_file.endswith("_leftImg8bit.png"):
                continue
            base = img_file.replace("_leftImg8bit.png", "")
            gt_path = os.path.join(city_gt_dir, f"{base}_gtFine_labelIds.png")
            if not os.path.exists(gt_path):
                logging.warning(f"Missing mask for: {img_file}")
                continue
            pairs.append((os.path.join(city_img_dir, img_file), gt_path, base))
    return pairs


def prepare_cityscapes(
    cityscapes_root: str,
    output_root: str,
    val_cities: Iterable[str] = CITYSCAPES_VAL_CITIES,
) -> Dict:
    """
    Reorganize Cityscapes into a flat structure and convert the labels.

    Writes:
        dataset/images/{split}/{name}.png
        dataset/masks/{split}/{name}.png
        dataset/dataset_stats.json

    Args:
        cityscapes_root: Root directory of extracted Cityscapes.
        output_root: Output directory for prepared dataset.
        val_cities: Official-train cities held out as our validation split.

    Returns:
        Dictionary with dataset statistics.
    """
    img_root = _locate(cityscapes_root, "leftImg8bit")
    gt_root = _locate(cityscapes_root, "gtFine")

    for path, name, archive in ((img_root, "leftImg8bit", "leftImg8bit_trainvaltest.zip"),
                                (gt_root, "gtFine", "gtFine_trainvaltest.zip")):
        if path is None:
            print(DOWNLOAD_INSTRUCTIONS)
            raise FileNotFoundError(
                f"No '{name}' folder with a train/ split found under: {cityscapes_root}\n"
                f"Please download and extract {archive}"
            )
    print(f"  Images: {img_root}\n  Labels: {gt_root}")

    val_cities = set(val_cities)
    train_cities = sorted(os.listdir(os.path.join(img_root, "train")))
    unknown = val_cities - set(train_cities)
    if unknown:
        raise ValueError(f"Validation cities not in the Cityscapes train split: {sorted(unknown)}")

    plan = {
        "train": _cityscapes_pairs(img_root, gt_root, "train",
                                   [c for c in train_cities if c not in val_cities]),
        "val": _cityscapes_pairs(img_root, gt_root, "train", val_cities),
        "test": _cityscapes_pairs(img_root, gt_root, "val",
                                  os.listdir(os.path.join(img_root, "val"))),
    }

    for split, pairs in plan.items():
        out_img_dir = os.path.join(output_root, "images", split)
        out_mask_dir = os.path.join(output_root, "masks", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        for src_img, gt_path, base in tqdm(pairs, desc=f"Preparing {split}"):
            shutil.copy2(src_img, os.path.join(out_img_dir, f"{base}.png"))
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(out_mask_dir, f"{base}.png"), remap_cityscapes_mask(mask))

        print(f"  {split}: {len(pairs)} image-mask pairs prepared")

    stats = {
        "source": "cityscapes",
        "classes": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "splits": {split: len(pairs) for split, pairs in plan.items()},
        "total": sum(len(p) for p in plan.values()),
        "val_cities": sorted(val_cities),
        "split_note": "train/val = official train split by city; test = official val split",
    }
    _write_stats(output_root, stats)
    print(f"\n[OK] Dataset prepared: {stats['total']} total samples")
    return stats


# ============================================================================
# MOCK DATASET GENERATION
# ============================================================================

def generate_mock_dataset(
    output_root: str,
    num_train: int = 300,
    num_val: int = 100,
    num_test: int = 100,
    image_size: Tuple[int, int] = (512, 1024),
    seed: int = 0,
) -> Dict:
    """
    Write synthetic road scenes (see mock_data.py) to disk.

    Args:
        output_root: Output directory for mock dataset.
        num_train, num_val, num_test: Samples per split.
        image_size: (height, width) of generated images.
        seed: Base random seed (each split/sample gets its own stream).

    Returns:
        Dictionary with dataset statistics.
    """
    splits = {"train": num_train, "val": num_val, "test": num_test}
    h, w = image_size

    for split_idx, (split, num_samples) in enumerate(splits.items()):
        img_dir = os.path.join(output_root, "images", split)
        mask_dir = os.path.join(output_root, "masks", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in tqdm(range(num_samples), desc=f"  Generating {split}"):
            rng = np.random.default_rng([seed, split_idx, i])
            image, mask = generate_mock_sample(h, w, rng)
            name = f"mock_{split}_{i:06d}"
            cv2.imwrite(os.path.join(img_dir, f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, f"{name}.png"), mask)

    stats = {
        "source": "mock",
        "mock": True,
        "classes": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "splits": splits,
        "total": sum(splits.values()),
        "image_size": list(image_size),
        "seed": seed,
    }
    _write_stats(output_root, stats)
    print(f"\n[OK] Mock dataset generated: {stats['total']} total samples at {output_root}")
    return stats


def _write_stats(output_root: str, stats: Dict) -> None:
    os.makedirs(output_root, exist_ok=True)
    stats_path = os.path.join(output_root, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved to: {stats_path}")


# ============================================================================
# DATASET STATISTICS
# ============================================================================

def compute_dataset_statistics(dataset_root: str) -> Dict:
    """
    Compute pixel-level class statistics for the prepared training split.

    Args:
        dataset_root: Root directory of prepared dataset.

    Returns:
        Dictionary with class pixel counts and proportions of labeled pixels.
    """
    counts: Counter = Counter()
    mask_dir = os.path.join(dataset_root, "masks", "train")
    if not os.path.exists(mask_dir):
        logging.warning(f"Training mask directory not found: {mask_dir}")
        return {}

    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    print(f"Computing statistics over {len(mask_files)} training masks...")

    for mask_file in tqdm(mask_files, desc="  Computing stats"):
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        values, value_counts = np.unique(mask, return_counts=True)
        for val, count in zip(values, value_counts):
            counts[int(val)] += int(count)

    unexpected = sorted(v for v in counts if v >= NUM_CLASSES and v != IGNORE_INDEX)
    if unexpected:
        raise ValueError(f"Masks contain values outside the class scheme: {unexpected}")

    labeled = sum(counts[c] for c in range(NUM_CLASSES))
    proportions = {CLASS_NAMES[c]: 100.0 * counts[c] / max(labeled, 1) for c in range(NUM_CLASSES)}

    print("\n  Class Distribution (% of labeled pixels):")
    for name, prop in proportions.items():
        print(f"    {name:15s}: {prop:6.2f}%")
    total = labeled + counts[IGNORE_INDEX]
    print(f"    {'(ignored)':15s}: {100.0 * counts[IGNORE_INDEX] / max(total, 1):6.2f}% of all pixels")

    return {"class_pixel_counts": dict(counts), "class_proportions": proportions}


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
    parser.add_argument("--val-cities", type=str, nargs="+", default=list(CITYSCAPES_VAL_CITIES),
                        help="Cityscapes train cities held out for validation")
    parser.add_argument("--mock", action="store_true",
                        help="Generate mock dataset instead of processing Cityscapes")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Total number of mock samples, split 60/20/20 (mock mode only)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 1024],
                        help="Image size for mock dataset (H W)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the mock dataset")
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
        n = args.num_samples
        n_val = n_test = int(n * 0.2)
        generate_mock_dataset(
            output_root=args.output_root,
            num_train=n - n_val - n_test,
            num_val=n_val,
            num_test=n_test,
            image_size=tuple(args.image_size),
            seed=args.seed,
        )
    else:
        prepare_cityscapes(
            cityscapes_root=args.cityscapes_root,
            output_root=args.output_root,
            val_cities=args.val_cities,
        )

    compute_dataset_statistics(args.output_root)


if __name__ == "__main__":
    main()
