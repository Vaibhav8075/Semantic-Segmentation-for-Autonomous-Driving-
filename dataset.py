"""
dataset.py — Dataset Loading & Augmentation Pipeline
=====================================================

Implements a PyTorch Dataset for the Cityscapes semantic segmentation task,
remapping the original 30+ classes to our 7-class scheme for autonomous driving.

Includes Albumentations-based augmentation pipelines for training and validation.

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from config import (
    CITYSCAPES_LABEL_MAP,
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    Config,
    DataConfig,
)


# ============================================================================
# AUGMENTATION PIPELINES
# ============================================================================

def get_training_augmentations(height: int = 512, width: int = 512) -> A.Compose:
    """
    Training augmentation pipeline using Albumentations.

    Includes spatial and photometric augmentations designed for driving scenes:
    - Horizontal flip (roads are symmetric)
    - Random brightness/contrast (varying lighting conditions)
    - Random rotation (minor camera tilt)
    - Random crop (focus on different regions)
    - Gaussian blur (simulate varying focus/weather)

    Args:
        height: Target image height.
        width: Target image width.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.0),  # Disabled for driving (road orientation matters)
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.RandomCrop(height=height, width=width, p=1.0),

        # Photometric augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3,
        ),

        # Weather-like augmentations
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.1),
        A.RandomShadow(p=0.2),

        # Normalize and convert
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_validation_augmentations(height: int = 512, width: int = 512) -> A.Compose:
    """
    Validation/test augmentation pipeline — resize and normalize only.

    Args:
        height: Target image height.
        width: Target image width.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================================
# CITYSCAPES DATASET
# ============================================================================

class CityscapesSegDataset(Dataset):
    """
    PyTorch Dataset for Cityscapes semantic segmentation.

    Handles loading images and corresponding annotation masks from the
    prepared dataset directory. Masks are remapped from Cityscapes' native
    label IDs to our 7-class scheme.

    Expected directory structure:
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── masks/
            ├── train/
            ├── val/
            └── test/

    Args:
        root_dir: Root directory of the prepared dataset.
        split: One of 'train', 'val', 'test'.
        transform: Albumentations augmentation pipeline.
        label_map: Mapping from Cityscapes label IDs to target classes.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        label_map: Optional[Dict[int, int]] = None,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_map = label_map or CITYSCAPES_LABEL_MAP

        # Setup paths
        self.image_dir = os.path.join(root_dir, "images", split)
        self.mask_dir = os.path.join(root_dir, "masks", split)

        # Discover image-mask pairs
        self.image_files: List[str] = []
        self.mask_files: List[str] = []
        self._load_file_pairs()

        logging.info(
            f"CityscapesSegDataset [{split}]: "
            f"{len(self)} samples loaded from {root_dir}"
        )

    def _load_file_pairs(self) -> None:
        """Discover and pair image files with their corresponding masks."""
        if not os.path.exists(self.image_dir):
            logging.warning(f"Image directory not found: {self.image_dir}")
            return

        # Get sorted image files
        image_extensions = {".png", ".jpg", ".jpeg"}
        images = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        for img_name in images:
            img_path = os.path.join(self.image_dir, img_name)

            # Try to find corresponding mask
            # Cityscapes naming: {city}_{seq}_{frame}_leftImg8bit.png
            # Mask naming: {city}_{seq}_{frame}_gtFine_labelIds.png
            base_name = os.path.splitext(img_name)[0]

            # Try different mask naming conventions
            mask_candidates = [
                base_name + ".png",
                base_name.replace("_leftImg8bit", "_gtFine_labelIds") + ".png",
                base_name.replace("_leftImg8bit", "_mask") + ".png",
            ]

            mask_path = None
            for candidate in mask_candidates:
                candidate_path = os.path.join(self.mask_dir, candidate)
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break

            if mask_path:
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)
            else:
                logging.warning(f"No mask found for image: {img_name}")

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remap Cityscapes label IDs to our 7-class scheme.

        Args:
            mask: Original mask with Cityscapes label IDs.

        Returns:
            Remapped mask with values in [0, NUM_CLASSES-1].
        """
        remapped = np.full_like(mask, fill_value=NUM_CLASSES - 1, dtype=np.int64)

        for src_label, tgt_class in self.label_map.items():
            remapped[mask == src_label] = tgt_class

        return remapped

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single (image, mask) pair.

        Returns:
            image: Tensor of shape (3, H, W), normalized.
            mask: Tensor of shape (H, W), int64 class indices.
        """
        # Load image (BGR → RGB)
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)

        # Handle 16-bit masks (Cityscapes labelIds can be 8-bit or 16-bit)
        if mask.dtype == np.uint16:
            mask = mask.astype(np.int32)

        # Remap to 7 classes
        mask = self._remap_mask(mask)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a raw (un-augmented) sample for visualization.

        Returns:
            image: RGB numpy array (H, W, 3) as uint8.
            mask: Remapped mask array (H, W) as int64.
        """
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask.dtype == np.uint16:
            mask = mask.astype(np.int32)
        mask = self._remap_mask(mask)

        return image, mask


# ============================================================================
# MOCK DATASET (for development without Cityscapes)
# ============================================================================

class MockSegDataset(Dataset):
    """
    Mock dataset for development and testing without Cityscapes access.

    Generates random images and plausible masks for pipeline validation.

    Args:
        num_samples: Number of mock samples to generate.
        image_size: (height, width) of generated images.
        transform: Albumentations augmentation pipeline.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[A.Compose] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        logging.info(f"MockSegDataset: {num_samples} synthetic samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic image-mask pair."""
        h, w = self.image_size

        # Create a plausible road scene mask
        mask = self._generate_road_mask(h, w)

        # Create a corresponding image (colored by mask + noise)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            image[mask == class_id] = color

        # Add realistic noise and variation
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        # Apply transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get an un-augmented mock sample."""
        h, w = self.image_size
        mask = self._generate_road_mask(h, w)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            image[mask == class_id] = color
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        return image, mask

    @staticmethod
    def _generate_road_mask(h: int, w: int) -> np.ndarray:
        """Generate a synthetic road scene segmentation mask."""
        mask = np.full((h, w), fill_value=6, dtype=np.int64)  # Background

        # Sky (top portion)
        sky_line = h // 3 + np.random.randint(-20, 20)
        mask[:sky_line, :] = 6  # Background

        # Road (bottom trapezoid)
        road_top = h // 2 + np.random.randint(-20, 20)
        for y in range(road_top, h):
            progress = (y - road_top) / max(1, h - road_top)
            road_width = int(w * (0.3 + 0.7 * progress))
            left = (w - road_width) // 2
            right = left + road_width
            mask[y, left:right] = 0  # Road

        # Lane markings (center dashed line)
        center = w // 2
        for y in range(road_top, h, 20):
            y_end = min(y + 10, h)
            mask[y:y_end, center - 3:center + 3] = 1  # Lane marking

        # Sidewalks (edges of road)
        for y in range(road_top, h):
            progress = (y - road_top) / max(1, h - road_top)
            road_width = int(w * (0.3 + 0.7 * progress))
            left = (w - road_width) // 2
            right = left + road_width
            sw = int(20 * progress) + 5
            mask[y, max(0, left - sw):left] = 4  # Left sidewalk
            mask[y, right:min(w, right + sw)] = 4  # Right sidewalk

        # Vehicles (random rectangles on road)
        num_vehicles = np.random.randint(1, 4)
        for _ in range(num_vehicles):
            vy = np.random.randint(road_top + 20, h - 40)
            vx = np.random.randint(w // 4, 3 * w // 4)
            vh, vw = np.random.randint(30, 60), np.random.randint(20, 40)
            mask[vy:min(vy + vh, h), vx:min(vx + vw, w)] = 2  # Vehicle

        # Pedestrians (small rectangles on sidewalk)
        if np.random.random() > 0.5:
            py = np.random.randint(road_top + 10, h - 30)
            px = np.random.randint(10, w // 4)
            mask[py:py + 25, px:px + 10] = 3  # Pedestrian

        # Traffic signs (small squares in upper half)
        if np.random.random() > 0.6:
            ty = np.random.randint(sky_line, road_top)
            tx = np.random.randint(10, w - 30)
            mask[ty:ty + 20, tx:tx + 15] = 5  # Traffic sign

        return mask


# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def get_dataloaders(
    config: Config,
    use_mock: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        config: Project configuration.
        use_mock: If True, use MockSegDataset instead of Cityscapes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    h, w = config.data.image_height, config.data.image_width

    # Augmentation pipelines
    train_transform = get_training_augmentations(h, w) if config.data.use_augmentation else get_validation_augmentations(h, w)
    val_transform = get_validation_augmentations(h, w)

    if use_mock:
        train_dataset = MockSegDataset(num_samples=200, image_size=(h, w), transform=train_transform)
        val_dataset = MockSegDataset(num_samples=50, image_size=(h, w), transform=val_transform)
        test_dataset = MockSegDataset(num_samples=50, image_size=(h, w), transform=val_transform)
    else:
        train_dataset = CityscapesSegDataset(
            root_dir=config.data.dataset_root,
            split="train",
            transform=train_transform,
        )
        val_dataset = CityscapesSegDataset(
            root_dir=config.data.dataset_root,
            split="val",
            transform=val_transform,
        )
        test_dataset = CityscapesSegDataset(
            root_dir=config.data.dataset_root,
            split="test",
            transform=val_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    logging.info(
        f"DataLoaders created — "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# ENTRY POINT — Dataset Statistics
# ============================================================================

if __name__ == "__main__":
    """Print dataset statistics when run directly."""
    from config import Config

    config = Config()
    print("\n" + "=" * 60)
    print("  Dataset Configuration")
    print("=" * 60)
    print(f"  Root       : {config.data.dataset_root}")
    print(f"  Image Size : {config.data.image_size}")
    print(f"  Batch Size : {config.data.batch_size}")
    print(f"  Classes    : {NUM_CLASSES}")
    print(f"  Class Names: {CLASS_NAMES}")
    print(f"  Colors     : {CLASS_COLORS}")
    print("=" * 60)

    # Test with mock dataset
    print("\nTesting with MockSegDataset...")
    train_loader, val_loader, test_loader = get_dataloaders(config, use_mock=True)

    batch = next(iter(train_loader))
    images, masks = batch
    print(f"  Image batch shape: {images.shape}")
    print(f"  Mask batch shape : {masks.shape}")
    print(f"  Mask unique vals : {torch.unique(masks).tolist()}")
    print("  [OK] Dataset pipeline working correctly!")
