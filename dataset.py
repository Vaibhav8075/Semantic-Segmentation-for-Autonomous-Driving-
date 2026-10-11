"""
dataset.py — Dataset Loading & Augmentation Pipeline
=====================================================

PyTorch Datasets for the prepared segmentation data and for synthetic mock
data, plus the Albumentations augmentation pipelines.

Masks on disk are expected to be *already* in our class scheme (values
0..NUM_CLASSES-1, and IGNORE_INDEX for unlabeled pixels). prepare_dataset.py
does the one and only Cityscapes label-ID → class conversion.

Train and validation share the same working resolution: every image is first
resized to (image_height, image_width). Training then takes random crops of
that resized image, so objects appear at the same scale in training,
validation, and inference.

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import json
import logging
import os
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    Config,
)
from mock_data import generate_mock_sample


# ============================================================================
# AUGMENTATION PIPELINES
# ============================================================================

def get_training_augmentations(
    height: int = 512,
    width: int = 1024,
    crop_height: int = 512,
    crop_width: int = 512,
) -> A.Compose:
    """
    Training augmentation pipeline.

    Resize to the working resolution, apply random scale/rotation, then take a
    random crop. Pixels created by the geometric transforms (borders, padding)
    get IGNORE_INDEX so they don't teach the model a fake class.

    Args:
        height, width: Working resolution (same as validation/inference).
        crop_height, crop_width: Size of the random training crop.

    Returns:
        Albumentations Compose pipeline.
    """
    crop_height = min(crop_height, height)
    crop_width = min(crop_width, width)

    return A.Compose([
        # Geometry
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.75, 1.5),
            translate_percent=(-0.05, 0.05),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=IGNORE_INDEX,
            p=0.7,
        ),
        A.PadIfNeeded(
            min_height=crop_height,
            min_width=crop_width,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=IGNORE_INDEX,
        ),
        A.RandomCrop(height=crop_height, width=crop_width),

        # Photometric (lighting, focus, weather)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.1),
        A.RandomShadow(p=0.2),

        # Normalize and convert
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_validation_augmentations(height: int = 512, width: int = 1024) -> A.Compose:
    """
    Validation / test / inference pipeline — resize and normalize only.

    Args:
        height, width: Working resolution.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_image(image: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Turn an RGB uint8 image into a model input tensor (1, 3, H, W).

    Uses exactly the validation pipeline, so inference matches evaluation.
    """
    transform = get_validation_augmentations(*image_size)
    return transform(image=image)["image"].unsqueeze(0)


# ============================================================================
# PREPARED DATASET (Cityscapes or mock data written by prepare_dataset.py)
# ============================================================================

def check_dataset_classes(root_dir: str) -> None:
    """
    Refuse to load a dataset prepared for a different class scheme.

    prepare_dataset.py records the class names in dataset_stats.json. Masks
    written for another scheme would silently train the wrong labels.
    """
    stats_path = os.path.join(root_dir, "dataset_stats.json")
    if not os.path.exists(stats_path):
        logging.warning(
            f"{stats_path} not found; cannot verify that masks use the classes "
            f"{CLASS_NAMES}. Re-run prepare_dataset.py if unsure."
        )
        return

    with open(stats_path) as f:
        stats = json.load(f)

    classes = stats.get("classes")
    if classes != CLASS_NAMES:
        raise ValueError(
            f"The dataset at '{root_dir}' was prepared for the classes "
            f"{classes or '(unrecorded: an older 7-class version)'}, but this code "
            f"uses {CLASS_NAMES}. Re-run prepare_dataset.py to regenerate it."
        )


class CityscapesSegDataset(Dataset):
    """
    PyTorch Dataset for a prepared segmentation dataset.

    Expected directory structure (created by prepare_dataset.py):
        dataset/
        ├── images/{train,val,test}/<name>.png
        ├── masks/{train,val,test}/<name>.png   (class indices, 255 = ignore)
        └── dataset_stats.json

    Args:
        root_dir: Root directory of the prepared dataset.
        split: One of 'train', 'val', 'test'.
        transform: Albumentations pipeline applied to image and mask.
        full_res_mask: If True, the transform is applied to the image only and
            the mask is returned at its original resolution (for evaluating
            at full label resolution).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        full_res_mask: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.full_res_mask = full_res_mask

        self.image_dir = os.path.join(root_dir, "images", split)
        self.mask_dir = os.path.join(root_dir, "masks", split)

        check_dataset_classes(root_dir)

        self.image_files: List[str] = []
        self.mask_files: List[str] = []
        self._load_file_pairs()

        logging.info(
            f"CityscapesSegDataset [{split}]: "
            f"{len(self)} samples loaded from {root_dir}"
        )

    def _load_file_pairs(self) -> None:
        """Pair each image with the mask of the same name."""
        if not os.path.exists(self.image_dir):
            logging.warning(f"Image directory not found: {self.image_dir}")
            return

        image_extensions = {".png", ".jpg", ".jpeg"}
        images = sorted(
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        )

        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.mask_dir, base_name + ".png")
            if os.path.exists(mask_path):
                self.image_files.append(os.path.join(self.image_dir, img_name))
                self.mask_files.append(mask_path)
            else:
                logging.warning(f"No mask found for image: {img_name}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.image_files[idx])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {self.mask_files[idx]}")
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor (3, H, W), normalized.
            mask: Tensor (H, W), int64 class indices (255 = ignore).
        """
        image, mask = self._read(idx)
        return _apply_transform(image, mask, self.transform, self.full_res_mask)

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Un-augmented (image uint8 RGB, mask uint8) pair for visualization."""
        return self._read(idx)


def _apply_transform(
    image: np.ndarray,
    mask: np.ndarray,
    transform: Optional[A.Compose],
    full_res_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an Albumentations pipeline and convert to tensors."""
    if transform is None:
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image_t, torch.from_numpy(mask).long()

    if full_res_mask:
        image_t = transform(image=image)["image"]
        return image_t, torch.from_numpy(mask).long()

    augmented = transform(image=image, mask=mask)
    return augmented["image"], augmented["mask"].long()


# ============================================================================
# MOCK DATASET (synthetic scenes generated on the fly)
# ============================================================================

class MockSegDataset(Dataset):
    """
    Synthetic road scenes generated on the fly (see mock_data.py).

    Sample `idx` is always the same scene (seeded by `seed + idx`), so the
    validation and test sets stay fixed between epochs. Augmentations are
    still random.

    Args:
        num_samples: Number of samples.
        image_size: (height, width) of generated images.
        transform: Albumentations augmentation pipeline.
        seed: Base seed; use different seeds for different splits.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 1024),
        transform: Optional[A.Compose] = None,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.seed = seed
        logging.info(f"MockSegDataset: {num_samples} synthetic samples (seed {seed})")

    def __len__(self) -> int:
        return self.num_samples

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Un-augmented (image uint8 RGB, mask uint8) pair."""
        rng = np.random.default_rng(self.seed + idx)
        return generate_mock_sample(self.image_size[0], self.image_size[1], rng)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.get_raw_sample(idx)
        return _apply_transform(image, mask, self.transform)


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
        use_mock: If True, generate synthetic scenes instead of reading disk.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    d = config.data
    h, w = d.image_height, d.image_width

    if d.use_augmentation:
        train_transform = get_training_augmentations(h, w, d.crop_height, d.crop_width)
    else:
        train_transform = get_validation_augmentations(h, w)
    val_transform = get_validation_augmentations(h, w)

    if use_mock:
        train_dataset = MockSegDataset(300, (h, w), train_transform, seed=0)
        val_dataset = MockSegDataset(100, (h, w), val_transform, seed=100_000)
        test_dataset = MockSegDataset(100, (h, w), val_transform, seed=200_000)
    else:
        train_dataset = CityscapesSegDataset(d.dataset_root, "train", train_transform)
        val_dataset = CityscapesSegDataset(d.dataset_root, "val", val_transform)
        test_dataset = CityscapesSegDataset(d.dataset_root, "test", val_transform)

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No training samples found in '{d.dataset_root}'. Run prepare_dataset.py "
            f"first, or pass --use-mock."
        )

    pin_memory = d.pin_memory and torch.cuda.is_available()

    def loader(dataset, num_workers: int, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=d.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    # Every worker is a separate process that imports PyTorch; on Windows each
    # reserves ~2 GB of commit memory. Evaluation needs far less throughput than
    # training, so cap its workers to keep the total within a 16 GB machine.
    eval_workers = min(2, d.num_workers)
    train_loader = loader(train_dataset, d.num_workers, train=True)
    val_loader = loader(val_dataset, eval_workers, train=False)
    test_loader = loader(test_dataset, eval_workers, train=False)

    logging.info(
        f"DataLoaders created — "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# ENTRY POINT — Dataset Statistics
# ============================================================================

if __name__ == "__main__":
    config = Config()
    print("\n" + "=" * 60)
    print("  Dataset Configuration")
    print("=" * 60)
    print(f"  Root       : {config.data.dataset_root}")
    print(f"  Image Size : {config.data.image_size}")
    print(f"  Crop Size  : {config.data.crop_size}")
    print(f"  Batch Size : {config.data.batch_size}")
    print(f"  Classes    : {NUM_CLASSES}")
    print(f"  Class Names: {CLASS_NAMES}")
    print(f"  Colors     : {CLASS_COLORS}")
    print("=" * 60)

    print("\nTesting with MockSegDataset...")
    config.data.num_workers = 0
    train_loader, val_loader, test_loader = get_dataloaders(config, use_mock=True)

    images, masks = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")
    print(f"  Mask batch shape : {masks.shape}")
    print(f"  Mask unique vals : {torch.unique(masks).tolist()}")
    print("  [OK] Dataset pipeline working correctly!")
