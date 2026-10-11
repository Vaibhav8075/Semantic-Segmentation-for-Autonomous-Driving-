"""
Regression tests for the data, metric, model and checkpoint pipeline.

Run from the project directory:
    python -m pytest tests -q
"""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from config import (
    BACKGROUND_CLASS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    ModelConfig,
)
from dataset import (
    CityscapesSegDataset,
    MockSegDataset,
    get_training_augmentations,
    get_validation_augmentations,
)
from losses import CombinedLoss
from metrics import SegmentationMetrics
from mock_data import generate_mock_sample
from model import DeepLabV3Plus, ResNetUNet, SegFormer, UNet
from model_factory import get_model, load_model_from_checkpoint
from prepare_dataset import generate_mock_dataset, remap_cityscapes_mask
from utils import save_checkpoint

ROAD, VEHICLE, PEDESTRIAN, SIDEWALK, SIGN = (CLASS_NAMES.index(n) for n in
                                              ("Road", "Vehicle", "Pedestrian", "Sidewalk", "Traffic Sign"))


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------

def test_cityscapes_remap():
    ids = np.array([[0, 1, 2, 3],      # void → ignore
                    [7, 8, 26, 33],    # road, sidewalk, car, bicycle
                    [24, 25, 19, 20],  # person, rider, traffic light, traffic sign
                    [6, 11, 23, 200]], dtype=np.uint8)  # ground, building, sky, unknown
    expected = np.array([[IGNORE_INDEX] * 4,
                         [ROAD, SIDEWALK, VEHICLE, VEHICLE],
                         [PEDESTRIAN, PEDESTRIAN, SIGN, SIGN],
                         [BACKGROUND_CLASS] * 4], dtype=np.uint8)
    np.testing.assert_array_equal(remap_cityscapes_mask(ids), expected)


def test_prepare_cityscapes_kaggle_layout(tmp_path):
    """Nested mirror layout; split by city; official val becomes our test set."""
    from prepare_dataset import prepare_cityscapes

    raw = tmp_path / "raw"
    for split, city in [("train", "citya"), ("train", "cityb"), ("val", "cityc"), ("test", "cityd")]:
        img_dir = raw / "leftImg8bit" / split / city
        gt_dir = raw / "gtFine_trainvaltest" / "gtFine" / split / city
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        name = f"{city}_000000_000019"
        cv2.imwrite(str(img_dir / f"{name}_leftImg8bit.png"), np.zeros((8, 16, 3), np.uint8))
        cv2.imwrite(str(gt_dir / f"{name}_gtFine_labelIds.png"), np.full((8, 16), 26, np.uint8))

    out = tmp_path / "prepared"
    stats = prepare_cityscapes(str(raw), str(out), val_cities=["cityb"])
    assert stats["splits"] == {"train": 1, "val": 1, "test": 1}
    assert os.listdir(out / "images" / "test") == ["cityc_000000_000019.png"]
    mask = cv2.imread(str(out / "masks" / "train" / "citya_000000_000019.png"), cv2.IMREAD_GRAYSCALE)
    assert (mask == VEHICLE).all()  # labelId 26 (car) → Vehicle


def test_prepared_masks_load_unchanged(tmp_path):
    """Regression: the dataset used to remap already-remapped masks a second time."""
    generate_mock_dataset(str(tmp_path), num_train=2, num_val=1, num_test=1, image_size=(64, 128))
    ds = CityscapesSegDataset(str(tmp_path), "train", transform=None)
    on_disk = cv2.imread(ds.mask_files[0], cv2.IMREAD_GRAYSCALE)
    _, loaded = ds[0]
    np.testing.assert_array_equal(loaded.numpy(), on_disk)
    assert loaded.max() < NUM_CLASSES


def test_dataset_rejects_other_class_scheme(tmp_path):
    generate_mock_dataset(str(tmp_path), num_train=1, num_val=1, num_test=1, image_size=(32, 64))
    stats_path = tmp_path / "dataset_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["classes"] = ["Road", "Lane Marking"]
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="Re-run prepare_dataset.py"):
        CityscapesSegDataset(str(tmp_path), "train")


def test_mock_images_are_not_label_colors():
    """The model must not be able to read the class off the pixel color."""
    image, mask = generate_mock_sample(128, 256, np.random.default_rng(0))
    assert image.shape == (128, 256, 3) and mask.shape == (128, 256)
    assert set(np.unique(mask)) <= set(range(NUM_CLASSES))
    road_pixels = image[mask == ROAD].astype(float)
    assert road_pixels.std(axis=0).mean() > 5  # textured, with painted lane lines


def test_mock_dataset_is_deterministic_per_index():
    ds = MockSegDataset(4, (64, 128), transform=None, seed=7)
    _, a = ds.get_raw_sample(2)
    _, b = ds.get_raw_sample(2)
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------
# Augmentations
# ----------------------------------------------------------------------------

def test_train_and_val_share_working_resolution():
    image, mask = generate_mock_sample(256, 512, np.random.default_rng(1))
    val = get_validation_augmentations(128, 256)(image=image, mask=mask)
    assert tuple(val["image"].shape) == (3, 128, 256)

    train = get_training_augmentations(128, 256, 96, 96)
    for _ in range(10):
        out = train(image=image, mask=mask)
        assert tuple(out["image"].shape) == (3, 96, 96)
        values = set(np.unique(out["mask"].numpy()))
        # Border pixels from rotation/zoom-out must be ignored, never "Road" (0)
        assert values <= set(range(NUM_CLASSES)) | {IGNORE_INDEX}


# ----------------------------------------------------------------------------
# Metrics and losses
# ----------------------------------------------------------------------------

def test_metrics_count_false_positives_on_absent_classes():
    target = torch.zeros(1, 2, 4, dtype=torch.long)       # all Road
    pred = torch.zeros(1, 2, 4, dtype=torch.long)
    pred[0, 0, 0] = VEHICLE                               # one wrong pixel, Vehicle not in GT

    m = SegmentationMetrics(NUM_CLASSES)
    m.update(pred, target)
    iou = m.per_class_iou()
    assert iou[ROAD] == pytest.approx(7 / 8)
    assert iou[VEHICLE] == 0.0
    assert np.isnan(iou[PEDESTRIAN])                      # absent everywhere → excluded
    assert m.mean_iou() == pytest.approx((7 / 8 + 0) / 2)
    assert m.get_summary()["mean_precision"] == pytest.approx((1.0 + 0.0) / 2)
    json.dumps(m.get_detailed_metrics())                  # NaN must not leak into JSON


def test_ignore_index_is_ignored():
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    target[0, 0, 0] = ROAD
    pred = torch.full((1, 4, 4), VEHICLE, dtype=torch.long)
    pred[0, 0, 0] = ROAD

    m = SegmentationMetrics(NUM_CLASSES)
    m.update(pred, target)
    assert m.pixel_accuracy() == 1.0
    assert m.mean_iou() == 1.0

    logits = torch.randn(1, NUM_CLASSES, 4, 4, requires_grad=True)
    loss = CombinedLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


# ----------------------------------------------------------------------------
# Models and checkpoints
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    UNet(num_classes=NUM_CLASSES, init_features=8),
    ResNetUNet(num_classes=NUM_CLASSES, pretrained=False),
    DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False),
    SegFormer(num_classes=NUM_CLASSES, variant="b0"),
], ids=["unet", "resnet_unet", "deeplabv3plus", "segformer_native"])
def test_model_output_shape(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 128))
    assert tuple(out.shape) == (1, NUM_CLASSES, 64, 128)


def test_frozen_encoder_keeps_batchnorm_in_eval_mode():
    model = ResNetUNet(num_classes=NUM_CLASSES, pretrained=False, freeze_encoder=True)
    model.train()
    assert not model.encoder.training
    assert model.decoder.training
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_checkpoint_rebuilds_its_own_model(tmp_path):
    cfg = ModelConfig(architecture="unet", unet_init_features=8)
    model = get_model(cfg).eval()
    path = str(tmp_path / "unet_best.pth")
    save_checkpoint(
        model, torch.optim.AdamW(model.parameters()), None, None,
        epoch=3, best_metric=0.5, history={}, save_path=path,
        extra={"model_config": cfg.__dict__, "class_names": CLASS_NAMES, "image_size": [64, 128]},
    )

    loaded, meta = load_model_from_checkpoint(path)
    assert isinstance(loaded, UNet)
    assert meta["image_size"] == (64, 128)
    x = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))

    with pytest.raises(ValueError, match="'unet' model"):
        load_model_from_checkpoint(path, architecture="deeplabv3plus")


def test_checkpoint_with_other_classes_is_rejected(tmp_path):
    cfg = ModelConfig(architecture="unet", unet_init_features=8)
    model = get_model(cfg)
    path = str(tmp_path / "old.pth")
    save_checkpoint(
        model, torch.optim.AdamW(model.parameters()), None, None,
        epoch=0, best_metric=0.0, history={}, save_path=path,
        extra={"model_config": cfg.__dict__, "class_names": ["Road", "Lane Marking"]},
    )
    with pytest.raises(ValueError, match="Retrain"):
        load_model_from_checkpoint(path)


def test_gradcam_hooks_are_removed():
    from explainability import SegmentationGradCAM

    model = UNet(num_classes=NUM_CLASSES, init_features=8)
    with SegmentationGradCAM(model) as cam:
        heatmap = cam.generate(torch.randn(1, 3, 64, 64), ROAD)
    assert heatmap.shape == (64, 64)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert not model.bottleneck._forward_hooks
