"""Quick verification script for the semantic segmentation project."""
import torch
import sys

if __name__ == "__main__":
    print("=" * 60)
    print("  Semantic Segmentation Project - Verification")
    print("=" * 60)

    # 1. Loss Functions
    print("\n[1] Testing Loss Functions...")
    from losses import CrossEntropyLoss, DiceLoss, FocalLoss, CombinedLoss

    pred = torch.randn(2, 7, 64, 64)
    target = torch.randint(0, 7, (2, 64, 64))

    losses = {
        "CrossEntropy": CrossEntropyLoss(),
        "Dice": DiceLoss(),
        "Focal": FocalLoss(),
        "Combined": CombinedLoss(),
    }

    for name, fn in losses.items():
        v = fn(pred, target)
        print(f"  {name:15s}: {v.item():.4f} OK")

    # 2. Metrics
    print("\n[2] Testing Metrics...")
    from metrics import SegmentationMetrics

    m = SegmentationMetrics(7)
    preds = torch.randint(0, 7, (4, 64, 64))
    targets = torch.randint(0, 7, (4, 64, 64))
    # Make 70% match
    match = torch.rand(4, 64, 64) > 0.3
    preds[match] = targets[match]
    m.update(preds, targets)
    summary = m.get_summary()
    print(f"  Pixel Accuracy : {summary['pixel_accuracy']:.4f}")
    print(f"  Mean IoU       : {summary['mean_iou']:.4f}")
    print(f"  Mean Dice      : {summary['mean_dice']:.4f}")

    # 3. Model Architectures
    print("\n[3] Testing Model Architectures...")
    from model import UNet, ResNetUNet, DeepLabV3Plus, SegFormer

    x = torch.randn(2, 3, 256, 256)
    models_test = {
        "UNet": UNet(num_classes=7),
        "ResNet50-UNet": ResNetUNet(num_classes=7, pretrained=False),
        "DeepLabV3+": DeepLabV3Plus(num_classes=7, pretrained=False),
        "SegFormer-B0": SegFormer(num_classes=7, variant="b0"),
    }

    for name, model in models_test.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name:15s}: output={tuple(out.shape)}, params={params:.1f}M OK")

    # 4. Dataset
    print("\n[4] Testing Dataset Pipeline...")
    from config import Config
    from dataset import get_dataloaders

    config = Config()
    train_loader, val_loader, test_loader = get_dataloaders(config, use_mock=True)
    batch = next(iter(train_loader))
    images, masks = batch
    print(f"  Image batch : {tuple(images.shape)}")
    print(f"  Mask batch  : {tuple(masks.shape)}")
    print(f"  Mask classes: {torch.unique(masks).tolist()}")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATIONS PASSED!")
    print("=" * 60)

