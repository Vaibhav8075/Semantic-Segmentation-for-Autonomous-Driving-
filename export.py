"""
export.py — Model Export (ONNX + TensorRT Guide)
==================================================

Export trained PyTorch models to ONNX format for deployment.
Includes validation to ensure ONNX output matches PyTorch output.

Also provides a detailed TensorRT deployment guide in the docstrings.

Usage:
    python export.py --model unet --checkpoint best_model.pth --format onnx

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from config import ModelConfig, NUM_CLASSES
from model_factory import get_model
from utils import get_device, load_checkpoint, setup_logging


# ============================================================================
# ONNX EXPORT
# ============================================================================

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (512, 512),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export a PyTorch segmentation model to ONNX format.

    ONNX (Open Neural Network Exchange) provides a standard format for
    model deployment across different inference frameworks (TensorRT,
    OpenVINO, ONNX Runtime, etc.).

    Args:
        model: Trained PyTorch model.
        output_path: Path for the exported .onnx file.
        input_size: (H, W) input image dimensions.
        opset_version: ONNX opset version (17 recommended).
        dynamic_batch: Enable dynamic batch size for flexible inference.

    Returns:
        Path to the exported ONNX file.
    """
    model.eval()
    model.cpu()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Dynamic axes for flexible batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,  # Optimize constant folding
    )

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"ONNX model exported: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_size: Tuple[int, int] = (512, 512),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate ONNX model output matches PyTorch model output.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to exported ONNX file.
        input_size: Input image dimensions.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logging.warning(
            "onnx/onnxruntime not installed. "
            "Install with: pip install onnx onnxruntime"
        )
        return False

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model structure validated [OK]")

    # Create test input
    test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

    # PyTorch inference
    pytorch_model.eval()
    pytorch_model.cpu()
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {"input": test_input})[0]

    # Compare
    try:
        np.testing.assert_allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        logging.info("ONNX output matches PyTorch output [OK]")
        max_diff = np.abs(pytorch_output - onnx_output).max()
        logging.info(f"Maximum absolute difference: {max_diff:.6e}")
        return True
    except AssertionError as e:
        logging.warning(f"ONNX output mismatch: {e}")
        max_diff = np.abs(pytorch_output - onnx_output).max()
        logging.warning(f"Maximum absolute difference: {max_diff:.6e}")
        return False


def benchmark_onnx(
    onnx_path: str,
    input_size: Tuple[int, int] = (512, 512),
    num_runs: int = 50,
) -> dict:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model.
        input_size: Input dimensions.
        num_runs: Number of inference runs for benchmarking.

    Returns:
        Dictionary with timing statistics.
    """
    try:
        import onnxruntime as ort
        import time
    except ImportError:
        logging.warning("onnxruntime not installed")
        return {}

    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"input": test_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"input": test_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    stats = {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "fps": 1000.0 / np.mean(times),
    }

    print(f"\n  ONNX Inference Benchmark ({num_runs} runs):")
    print(f"    Mean: {stats['mean_ms']:.2f} ms")
    print(f"    Std:  {stats['std_ms']:.2f} ms")
    print(f"    Min:  {stats['min_ms']:.2f} ms")
    print(f"    Max:  {stats['max_ms']:.2f} ms")
    print(f"    FPS:  {stats['fps']:.1f}")

    return stats


# ============================================================================
# TENSORRT DEPLOYMENT GUIDE
# ============================================================================

TENSORRT_GUIDE = """
================================================================================
  TENSORRT DEPLOYMENT GUIDE
================================================================================

TensorRT is NVIDIA's high-performance deep learning inference optimizer and
runtime. It can significantly accelerate model inference on NVIDIA GPUs.

Prerequisites:
  - NVIDIA GPU with CUDA support
  - TensorRT 8.x+ installed
  - ONNX model exported from this project

Step 1: Install TensorRT
  # Via pip (recommended)
  pip install tensorrt

  # Or via NVIDIA's installation guide:
  # https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

Step 2: Convert ONNX to TensorRT Engine
  # Option A: Using trtexec (command-line tool)
  trtexec --onnx=model.onnx \\
          --saveEngine=model.engine \\
          --fp16 \\
          --workspace=4096 \\
          --minShapes=input:1x3x512x512 \\
          --optShapes=input:4x3x512x512 \\
          --maxShapes=input:8x3x512x512

  # Option B: Using Python API
  import tensorrt as trt

  logger = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(logger)
  network = builder.create_network(
      1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  )
  parser = trt.OnnxParser(network, logger)

  with open("model.onnx", "rb") as f:
      parser.parse(f.read())

  config = builder.create_builder_config()
  config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for speed
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

  engine = builder.build_serialized_network(network, config)

  with open("model.engine", "wb") as f:
      f.write(engine)

Step 3: Run Inference with TensorRT
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy as np

  # Load engine
  with open("model.engine", "rb") as f:
      engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

  context = engine.create_execution_context()

  # Allocate buffers
  input_shape = (1, 3, 512, 512)
  output_shape = (1, 7, 512, 512)

  d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
  d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

  # Run inference
  input_data = np.random.randn(*input_shape).astype(np.float32)
  cuda.memcpy_htod(d_input, input_data)
  context.execute_v2([int(d_input), int(d_output)])

  output = np.empty(output_shape, dtype=np.float32)
  cuda.memcpy_dtoh(output, d_output)

Expected Performance Improvements:
  ┌──────────────────┬──────────┬──────────┬──────────┐
  │ Framework        │ FP32     │ FP16     │ INT8     │
  ├──────────────────┼──────────┼──────────┼──────────┤
  │ PyTorch          │ ~50ms    │ ~25ms    │ N/A      │
  │ ONNX Runtime     │ ~40ms    │ ~20ms    │ N/A      │
  │ TensorRT         │ ~15ms    │ ~8ms     │ ~5ms     │
  └──────────────────┴──────────┴──────────┴──────────┘
  * Times are approximate for 512×512 input on RTX 3080

================================================================================
"""


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export segmentation model to deployment format"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "resnet_unet", "deeplabv3plus", "segformer"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx"],
                        help="Export format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: checkpoints/{model}.onnx)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--validate", action="store_true",
                        help="Validate exported model against PyTorch")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference speed")
    parser.add_argument("--tensorrt-guide", action="store_true",
                        help="Print TensorRT deployment guide")

    args = parser.parse_args()

    if args.tensorrt_guide:
        print(TENSORRT_GUIDE)
        return

    setup_logging()

    # Load model
    model_config = ModelConfig(architecture=args.model)
    model = get_model(model_config)
    load_checkpoint(args.checkpoint, model, device=torch.device("cpu"))
    model.eval()

    # Determine output path
    if args.output is None:
        args.output = os.path.join("checkpoints", f"{args.model}.onnx")

    if args.format == "onnx":
        # Export
        onnx_path = export_to_onnx(
            model, args.output,
            input_size=tuple(args.image_size),
        )

        # Validate
        if args.validate:
            print("\n  Validating ONNX export...")
            validate_onnx(model, onnx_path, tuple(args.image_size))

        # Benchmark
        if args.benchmark:
            print("\n  Benchmarking ONNX inference...")
            benchmark_onnx(onnx_path, tuple(args.image_size))

    print(f"\n[OK] Export complete!")
    print(f"   Model: {args.output}")
    print(f"\n   For TensorRT deployment, run:")
    print(f"   python export.py --tensorrt-guide")


if __name__ == "__main__":
    main()
