"""
export.py — Model Export (ONNX + TensorRT Guide)
==================================================

Export trained PyTorch models to ONNX format for deployment.
Includes validation to ensure ONNX output matches PyTorch output.

Also provides a detailed TensorRT deployment guide in the docstrings.

Usage:
    python export.py --checkpoint checkpoints/unet_best.pth --validate --benchmark

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

from config import ARCHITECTURES
from model_factory import load_model_from_checkpoint
from utils import setup_logging


# ============================================================================
# ONNX EXPORT
# ============================================================================

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (512, 1024),
    opset_version: int = 18,
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
        opset_version: ONNX opset version (18 is the lowest the PyTorch 2.9
            exporter implements natively).
        dynamic_batch: Enable dynamic batch size for flexible inference.

    Returns:
        Path to the exported ONNX file.
    """
    model.eval()
    model.cpu()

    # Dummy input with batch 2, so the exporter can't specialize batch=1
    dummy_input = torch.randn(2, 3, input_size[0], input_size[1])

    # Dynamic batch dimension (torch.export style)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch_size", min=1, max=64)},)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    common = dict(opset_version=opset_version, input_names=["input"],
                  output_names=["output"], do_constant_folding=True)

    try:
        torch.onnx.export(model, (dummy_input,), output_path,
                          dynamo=True, dynamic_shapes=dynamic_shapes, **common)
    except Exception as e:
        # torch.export can't keep the batch dimension symbolic through some
        # third-party code (e.g. Hugging Face SegFormer); the TorchScript-based
        # exporter handles those models.
        logging.warning(f"torch.export-based ONNX export failed ({type(e).__name__}); "
                        f"retrying with the TorchScript exporter.")
        dynamic_axes = ({"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                        if dynamic_batch else None)
        torch.onnx.export(model, (dummy_input,), output_path,
                          dynamo=False, dynamic_axes=dynamic_axes, **common)

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"ONNX model exported: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_size: Tuple[int, int] = (512, 1024),
    rtol: float = 1e-3,
    atol: float = 1e-4,
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

    # Compare logits (float32 kernels differ slightly) and predicted classes
    max_diff = np.abs(pytorch_output - onnx_output).max()
    same_class = (pytorch_output.argmax(1) == onnx_output.argmax(1)).mean()
    logits_ok = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    log = logging.info if logits_ok else logging.warning
    log(f"Max absolute logit difference: {max_diff:.3e} "
        f"({'within' if logits_ok else 'outside'} rtol={rtol}, atol={atol})")
    logging.info(f"Pixels with the same predicted class: {100 * same_class:.4f}%")
    if logits_ok:
        logging.info("ONNX output matches PyTorch output [OK]")
    return bool(logits_ok)


def benchmark_onnx(
    onnx_path: str,
    input_size: Tuple[int, int] = (512, 1024),
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

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(onnx_path, providers=providers)
    logging.info(f"ONNX Runtime providers: {session.get_providers()}")
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

    stats["provider"] = session.get_providers()[0]
    print(f"\n  ONNX Inference Benchmark ({num_runs} runs, {stats['provider']}, "
          f"input {input_size[0]}x{input_size[1]}):")
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
          --memPoolSize=workspace:4096 \\
          --minShapes=input:1x3x512x1024 \\
          --optShapes=input:4x3x512x1024 \\
          --maxShapes=input:8x3x512x1024

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
  input_shape = (1, 3, 512, 1024)
  output_shape = (1, NUM_CLASSES, 512, 1024)  # NUM_CLASSES = 6

  d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
  d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

  # Run inference
  input_data = np.random.randn(*input_shape).astype(np.float32)
  cuda.memcpy_htod(d_input, input_data)
  context.execute_v2([int(d_input), int(d_output)])

  output = np.empty(output_shape, dtype=np.float32)
  cuda.memcpy_dtoh(output, d_output)

Expected gains: TensorRT FP16 is typically several times faster than
ONNX Runtime FP32 on the same GPU, and INT8 (with calibration) faster still.
Measure on your own hardware with `trtexec --loadEngine=model.engine`.

================================================================================
"""


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export segmentation model to deployment format"
    )
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default=None, choices=ARCHITECTURES,
                        help="Model architecture (only needed for old checkpoints "
                             "without metadata)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx"],
                        help="Export format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: next to the checkpoint, .onnx)")
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        help="Export input size H W (default: the size the model was trained at)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate exported model against PyTorch")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference speed")
    parser.add_argument("--tensorrt-guide", action="store_true",
                        help="Print TensorRT deployment guide")

    args = parser.parse_args()

    # The ONNX exporter prints emoji; don't crash on legacy Windows consoles
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    if args.tensorrt_guide:
        print(TENSORRT_GUIDE)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    setup_logging()

    # Load model (architecture and input size come from the checkpoint)
    model, meta = load_model_from_checkpoint(args.checkpoint, torch.device("cpu"), args.model)
    input_size = tuple(args.image_size or meta["image_size"] or (512, 1024))

    # Determine output path
    if args.output is None:
        args.output = os.path.splitext(args.checkpoint)[0] + ".onnx"

    if args.format == "onnx":
        onnx_path = export_to_onnx(model, args.output, input_size=input_size)

        if args.validate:
            print("\n  Validating ONNX export...")
            validate_onnx(model, onnx_path, input_size)

        if args.benchmark:
            print("\n  Benchmarking ONNX inference...")
            benchmark_onnx(onnx_path, input_size)

    print(f"\n[OK] Export complete!")
    print(f"   Model: {args.output}")
    print(f"\n   For TensorRT deployment, run:")
    print(f"   python export.py --tensorrt-guide")


if __name__ == "__main__":
    main()
