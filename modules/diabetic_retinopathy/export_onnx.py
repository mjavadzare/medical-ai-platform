"""Export ResNet50 DR model to ONNX, verify numerical parity, and benchmark CPU latency."""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import resnet50

DEFAULT_CHECKPOINT = Path(
    "modules/diabetic_retinopathy/artifacts/checkpoints/"
    "ResNet50_layer3_layer4_ft_bn_frozen_focal/"
    "ResNet50_layer3_layer4_ft_bn_frozen_focal_best_model.pth"
)
DEFAULT_ONNX_PATH = Path(
    "modules/diabetic_retinopathy/artifacts/onnx/"
    "ResNet50_layer3_layer4_ft_bn_frozen_focal.onnx"
)
DEFAULT_REPORT_PATH = Path("reports/onnx/onnx_benchmark.json")


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
        except (pickle.UnpicklingError, RuntimeError):
            checkpoint = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    else:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")

    return {k.removeprefix("module."): v for k, v in checkpoint.items()}


def build_resnet50_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    num_classes: int = 5,
) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features

    if "fc.weight" in state_dict:
        model.fc = nn.Linear(in_features, num_classes)
    elif "fc.1.weight" in state_dict:
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    else:
        fc_keys = [k for k in state_dict if k.startswith("fc.")]
        raise KeyError(f"Unrecognized classifier head keys in checkpoint: {fc_keys}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_model_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.inference_mode():
        torch.onnx.export(model, dummy_input, str(onnx_path), **export_kwargs)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def create_ort_session(
    onnx_path: Path, num_threads: int | None = None
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads is not None:
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def verify_parity(
    model: nn.Module,
    session: ort.InferenceSession,
    temperature: float,
    image_size: int = 224,
    batch_sizes: tuple[int, ...] = (1, 4),
    atol: float = 1e-4,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    all_argmax_match = True

    for batch_size in batch_sizes:
        sample = rng.standard_normal((batch_size, 3, image_size, image_size)).astype(
            np.float32
        )
        with torch.inference_mode():
            pt_logits = model(torch.from_numpy(sample)).numpy()
        ort_logits = session.run(["logits"], {"input": sample})[0]

        pt_probs = softmax_np(pt_logits, temperature)
        ort_probs = softmax_np(ort_logits, temperature)

        max_logit_diff = max(
            max_logit_diff, float(np.max(np.abs(pt_logits - ort_logits)))
        )
        max_prob_diff = max(max_prob_diff, float(np.max(np.abs(pt_probs - ort_probs))))
        if not np.array_equal(
            np.argmax(pt_probs, axis=-1), np.argmax(ort_probs, axis=-1)
        ):
            all_argmax_match = False

    passed = (max_logit_diff <= atol) and all_argmax_match
    return {
        "passed": passed,
        "tolerance": atol,
        "batch_sizes_tested": list(batch_sizes),
        "temperature": temperature,
        "max_abs_logit_diff": round(max_logit_diff, 8),
        "max_abs_prob_diff": round(max_prob_diff, 8),
        "argmax_match": all_argmax_match,
    }


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    sorted_vals = sorted(latencies_ms)
    p95_idx = min(int(0.95 * len(sorted_vals)), len(sorted_vals) - 1)
    mean_ms = statistics.fmean(sorted_vals)
    return {
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(statistics.median(sorted_vals), 2),
        "p95_ms": round(sorted_vals[p95_idx], 2),
        "min_ms": round(sorted_vals[0], 2),
        "throughput_img_per_sec": round(1000.0 / mean_ms, 2),
    }


def benchmark_cpu(
    model: nn.Module,
    session: ort.InferenceSession,
    image_size: int = 224,
    warmup: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    rng = np.random.default_rng(123)
    np_input = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)
    pt_input = torch.from_numpy(np_input)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(pt_input)

        pt_times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(pt_input)
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(warmup):
        _ = session.run(["logits"], {"input": np_input})

    ort_times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(["logits"], {"input": np_input})
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    pt_stats = summarize_latencies(pt_times)
    ort_stats = summarize_latencies(ort_times)
    speedup = round(pt_stats["mean_ms"] / ort_stats["mean_ms"], 2)

    return {
        "batch_size": 1,
        "image_size": image_size,
        "warmup_runs": warmup,
        "timed_runs": runs,
        "pytorch_cpu": pt_stats,
        "onnxruntime_cpu": ort_stats,
        "speedup_factor": speedup,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DR ResNet50 to ONNX and benchmark on CPU."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.9196)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_state_dict(args.checkpoint)
    model = build_resnet50_from_state_dict(state_dict, num_classes=args.num_classes)

    print(f"Exporting ONNX model (opset {args.opset}) -> {args.output}")
    export_model_to_onnx(
        model=model,
        onnx_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset,
    )

    session = create_ort_session(args.output, num_threads=args.threads)
    parity = verify_parity(
        model=model,
        session=session,
        temperature=args.temperature,
        image_size=args.image_size,
    )
    if not parity["passed"]:
        raise RuntimeError(f"ONNX numerical parity check failed: {parity}")

    bench = benchmark_cpu(
        model=model,
        session=session,
        image_size=args.image_size,
        warmup=args.warmup,
        runs=args.runs,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(args.output),
        "onnx_size_mb": round(args.output.stat().st_size / (1024 * 1024), 2),
        "opset_version": args.opset,
        "cpu_threads": args.threads or torch.get_num_threads(),
        "parity": parity,
        "benchmark": bench,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Parity Check ---")
    print(json.dumps(parity, indent=2))
    print("\n--- CPU Benchmark (Batch Size = 1) ---")
    print(json.dumps(bench, indent=2))
    print(f"\nSaved report -> {args.report}")


if __name__ == "__main__":
    main()
