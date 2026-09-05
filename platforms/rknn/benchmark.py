#!/usr/bin/env python3
"""RKNN benchmark with independent per-context workers and explicit metric scopes."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _stats(values):
    if not values:
        raise RuntimeError("benchmark produced no samples")
    values = sorted(values)
    return {"mean": statistics.mean(values), "p50": values[min(len(values) - 1, int(len(values) * .5))],
            "p95": values[min(len(values) - 1, int(len(values) * .95))], "max": max(values)}


def pct(values, p):
    """Compatibility helper retained for callers of the original benchmark."""
    return _stats(values)["p50" if p == .5 else "p95"]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_contract(outputs):
    if outputs is None:
        raise RuntimeError("RKNN inference returned None")
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    if len(outputs) != 9:
        raise RuntimeError(f"expected exactly 9 RKNN outputs, got {len(outputs)}")
    expected = sorted((grid, channels) for grid in (80, 40, 20) for channels in (64, 1, 51))
    observed = []
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1:
            raise RuntimeError(f"output[{index}] must be rank-4 batch-1, got {array.shape}")
        if array.shape[1] in (64, 1, 51) and array.shape[2] == array.shape[3] in (80, 40, 20):
            observed.append((int(array.shape[2]), int(array.shape[1])))
        elif array.shape[-1] in (64, 1, 51) and array.shape[1] == array.shape[2] in (80, 40, 20):
            observed.append((int(array.shape[1]), int(array.shape[-1])))
        else:
            raise RuntimeError(f"output[{index}] has unsupported pose-head shape {array.shape}")
        if not array.size or not np.all(np.isfinite(array)):
            raise RuntimeError(f"RKNN inference returned empty or non-finite output[{index}]")
    if sorted(observed) != expected:
        raise RuntimeError(f"pose head set mismatch: got {sorted(observed)}, expected {expected}")


def _validate(args):
    if args.contexts < 1 or (args.iterations is not None and args.iterations < 1):
        raise ValueError("--contexts and --iterations must be >= 1")
    if args.iterations is not None and args.iterations < args.contexts:
        raise ValueError("--iterations must be >= --contexts")
    if args.warmup < 0 or (args.warmup_seconds is not None and args.warmup_seconds < 0):
        raise ValueError("warmup values must be >= 0")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if args.duration_seconds is not None and getattr(args, "iterations_explicit", False):
        raise ValueError("--duration-seconds cannot be combined with --iterations")
    if args.core_mask != "auto":
        try:
            mask = int(args.core_mask, 0)
        except ValueError as exc:
            raise ValueError("--core-mask must be auto or a positive integer") from exc
        if mask <= 0:
            raise ValueError("--core-mask must be auto or a positive integer")
    masks = getattr(args, "core_masks", None)
    if masks is not None:
        if args.core_mask != "auto":
            raise ValueError("--core-masks cannot be combined with --core-mask")
        if len(masks) != args.contexts or any(int(mask) <= 0 for mask in masks):
            raise ValueError("--core-masks must contain one positive integer per context")


def _load_frame(args):
    if args.input_npy:
        frame = np.load(args.input_npy)
    elif args.image:
        import cv2
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise RuntimeError(f"cannot read image: {args.image}")
        h, w = bgr.shape[:2]
        scale = min(640 / w, 640 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (nw, nh))
        frame = np.full((640, 640, 3), 114, np.uint8)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        frame[top:top + nh, left:left + nw] = resized
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = np.full((640, 640, 3), 114, np.uint8)
    frame = np.ascontiguousarray(frame)
    if frame.shape != (640, 640, 3):
        raise ValueError(f"input frame must have shape (640, 640, 3), got {frame.shape}")
    return frame


def _run_context(index, args, frame, runner_factory, decoder_factory, clock, barrier, start_holder):
    masks = getattr(args, "core_masks", None)
    core_mask = (None if masks is None else int(masks[index])) if args.core_mask == "auto" else int(args.core_mask, 0)
    runner = decoder = None
    try:
        runner = runner_factory(args.model, core_mask)
        decoder = decoder_factory({"backend": args.postprocess_backend, "strict": True, "fallback": "none"})
        if args.warmup_seconds is not None:
            warmup_start = clock()
            while clock() - warmup_start < args.warmup_seconds:
                outputs, _ = runner.infer(frame)
                _output_contract(outputs)
                decoder.decode(outputs)
        else:
            for _ in range(args.warmup):
                outputs, _ = runner.infer(frame)
                _output_contract(outputs)
                decoder.decode(outputs)
        # All contexts finish warmup before any steady-state timer starts.
        barrier.wait()
        infer_ms, pipeline_ms, detections = [], [], []
        started = start_holder["value"]
        if args.duration_seconds is not None:
            work = lambda: clock() - started < args.duration_seconds
        else:
            remaining = args.iterations_for_context[index]
            work = lambda: remaining_check(remaining)
        first_iteration = True
        while first_iteration or work():
            first_iteration = False
            if args.duration_seconds is None:
                remaining -= 1
            sample_start = clock()
            outputs, native_ms = runner.infer(frame)
            _output_contract(outputs)
            found = decoder.decode(outputs)
            infer_ms.append(float(native_ms))
            pipeline_ms.append((clock() - sample_start) * 1000)
            detections.append(len(found))
        return {"context": index, "core_mask": core_mask, "iterations": len(infer_ms), "inference_ms": infer_ms,
                "pipeline_ms": pipeline_ms, "detections": detections,
                "postprocess_backend": decoder.active_backend, "wall_seconds": clock() - started,
                "metrics": {"rknn_call_ms": _stats(infer_ms), "pipeline_ms": _stats(pipeline_ms)}}
    except BaseException:
        barrier.abort()
        raise
    finally:
        if runner is not None:
            runner.close()


def remaining_check(value):
    return value > 0


def run_benchmark(args, runner_factory, decoder_factory, clock=time.perf_counter):
    _validate(args)
    frame = _load_frame(args)
    if args.duration_seconds is None:
        total = args.iterations
        args.iterations_for_context = [total // args.contexts + (i < total % args.contexts)
                                       for i in range(args.contexts)]
    else:
        args.iterations_for_context = [0] * args.contexts
    repetitions = []
    for _ in range(args.repetitions):
        with ThreadPoolExecutor(max_workers=args.contexts) as pool:
            start_holder = {}
            barrier = threading.Barrier(args.contexts, action=lambda: start_holder.update(value=clock()))
            futures = [pool.submit(_run_context, i, args, frame, runner_factory, decoder_factory, clock,
                                   barrier, start_holder)
                       for i in range(args.contexts)]
            contexts = [future.result() for future in futures]
        # Context wall time starts after runner/decoder creation and the warmup
        # barrier, and is evaluated before the executor closes the runners.
        repetitions.append({"contexts": contexts,
                            "wall_seconds": max(c["wall_seconds"] for c in contexts),
                            "metrics": {"rknn_call_ms": _stats([v for c in contexts for v in c["inference_ms"]]),
                                         "pipeline_ms": _stats([v for c in contexts for v in c["pipeline_ms"]])}})
    infer = [v for rep in repetitions for ctx in rep["contexts"] for v in ctx["inference_ms"]]
    pipeline = [v for rep in repetitions for ctx in rep["contexts"] for v in ctx["pipeline_ms"]]
    detection = [v for rep in repetitions for ctx in rep["contexts"] for v in ctx["detections"]]
    wall = sum(rep["wall_seconds"] for rep in repetitions)
    samples = len(infer)
    return {
        "contexts": args.contexts, "iterations": args.iterations if args.duration_seconds is None else samples,
        "mode": "duration" if args.duration_seconds is not None else "iterations",
        "repetitions": args.repetitions, "warmup": args.warmup, "warmup_seconds": args.warmup_seconds,
        "duration_seconds": args.duration_seconds, "core_mask": args.core_mask,
        "core_masks": list(getattr(args, "core_masks", None) or []),
        "model_sha256": _sha256(args.model),
        "input_file_sha256": _sha256(args.input_npy) if args.input_npy else None,
        "input_array_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "input_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "postprocess_backend": repetitions[0]["contexts"][0]["postprocess_backend"],
        "metric_scopes": {"rknn_call_ms": "RKNNPose.infer internal RKNNLite.inference call, including runtime input/output handling",
                          "pipeline_ms": "prebuilt RGB frame through infer and decode; excludes video decode, tracking, MQTT",
                          "throughput_fps": "all completed context samples divided by measured repetition wall time"},
        "inference_ms": _stats(infer), "pipeline_ms": _stats(pipeline), "rknn_call_ms": _stats(infer),
        "detections_per_frame": {"mean": statistics.mean(detection), "min": min(detection), "max": max(detection)},
        "throughput_fps": samples / wall, "samples": samples,
        "repetitions_detail": [{"wall_seconds": rep["wall_seconds"],
                                 "samples": sum(c["iterations"] for c in rep["contexts"]),
                                 "metrics": rep["metrics"],
                                 "contexts": [{"context": c["context"], "samples": c["iterations"],
                                               "core_mask": c["core_mask"], "metrics": c["metrics"]} for c in rep["contexts"]]}
                                for rep in repetitions],
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--contexts", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=None); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--warmup-seconds", type=float); ap.add_argument("--duration-seconds", type=float)
    ap.add_argument("--repetitions", type=int, default=1); ap.add_argument("--core-mask", default="auto")
    def parse_core_masks(value):
        parts = value.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise argparse.ArgumentTypeError("--core-masks cannot contain empty elements")
        try:
            return [int(part, 0) for part in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--core-masks must be comma-separated integers") from exc
    ap.add_argument("--core-masks", type=parse_core_masks)
    ap.add_argument("--input-npy"); ap.add_argument("--image"); ap.add_argument("--json-out")
    ap.add_argument("--postprocess-backend", choices=("cpp", "numpy", "auto"), default="cpp")
    args = ap.parse_args(argv)
    args.iterations_explicit = args.iterations is not None
    if args.iterations is None and args.duration_seconds is None:
        args.iterations = 200
    _validate(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    from rknn_pose import PoseDecoder, RKNNPose
    result = run_benchmark(args, RKNNPose, PoseDecoder)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
