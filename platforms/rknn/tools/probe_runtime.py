#!/usr/bin/env python3
"""Probe one RKNN pose artifact and enforce the aligned nine-head contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


GRIDS = (80, 40, 20)
CHANNELS = (64, 1, 51)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def output_contract(outputs) -> list[dict]:
    if outputs is None:
        raise RuntimeError("RKNN inference returned None")
    values = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    if len(values) != 9:
        raise RuntimeError(f"expected exactly 9 RKNN outputs, got {len(values)}")
    records, identities = [], []
    for index, output in enumerate(values):
        array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1:
            raise RuntimeError(f"output[{index}] must be rank-4 batch-1, got {array.shape}")
        if array.shape[1] in CHANNELS and array.shape[2] == array.shape[3] in GRIDS:
            layout, channel, grid = "NCHW", int(array.shape[1]), int(array.shape[2])
        elif array.shape[-1] in CHANNELS and array.shape[1] == array.shape[2] in GRIDS:
            layout, channel, grid = "NHWC", int(array.shape[-1]), int(array.shape[1])
        else:
            raise RuntimeError(f"output[{index}] has unsupported pose-head shape {array.shape}")
        identity = (grid, channel)
        if identity in identities:
            raise RuntimeError(f"duplicate pose head grid={grid} channels={channel}")
        identities.append(identity)
        finite = bool(array.size and np.isfinite(array).all())
        if not finite:
            raise RuntimeError(f"output[{index}] is empty or non-finite")
        records.append({"index": index, "shape": list(array.shape), "layout": layout,
                        "grid": grid, "channels": channel, "dtype": str(array.dtype),
                        "finite": True, "min": float(array.min()), "max": float(array.max()),
                        "sha256": sha256_array(array)})
    expected = sorted((grid, channel) for grid in GRIDS for channel in CHANNELS)
    if sorted(identities) != expected:
        raise RuntimeError(f"pose head set mismatch: got {sorted(identities)}, expected {expected}")
    return records


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args, runner_factory=None) -> dict:
    model, input_path = Path(args.model), Path(args.input_npy)
    frame = np.load(input_path, allow_pickle=False)
    if frame.shape != (640, 640, 3) or frame.dtype != np.uint8:
        raise ValueError(f"input must be uint8 HWC (640, 640, 3), got {frame.dtype} {frame.shape}")
    report = {"schema_version": 1, "ok": False, "stage": "inputs-verified",
              "model": str(model), "model_sha256": sha256_file(model),
              "input_file": str(input_path), "input_file_sha256": sha256_file(input_path),
              "input_array_sha256": sha256_array(frame), "warmup": args.warmup,
              "iterations": args.iterations, "core_mask": args.core_mask}
    write_report(Path(args.report), report)
    if runner_factory is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rknn_pose import RKNNPose
        runner_factory = RKNNPose
    runner = None
    try:
        core_mask = None if args.core_mask == "auto" else int(args.core_mask, 0)
        runner = runner_factory(str(model), core_mask) if core_mask is not None else runner_factory(str(model))
        report["stage"] = "runtime-initialized"
        write_report(Path(args.report), report)
        for _ in range(args.warmup):
            outputs, _ = runner.infer(frame)
            output_contract(outputs)
        timings, last_records, output_signatures = [], None, []
        for _ in range(args.iterations):
            outputs, elapsed_ms = runner.infer(frame)
            last_records = output_contract(outputs)
            output_signatures.append([record["sha256"] for record in last_records])
            timings.append(float(elapsed_ms))
        ordered = sorted(timings)
        report.update({"ok": True, "stage": "complete", "output_count": 9,
                       "outputs": last_records,
                       "steady_output_hash_stable": all(
                           signature == output_signatures[0] for signature in output_signatures[1:]),
                       "output_signature_sha256": hashlib.sha256(
                           "\n".join(output_signatures[-1]).encode("ascii")).hexdigest(),
                       "rknn_call_ms": {"mean": statistics.mean(timings),
                                         "p50": statistics.median(timings),
                                         "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
                                         "min": min(timings), "max": max(timings),
                                         "samples": len(timings)}})
        write_report(Path(args.report), report)
        return report
    except BaseException as exc:
        report.update({"stage": "failed", "error": repr(exc)})
        write_report(Path(args.report), report)
        raise
    finally:
        if runner is not None:
            runner.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--core-mask", default="auto",
                        help="auto or an RKNNLite public core-mask integer (for example 0x1)")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    if args.core_mask != "auto":
        try:
            if int(args.core_mask, 0) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--core-mask must be auto or a positive integer")
    return args


def main(argv=None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
