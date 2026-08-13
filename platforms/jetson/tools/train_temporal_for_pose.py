#!/usr/bin/env python3
"""Train/freeze temporal weights against one or more TensorRT pose frontends.

Only Subjects 1–2 fit weights and Subject 3 selects hyperparameters. Subject 4
is intentionally unsupported here so a test split cannot leak into selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_training_module():
    path = Path(__file__).resolve().parents[2] / "fall-detection" / "tools" / "train_temporal_model.py"
    spec = importlib.util.spec_from_file_location("recamera_temporal_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_namespace(header: str, namespace: str) -> str:
    return header.replace("namespace fall::temporal_weights {", f"namespace {namespace} {{").replace(
        "}  // namespace fall::temporal_weights", f"}}  // namespace {namespace}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, action="append", required=True,
                        help="repeat to train one model across multiple pose frontends")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--namespace", default="jetson_fall::temporal_weights")
    args = parser.parse_args()
    training = load_training_module()

    clips_by_source = {
        root.name: training.load_clips(root, args.dataset, {1, 2, 3})
        for root in args.traces
    }
    for name, clips in clips_by_source.items():
        counts = {subject: sum(clip.subject == subject for clip in clips) for subject in (1, 2, 3)}
        expected = {subject: training.EXPECTED_SUBJECT_CLIPS[subject] for subject in (1, 2, 3)}
        if counts != expected:
            raise RuntimeError(f"incomplete traces for {name}: {counts}, expected {expected}")
    clips = [clip for source in clips_by_source.values() for clip in source]
    validation = [clip for clip in clips if clip.subject == 3]

    candidates: list[dict] = []
    for variant, frame_mask in training.FRAME_MASKS.items():
        for hidden in (16, 32):
            for alpha in (1e-3, 1e-2):
                scaler, model = training.fit_model(
                    clips, {1, 2}, hidden, alpha, 2026, frame_mask)
                validation_probs = {
                    clip.path: training.clip_probability(clip, scaler, model, frame_mask)
                    for clip in validation
                }
                for threshold in np.arange(0.30, 0.81, 0.05):
                    for consecutive in (1, 2, 3):
                        overall = training.metrics(
                            validation, validation_probs, float(threshold), consecutive)
                        per_source = {
                            name: training.metrics(
                                [clip for clip in source if clip.subject == 3], validation_probs,
                                float(threshold), consecutive)
                            for name, source in clips_by_source.items()
                        }
                        candidates.append({
                            "variant": variant, "hidden": hidden, "alpha": alpha,
                            "threshold": float(threshold), "consecutive": consecutive,
                            "validation": overall, "validation_by_frontend": per_source,
                            # Prefer robust frontends: maximize the worst frontend F1
                            # before aggregate F1 and balanced accuracy.
                            "min_frontend_f1": min(item["f1"] for item in per_source.values()),
                            "validation_f1": overall["f1"],
                            "validation_balanced_accuracy": 0.5 * (
                                overall["recall"] + overall["specificity"]),
                        })
    best = max(candidates, key=lambda item: (
        item["min_frontend_f1"], item["validation_f1"],
        item["validation_balanced_accuracy"], item["consecutive"], item["threshold"],
        -item["hidden"], item["alpha"],
    ))
    mask = training.FRAME_MASKS[best["variant"]]
    scaler, model = training.fit_model(clips, {1, 2, 3}, best["hidden"], best["alpha"], 2026, mask)

    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary) / "weights.h"
        training.export_header(
            generated, scaler, model, best["threshold"], best["consecutive"], mask)
        args.header.write_text(
            replace_namespace(generated.read_text(encoding="utf-8"), args.namespace),
            encoding="utf-8",
        )
    report = {
        "protocol": "fit Subjects 1-2; select on Subject 3; refit Subjects 1-3; no Subject 4",
        "trace_frontends": [path.name for path in args.traces],
        "best": best,
        "window_frames": training.WINDOW,
        "stride_frames": training.STRIDE,
        "feature_dim": training.FEATURE_DIM,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
