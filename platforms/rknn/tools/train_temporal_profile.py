#!/usr/bin/env python3
"""Freeze one RK pose-frontend temporal profile and export dependency-light NPZ."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np


def load_training(project: Path, override: Path | None = None):
    path = override or (project / "platforms" / "recamera-sg2002" / "tools" / "train_temporal_model.py")
    spec = importlib.util.spec_from_file_location("rk_temporal_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export_npz(path: Path, scaler, model, best: dict, mask, training) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, window=training.WINDOW, frame_mask=mask.astype(np.float32),
        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
        w1=model.coefs_[0].astype(np.float32), b1=model.intercepts_[0].astype(np.float32),
        w2=model.coefs_[1].reshape(-1).astype(np.float32), b2=float(model.intercepts_[1][0]),
        threshold=float(best["threshold"]), consecutive=int(best["consecutive"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--development-report", type=Path, required=True)
    ap.add_argument("--training-module", type=Path,
                    help="optional explicit train_temporal_model.py path for a remote training host")
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[3]
    training = load_training(project, args.training_module)
    clips = training.load_clips(args.traces, args.dataset, {1, 2, 3})
    counts = {s: sum(c.subject == s for c in clips) for s in (1, 2, 3)}
    expected = {s: training.EXPECTED_SUBJECT_CLIPS[s] for s in (1, 2, 3)}
    if counts != expected:
        raise SystemExit(f"incomplete S1-3 traces: {counts}, expected {expected}")
    validation = [clip for clip in clips if clip.subject == 3]
    candidates = []
    for variant, mask in training.FRAME_MASKS.items():
        for hidden in (16, 32):
            for alpha in (1e-3, 1e-2):
                scaler, model = training.fit_model(clips, {1, 2}, hidden, alpha, 2026, mask)
                probs = {c.path: training.clip_probability(c, scaler, model, mask) for c in validation}
                for threshold in np.arange(0.30, 0.81, 0.05):
                    for consecutive in (1, 2, 3):
                        metrics = training.metrics(validation, probs, float(threshold), consecutive)
                        candidates.append({"variant": variant, "hidden": hidden, "alpha": alpha,
                                           "threshold": float(threshold), "consecutive": consecutive,
                                           "validation": metrics, "validation_f1": metrics["f1"],
                                           "validation_balanced_accuracy": .5 * (metrics["recall"] + metrics["specificity"])})
    best = max(candidates, key=lambda c: (c["validation_f1"], c["validation_balanced_accuracy"],
                                          c["consecutive"], c["threshold"], -c["hidden"], c["alpha"]))
    mask = training.FRAME_MASKS[best["variant"]]
    scaler, model = training.fit_model(clips, {1, 2, 3}, best["hidden"], best["alpha"], 2026, mask)
    export_npz(args.output, scaler, model, best, mask, training)
    report = {"protocol": "fit S1-2; select S3; refit S1-3; Subject 4 not read",
              "phase": "configuration_freeze", "platform": args.platform,
              "trace_root": str(args.traces), "clips_by_subject": counts,
              "window_frames": training.WINDOW, "stride_frames": training.STRIDE,
              "feature_dim": training.FEATURE_DIM, "best": best,
              "npz": str(args.output)}
    args.development_report.parent.mkdir(parents=True, exist_ok=True)
    args.development_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
