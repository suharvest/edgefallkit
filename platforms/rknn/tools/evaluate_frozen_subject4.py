#!/usr/bin/env python3
"""Evaluate an already-frozen RK temporal NPZ on the clean Subject-4 split."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_training(project: Path, override: Path | None = None):
    path = override or (project / "platforms" / "recamera-sg2002" / "tools" / "train_temporal_model.py")
    spec = importlib.util.spec_from_file_location("rk_temporal_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--development-report", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--training-module", type=Path,
                    help="optional explicit train_temporal_model.py path for a remote evaluation host")
    args = ap.parse_args()
    freeze = json.loads(args.development_report.read_text(encoding="utf-8"))
    if freeze.get("phase") != "configuration_freeze" or freeze.get("platform") != args.platform:
        raise SystemExit("development report is not a matching frozen configuration")
    manifest_path = args.traces / "extraction-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if 4 not in manifest.get("subjects", []) or not manifest.get("holdout_authorized"):
        raise SystemExit("Subject-4 extraction manifest lacks explicit post-freeze authorization")

    project = Path(__file__).resolve().parents[3]
    training = load_training(project, args.training_module)
    clips = training.load_clips(args.traces, args.dataset, {4})
    if len(clips) != training.EXPECTED_SUBJECT_CLIPS[4]:
        raise SystemExit(f"incomplete Subject-4 traces: {len(clips)}")
    holdout = [c for c in clips if (c.path.parent.name, c.path.stem) not in training.SMOKE_TEST_CLIPS]
    if len(holdout) != 27:
        raise SystemExit(f"expected 27 clean holdout clips, got {len(holdout)}")

    data = np.load(args.model)
    mask, mean, scale = data["frame_mask"], data["mean"], data["scale"]
    w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], float(data["b2"])
    threshold, consecutive = float(data["threshold"]), int(data["consecutive"])
    probs = {}
    for clip in holdout:
        x, _ = training.clip_windows(clip, False, mask)
        hidden = np.maximum(0, ((x - mean) / np.maximum(scale, 1e-12)) @ w1 + b1)
        logits = hidden @ w2 + b2
        probs[clip.path] = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    metrics = training.metrics(holdout, probs, threshold, consecutive)
    report = {
        "protocol": "frozen S1-3 configuration; clean S4 test excludes 10 prior smoke clips",
        "phase": "frozen_test", "platform": args.platform,
        "model": str(args.model), "model_sha256": sha256(args.model),
        "development_report": str(args.development_report),
        "subject4_total": len(clips), "subject4_clean_test": len(holdout),
        "threshold": threshold, "consecutive": consecutive,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
