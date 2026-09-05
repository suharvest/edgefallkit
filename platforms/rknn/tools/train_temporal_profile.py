#!/usr/bin/env python3
"""Freeze one RK pose-frontend temporal profile and export dependency-light NPZ."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_trace_identity(traces: Path, model: Path, platform: str) -> dict:
    manifest_path = traces / "extraction-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("platform") != platform or manifest.get("model_sha256") != sha256(model):
        raise SystemExit("extraction manifest platform/model mismatch")
    if set(manifest.get("subjects", [])) != {1, 2, 3}:
        raise SystemExit("training extraction manifest must contain S1-3 only")
    if not manifest.get("complete") or manifest.get("failed") or not manifest.get("extraction_identity"):
        raise SystemExit("extraction manifest is incomplete or missing identity")
    completed = manifest.get("completed", {})
    paths = set()
    counts = {1: 0, 2: 0, 3: 0}
    for key, row in completed.items():
        match = re.fullmatch(r"subject-([123])/(ADL|Fall)/(\d{2})\.mp4", key)
        if not match:
            raise SystemExit(f"invalid training trace key: {key}")
        path = traces / Path(key).with_suffix(".jsonl")
        if not path.is_file() or sha256(path) != row.get("trace_sha256"):
            raise SystemExit(f"training trace hash mismatch: {key}")
        paths.add(path)
        counts[int(match.group(1))] += 1
    actual = {p for s in (1, 2, 3) for p in (traces / f"subject-{s}").glob("*/*.jsonl")}
    if counts != {1: 32, 2: 48, 3: 43} or paths != actual:
        raise SystemExit("training trace inventory does not match the complete S1-3 manifest")
    return manifest


def load_development_clips(training, traces: Path, dataset: Path):
    # The shared loader scans all four onset CSVs even with allowed_subjects.
    # Supply only S1-3 annotations so training never reads Subject 4 metadata.
    with tempfile.TemporaryDirectory(prefix="rk-development-annotations-") as directory:
        view = Path(directory)
        for subject in (1, 2, 3):
            destination = view / f"subject-{subject}" / "Fall.csv"
            destination.parent.mkdir()
            destination.write_bytes((dataset / f"subject-{subject}" / "Fall.csv").read_bytes())
        return training.load_clips(traces, view, {1, 2, 3})


def export_npz(path: Path, scaler, model, best: dict, mask, training,
               source_sha256: str, model_sha256: str, extraction_manifest_sha256: str,
               extraction_identity: dict, trace_sha256s: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, window=training.WINDOW, frame_mask=mask.astype(np.float32),
        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
        w1=model.coefs_[0].astype(np.float32), b1=model.intercepts_[0].astype(np.float32),
        w2=model.coefs_[1].reshape(-1).astype(np.float32), b2=float(model.intercepts_[1][0]),
        threshold=float(best["threshold"]), consecutive=int(best["consecutive"]),
        training_source_sha256=np.asarray(source_sha256),
        pose_model_sha256=np.asarray(model_sha256),
        extraction_manifest_sha256=np.asarray(extraction_manifest_sha256),
        extraction_identity_json=np.asarray(json.dumps(extraction_identity, sort_keys=True)),
        trace_sha256s_json=np.asarray(json.dumps(trace_sha256s, sort_keys=True)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--development-report", type=Path, required=True)
    ap.add_argument("--model", type=Path,
                    help="pose model artifact; required with --require-trace-identity")
    ap.add_argument("--training-module", type=Path,
                    help="optional explicit train_temporal_model.py path for a remote training host")
    ap.add_argument("--require-trace-identity", action="store_true",
                    help="require an S1-3 extraction manifest and bind its hashes into the freeze")
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[3]
    extraction_manifest_path = args.traces / "extraction-manifest.json"
    # Validate the artifact/config binding before loading traces or fitting any
    # model.  In particular, this guard prevents an accidental S4-inclusive
    # manifest from reaching the training module.
    if args.require_trace_identity:
        if args.model is None:
            raise SystemExit("--require-trace-identity requires --model")
        if not extraction_manifest_path.exists():
            raise SystemExit("--require-trace-identity needs traces/extraction-manifest.json")
        preflight = validate_trace_identity(args.traces, args.model, args.platform)
        preflight_sha256 = sha256(extraction_manifest_path)
    training = load_training(project, args.training_module)
    clips = load_development_clips(training, args.traces, args.dataset)
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
    training_source_sha256 = sha256(args.training_module) if args.training_module else sha256(
        project / "platforms/recamera-sg2002/tools/train_temporal_model.py")
    extraction_manifest = None
    extraction_manifest_sha256 = ""
    extraction_identity = {}
    trace_sha256s = {}
    if args.require_trace_identity:
        extraction_manifest = validate_trace_identity(args.traces, args.model, args.platform)
        if sha256(extraction_manifest_path) != preflight_sha256:
            raise SystemExit("extraction manifest changed during training")
        extraction_identity = extraction_manifest.get("extraction_identity", {})
        trace_sha256s = {key: value.get("trace_sha256")
                         for key, value in extraction_manifest.get("completed", {}).items()}
        if len(trace_sha256s) != len(clips) or any(not value for value in trace_sha256s.values()):
            raise SystemExit("extraction manifest has incomplete trace hashes")
        extraction_manifest_sha256 = sha256(extraction_manifest_path)
    export_npz(args.output, scaler, model, best, mask, training, training_source_sha256,
               extraction_manifest.get("model_sha256", "") if extraction_manifest else "",
               extraction_manifest_sha256, extraction_identity, trace_sha256s)
    report = {"protocol": "fit S1-2; select S3; refit S1-3; Subject 4 not read",
              "phase": "configuration_freeze", "platform": args.platform,
              "trace_root": str(args.traces), "clips_by_subject": counts,
              "window_frames": training.WINDOW, "stride_frames": training.STRIDE,
              "feature_dim": training.FEATURE_DIM, "best": best,
              "npz": str(args.output), "npz_sha256": sha256(args.output),
              "training_source_sha256": training_source_sha256}
    if extraction_manifest:
        report.update({"extraction_manifest": str(extraction_manifest_path),
                       "extraction_manifest_sha256": extraction_manifest_sha256,
                       "extraction_identity": extraction_identity,
                       "trace_sha256s": trace_sha256s,
                       "pose_model_sha256": extraction_manifest.get("model_sha256")})
    args.development_report.parent.mkdir(parents=True, exist_ok=True)
    args.development_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
