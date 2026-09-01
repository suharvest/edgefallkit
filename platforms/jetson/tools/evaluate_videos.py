#!/usr/bin/env python3
"""Reproducible video-level evaluation for the deployed TensorRT pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGET_FPS = 15.0
SMOKE_TEST_CLIPS = {
    ("ADL", "01"), ("ADL", "05"), ("ADL", "10"), ("ADL", "15"), ("ADL", "20"),
    ("Fall", "01"), ("Fall", "05"), ("Fall", "09"), ("Fall", "13"), ("Fall", "17"),
}


def load_app():
    path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("jetson_fall_eval_app", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Clip:
    path: Path
    clip_id: str
    label: int
    onset_sec: float


def gmdcsa_clips(root: Path, subject_id: int = 4) -> list[Clip]:
    if subject_id not in (3, 4):
        raise ValueError("GMDCSA deployed evaluation supports development Subject 3 or test Subject 4")
    subject = root / f"subject-{subject_id}"
    onset_by_name: dict[str, float] = {}
    pattern = re.compile(r"fall(?:ing)?[^\[]*\[\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    csv_path = subject / "Fall.csv"
    for raw in csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[1:]:
        name = raw.split(",", 1)[0].strip()
        starts = [float(match.group(1)) for match in pattern.finditer(raw)]
        if re.fullmatch(r"\d{2}\.mp4", name) and starts:
            onset_by_name[name] = min(starts)
    clips: list[Clip] = []
    for kind in ("ADL", "Fall"):
        for path in sorted((subject / kind).glob("*.mp4")):
            if subject_id == 4 and (kind, path.stem) in SMOKE_TEST_CLIPS:
                continue
            clips.append(Clip(
                path=path,
                clip_id=f"subject-{subject_id}/{kind}/{path.stem}",
                label=int(kind == "Fall"),
                onset_sec=onset_by_name.get(path.name, math.inf),
            ))
    expected = {3: (43, 21), 4: (27, 12)}[subject_id]
    if len(clips) != expected[0] or sum(clip.label for clip in clips) != expected[1]:
        scope = "clean " if subject_id == 4 else ""
        raise RuntimeError(
            f"expected {expected[0]} {scope}Subject {subject_id} clips "
            f"({expected[1]} falls), got {len(clips)}")
    missing_onsets = [clip.clip_id for clip in clips if clip.label and not math.isfinite(clip.onset_sec)]
    if missing_onsets:
        raise RuntimeError(f"missing GMDCSA fall onsets: {missing_onsets}")
    return clips


def realbiomfall_clips(manifest_path: Path) -> list[Clip]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [
        Clip(Path(row["path"]), f"external/Fall/{Path(row['path']).stem}", 1,
             float(row["onset_sec"]))
        for row in rows if row.get("upstream_subset") == "testing"
    ]
    if len(clips) != 34:
        raise RuntimeError(f"expected 34 RealBiomFall testing clips, got {len(clips)}")
    return clips


def first_trigger(values: Iterable[bool]) -> float:
    for index, value in enumerate(values):
        if value:
            return index / TARGET_FPS
    return math.inf


def metric_summary(clips: list[Clip], trigger_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["clip_id"]: row for row in rows}
    tp = fn = tn = fp = early_count = 0
    latencies: list[float] = []
    misclassified: list[str] = []
    for clip in clips:
        trigger = by_id[clip.clip_id][trigger_key]
        when = math.inf if trigger is None else float(trigger)
        if clip.label:
            early = math.isfinite(when) and when < clip.onset_sec - 0.5
            detected = math.isfinite(when) and not early
            early_count += int(early)
            if detected:
                tp += 1
                latencies.append(when - clip.onset_sec)
            else:
                fn += 1
                misclassified.append(clip.clip_id)
        elif math.isfinite(when):
            fp += 1
            misclassified.append(clip.clip_id)
        else:
            tn += 1
    total = max(len(clips), 1)
    has_positive = tp + fn > 0
    has_negative = tn + fp > 0
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "n": len(clips), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        # A fall-only external subset can measure recall and latency, but not
        # general classifier accuracy/specificity/precision/F1.
        "accuracy": (tp + tn) / total if has_positive and has_negative else None,
        "recall": recall if has_positive else None,
        "specificity": specificity if has_negative else None,
        "precision": precision if has_positive and has_negative else None,
        "f1": (2 * precision * recall / max(precision + recall, 1e-9))
              if has_positive and has_negative else None,
        "early_fall_alerts": early_count,
        "mean_detection_latency_sec": statistics.fmean(latencies) if latencies else None,
        "median_detection_latency_sec": statistics.median(latencies) if latencies else None,
        "misclassified": misclassified,
    }


def close_tracker(tracker) -> None:
    for track in tracker.tracks:
        track.close()
    tracker.tracks.clear()


def evaluate_clip(app, bridge, config: dict[str, Any], clip: Clip) -> dict[str, Any]:
    capture = app.cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {clip.path}")
    source_fps = float(capture.get(app.cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = TARGET_FPS
    tracker = app.MultiPersonTracker(bridge, config)
    next_sample = 0.0
    source_index = 0
    temporal_flags: list[bool] = []
    deployed_flags: list[bool] = []
    detected_frames = 0
    person_frames = 0
    inference_ms: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_time = source_index / source_fps
            source_index += 1
            if source_time + 0.5 / source_fps < next_sample:
                continue
            detections, elapsed = bridge.infer(frame)
            persons = tracker.update(detections, next_sample, frame.shape[1], frame.shape[0])
            inference_ms.append(elapsed)
            detected_frames += int(bool(detections))
            person_frames += len(detections)
            temporal_flags.append(any(track.features.temporal_positive for track in persons))
            deployed_flags.append(any(track.state in ("fallen", "recovering") for track in persons))
            next_sample += 1.0 / TARGET_FPS
    finally:
        capture.release()
        close_tracker(tracker)
    frames = len(temporal_flags)
    if frames == 0:
        raise RuntimeError(f"decoded zero sampled frames: {clip.path}")
    temporal_trigger = first_trigger(temporal_flags)
    deployed_trigger = first_trigger(deployed_flags)
    return {
        "clip_id": clip.clip_id,
        "path": str(clip.path),
        "label": clip.label,
        "onset_sec": None if not math.isfinite(clip.onset_sec) else clip.onset_sec,
        "frames": frames,
        "source_fps": source_fps,
        "duration_sec": frames / TARGET_FPS,
        "temporal_gate_trigger_sec": None if not math.isfinite(temporal_trigger) else temporal_trigger,
        "deployed_alert_trigger_sec": None if not math.isfinite(deployed_trigger) else deployed_trigger,
        "pose_coverage": detected_frames / frames,
        "mean_people_per_frame": person_frames / frames,
        "mean_inference_ms": statistics.fmean(inference_ms),
        "p95_inference_ms": sorted(inference_ms)[min(len(inference_ms) - 1, math.ceil(len(inference_ms) * 0.95) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gmdcsa", type=Path)
    source.add_argument("--realbiomfall-manifest", type=Path)
    parser.add_argument("--gmdcsa-subject", type=int, choices=(3, 4), default=4,
                        help="3 for development selection; 4 for frozen clean test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke only; never use for reported metrics")
    args = parser.parse_args()

    app = load_app()
    if app.cv2 is None or app.np is None:
        raise RuntimeError("python3-opencv and numpy are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["engine_path"] = str(args.engine)
    config["trt_library"] = str(args.library)
    config.setdefault("mqtt", {})["enabled"] = False
    clips = (gmdcsa_clips(args.gmdcsa, args.gmdcsa_subject)
             if args.gmdcsa else realbiomfall_clips(args.realbiomfall_manifest))
    if args.limit > 0:
        clips = clips[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    bridge = app.TrtBridge(config)
    try:
        for index, clip in enumerate(clips, 1):
            row = evaluate_clip(app, bridge, config, clip)
            rows.append(row)
            print(f"[{index}/{len(clips)}] {clip.clip_id} frames={row['frames']} "
                  f"coverage={row['pose_coverage']:.3f} infer={row['mean_inference_ms']:.2f}ms",
                  file=sys.stderr, flush=True)
    finally:
        bridge.close()

    report = {
        "protocol": "15fps, frozen thresholds, one fresh tracker/temporal state per clip",
        "engine": str(args.engine),
        "dataset": (f"gmdcsa24-subject{args.gmdcsa_subject}-"
                    f"{'clean-test' if args.gmdcsa_subject == 4 else 'development'}")
                   if args.gmdcsa else "realbiomfall-testing",
        "temporal_gate": metric_summary(clips, "temporal_gate_trigger_sec", rows),
        "deployed_alert": metric_summary(clips, "deployed_alert_trigger_sec", rows),
        "pose_coverage": statistics.fmean(row["pose_coverage"] for row in rows),
        "mean_inference_ms": statistics.fmean(row["mean_inference_ms"] for row in rows),
        "clips": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset", "temporal_gate", "deployed_alert", "pose_coverage", "mean_inference_ms")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
