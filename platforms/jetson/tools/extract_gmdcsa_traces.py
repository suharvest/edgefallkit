#!/usr/bin/env python3
"""Extract reCamera-compatible 15 FPS pose traces using a TensorRT engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


TARGET_FPS = 15.0


def load_app():
    path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("jetson_fall_trace_app", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def close_tracker(tracker) -> None:
    for track in tracker.tracks:
        track.close()
    tracker.tracks.clear()


def frame_row(persons: list[Any], timestamp_sec: float, inference_ms: float) -> dict[str, Any]:
    visible = [track for track in persons if track.features.valid]
    primary = max(visible, key=lambda track: (track.age, track.score), default=None)
    if primary is None:
        return {
            "timestamp": round(timestamp_sec * 1000.0, 3),
            "tracking": False,
            "fall_event": False,
            "pose17": [],
            "features": {},
            "inference_time_ms": round(inference_ms, 3),
        }
    data = primary.as_json()
    return {
        "timestamp": round(timestamp_sec * 1000.0, 3),
        "tracking": True,
        "fall_event": bool(primary.fall_event),
        "track_id": primary.track_id,
        "pose17": data["pose17"],
        "features": data["features"],
        "inference_time_ms": round(inference_ms, 3),
    }


def extract_clip(app, bridge, config: dict[str, Any], source: Path, destination: Path) -> tuple[int, float]:
    capture = app.cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {source}")
    source_fps = float(capture.get(app.cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = TARGET_FPS
    tracker = app.MultiPersonTracker(bridge, config)
    source_index = 0
    next_sample = 0.0
    rows: list[dict[str, Any]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_time = source_index / source_fps
            source_index += 1
            if source_time + 0.5 / source_fps < next_sample:
                continue
            detections, inference_ms = bridge.infer(frame)
            persons = tracker.update(detections, next_sample, frame.shape[1], frame.shape[0])
            rows.append(frame_row(persons, next_sample, inference_ms))
            next_sample += 1.0 / TARGET_FPS
    finally:
        capture.release()
        close_tracker(tracker)
    if not rows:
        raise RuntimeError(f"zero sampled frames: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(destination)
    coverage = sum(bool(row["tracking"]) for row in rows) / len(rows)
    return len(rows), coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", default="1,2,3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()

    app = load_app()
    if app.cv2 is None or app.np is None:
        raise RuntimeError("python3-opencv and numpy are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["engine_path"] = str(args.engine)
    config["trt_library"] = str(args.library)
    config.setdefault("mqtt", {})["enabled"] = False
    subjects = sorted({int(value) for value in args.subjects.split(",") if value.strip()})
    sources: list[tuple[Path, Path]] = []
    for subject in subjects:
        for kind in ("ADL", "Fall"):
            for source in sorted((args.dataset / f"subject-{subject}" / kind).glob("*.mp4")):
                destination = args.output / f"subject-{subject}" / kind / f"{source.stem}.jsonl"
                sources.append((source, destination))
    if not sources:
        raise RuntimeError("no GMDCSA videos found")
    if args.limit > 0:
        sources = sources[:args.limit]

    bridge = app.TrtBridge(config)
    try:
        for index, (source, destination) in enumerate(sources, 1):
            if args.resume and destination.exists() and destination.stat().st_size > 0:
                print(f"[{index}/{len(sources)}] resume {destination}", file=sys.stderr, flush=True)
                continue
            frames, coverage = extract_clip(app, bridge, config, source, destination)
            print(f"[{index}/{len(sources)}] {source} frames={frames} coverage={coverage:.3f}",
                  file=sys.stderr, flush=True)
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
