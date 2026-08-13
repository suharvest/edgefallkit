#!/usr/bin/env python3
"""Resume-safe GMDCSA pose trace extraction with the target board's RKNN frontend."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fall_core import Detection, IoUTracker, make_observation  # noqa: E402
from rknn_pose import RKNNPose, decode_pose  # noqa: E402

TARGET_FPS = 15.0
EXPECTED = {1: 32, 2: 48, 3: 43, 4: 37}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def row_for_tracks(tracks, timestamp: float, width: int, height: int,
                   keypoint_threshold: float, inference_ms: float) -> dict:
    visible = [track for track in tracks if track.detection is not None]
    primary = max(visible, key=lambda track: (track.last_seen, track.detection.score), default=None)
    if primary is None:
        return {"timestamp": round(timestamp * 1000, 3), "tracking": False,
                "fall_event": False, "pose17": [], "features": {},
                "inference_time_ms": round(inference_ms, 3)}
    obs = make_observation(primary.detection, timestamp, width, height, keypoint_threshold)
    return {
        "timestamp": round(timestamp * 1000, 3),
        "tracking": True,
        "fall_event": False,
        "track_id": primary.track_id,
        "pose17": [[float(v) for v in point] for point in primary.detection.keypoints],
        "features": {
            "valid": obs.valid, "hip_y": obs.hip_y,
            "torso_angle_deg": obs.torso_angle, "bbox_aspect_ratio": obs.aspect,
            "person_score": obs.person_score,
        },
        "inference_time_ms": round(inference_ms, 3),
    }


def extract_clip(model: RKNNPose, source: Path, destination: Path, args) -> dict:
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = TARGET_FPS
    tracker = IoUTracker(args.iou_threshold, args.max_lost_sec)
    rows, infer = [], []
    source_index, next_sample = 0, 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            source_time = source_index / source_fps
            source_index += 1
            if source_time + 0.5 / source_fps < next_sample:
                continue
            rgb = letterbox(bgr, args.input_size)
            outputs, inference_ms = model.infer(rgb)
            raw = decode_pose(outputs, args.score_threshold, args.nms_threshold, args.input_size)
            detections = [Detection(item["box"], item["score"], item["keypoints"]) for item in raw]
            tracks = tracker.update(detections, next_sample)
            rows.append(row_for_tracks(tracks, next_sample, args.input_size, args.input_size,
                                       args.keypoint_threshold, inference_ms))
            infer.append(inference_ms)
            next_sample += 1.0 / TARGET_FPS
    finally:
        capture.release()
    if not rows:
        raise RuntimeError(f"zero sampled frames: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    with part.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    part.replace(destination)
    return {
        "frames": len(rows),
        "tracking_frames": sum(bool(row["tracking"]) for row in rows),
        "valid_pose_frames": sum(bool(row.get("features", {}).get("valid")) for row in rows),
        "coverage": sum(bool(row["tracking"]) for row in rows) / len(rows),
        "inference_mean_ms": sum(infer) / len(infer),
        "trace_sha256": sha256(destination),
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subjects", default="1,2,3")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="required before Subject 4 can be read; use only after config freeze")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--score-threshold", type=float, default=0.35)
    ap.add_argument("--keypoint-threshold", type=float, default=0.25)
    ap.add_argument("--nms-threshold", type=float, default=0.45)
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--max-lost-sec", type=float, default=0.75)
    args = ap.parse_args()
    subjects = sorted({int(value) for value in args.subjects.split(",") if value.strip()})
    if any(subject not in EXPECTED for subject in subjects):
        raise SystemExit("subjects must be in 1,2,3,4")
    if 4 in subjects and not args.allow_holdout:
        raise SystemExit("Subject 4 is frozen holdout: pass --allow-holdout only after S1-3 freeze")

    sources = []
    for subject in subjects:
        for kind in ("ADL", "Fall"):
            for source in sorted((args.dataset / f"subject-{subject}" / kind).glob("*.mp4")):
                destination = args.output / f"subject-{subject}" / kind / f"{source.stem}.jsonl"
                sources.append((subject, kind, source, destination))
    expected = sum(EXPECTED[s] for s in subjects)
    if args.limit <= 0 and len(sources) != expected:
        raise SystemExit(f"incomplete source split: got {len(sources)}, expected {expected}")
    if args.limit > 0:
        sources = sources[:args.limit]

    manifest_path = args.output / "extraction-manifest.json"
    manifest = {
        "schema_version": 1, "platform": args.platform,
        "model": str(args.model), "model_sha256": sha256(args.model),
        "dataset": str(args.dataset), "subjects": subjects,
        "holdout_authorized": bool(args.allow_holdout), "target_fps": TARGET_FPS,
        "input_size": args.input_size, "expected_clips": expected,
        "completed": {}, "failed": {}, "started_unix_ms": int(time.time() * 1000),
    }
    if args.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("platform") != args.platform or old.get("model_sha256") != manifest["model_sha256"]:
            raise SystemExit("resume manifest platform/model mismatch")
        manifest["completed"] = old.get("completed", {})
        manifest["failed"] = old.get("failed", {})
        manifest["started_unix_ms"] = old.get("started_unix_ms", manifest["started_unix_ms"])

    model = RKNNPose(str(args.model))
    try:
        for index, (subject, kind, source, destination) in enumerate(sources, 1):
            key = f"subject-{subject}/{kind}/{source.name}"
            completed = manifest["completed"].get(key)
            if args.resume and completed and destination.exists() and sha256(destination) == completed.get("trace_sha256"):
                print(f"[{index}/{len(sources)}] resume {key}", flush=True)
                continue
            try:
                result = extract_clip(model, source, destination, args)
                manifest["completed"][key] = {"source": str(source), "trace": str(destination), **result}
                manifest["failed"].pop(key, None)
                print(f"[{index}/{len(sources)}] {key} frames={result['frames']} coverage={result['coverage']:.3f}", flush=True)
            except Exception as exc:
                manifest["failed"][key] = {"error": repr(exc), "unix_ms": int(time.time() * 1000)}
                save_manifest(manifest_path, manifest)
                raise
            manifest["updated_unix_ms"] = int(time.time() * 1000)
            save_manifest(manifest_path, manifest)
    finally:
        model.close()
    requested_keys = {f"subject-{subject}/{kind}/{source.name}"
                      for subject, kind, source, _ in sources}
    manifest["complete"] = requested_keys.issubset(manifest["completed"]) and not any(
        key in manifest["failed"] for key in requested_keys)
    manifest["completed_clips"] = len(manifest["completed"])
    manifest["requested_clips"] = len(sources)
    manifest["finished_unix_ms"] = int(time.time() * 1000)
    save_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
