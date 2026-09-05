#!/usr/bin/env python3
"""Offline RKNN video evaluator using the production multi-person state machine.

The harness only replaces the source, clock, model and publisher.  Tracking,
temporal inference and fall-state transitions are executed by ``app.StreamWorker``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time as _real_time
from pathlib import Path

import cv2
import numpy as np

TARGET_FPS = 15.0
ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jetson_metrics():
    return load_module(ROOT / "platforms/jetson/tools/evaluate_videos.py", "jetson_metrics")


def source_hashes() -> dict[str, str]:
    paths = {
        "rknn_app": ROOT / "platforms/rknn/app.py",
        "fall_core": ROOT / "platforms/rknn/fall_core.py",
        "rknn_pose": ROOT / "platforms/rknn/rknn_pose.py",
        "video_source": ROOT / "platforms/rknn/video_source.py",
        "rknn_decoder_cpp": ROOT / "platforms/rknn/cpp/rknn_postprocess.cpp",
        "jetson_metrics": ROOT / "platforms/jetson/tools/evaluate_videos.py",
        "rk_evaluator": ROOT / "platforms/rknn/tools/evaluate_videos.py",
        "trace_extractor": ROOT / "platforms/rknn/tools/extract_gmdcsa_traces.py",
    }
    return {key: sha256(path) for key, path in paths.items()}


def corpus_identity(dataset: Path, subject: int = 4) -> dict:
    """Hash the complete holdout subject without interpreting its labels."""
    subject_root = dataset / f"subject-{subject}"
    if not subject_root.is_dir():
        raise SystemExit(f"dataset subject directory not found: {subject_root}")
    files = {
        path.relative_to(dataset).as_posix(): sha256(path)
        for path in sorted(subject_root.rglob("*")) if path.is_file()
    }
    if not files:
        raise SystemExit(f"dataset subject contains no files: {subject_root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"subject": subject, "files": files,
            "sha256": hashlib.sha256(canonical).hexdigest()}


class SyntheticClock:
    """Local clock object; app's module-level time reference is patched only temporarily."""
    def __init__(self):
        self.timestamp = 0.0

    def set(self, value: float) -> None:
        self.timestamp = float(value)

    def time(self) -> float:
        return self.timestamp

    def monotonic(self) -> float:
        return _real_time.monotonic()

    def perf_counter(self) -> float:
        return _real_time.perf_counter()

    def sleep(self, _seconds: float) -> None:
        return None


class FileSource:
    active_backend = "offline_opencv"

    def __init__(self, stream, _config, size, clock: SyntheticClock, app):
        self.path = Path(stream["path"])
        self.size = int(size)
        self.clock = clock
        self.app = app
        self.capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot decode {self.path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.source_fps = fps if np.isfinite(fps) and fps > 0 else TARGET_FPS
        self.index = 0
        self.next_sample = 0.0

    def read(self):
        while True:
            ok, bgr = self.capture.read()
            if not ok or bgr is None:
                self.close()
                self.app.RUNNING = False
                return None
            source_time = self.index / self.source_fps
            self.index += 1
            if source_time + 0.5 / self.source_fps < self.next_sample:
                continue
            h, w = bgr.shape[:2]
            scale = min(self.size / w, self.size / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.size, self.size, 3), 114, np.uint8)
            left, top = (self.size - nw) // 2, (self.size - nh) // 2
            canvas[top:top + nh, left:left + nw] = resized
            self.clock.set(self.next_sample)
            self.next_sample += 1.0 / TARGET_FPS
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class CapturePublisher:
    def __init__(self):
        self.payloads = []

    def publish(self, _topic, payload):
        self.payloads.append(payload)


def _first_trigger(values):
    for index, value in enumerate(values):
        if value:
            return index / TARGET_FPS
    return None


def run_clip(app, config: dict, clip, temporal_model: Path | None) -> dict:
    clock = SyntheticClock()
    publisher = CapturePublisher()
    stream = {"id": clip.clip_id, "path": str(clip.path), "enabled": True}
    local_config = json.loads(json.dumps(config))
    # The app reads the pose artifact from the top-level config, while the
    # evaluator's CLI keeps it explicit so reports can hash the exact input.
    local_config["model_path"] = str(config.get("model_path", ""))
    local_config["temporal_model"] = str(temporal_model) if temporal_model else ""
    local_config.setdefault("mqtt", {})["enabled"] = False
    local_config["postprocess"] = {"backend": "cpp", "strict": True, "fallback": "none"}
    originals = (app.create_video_source, app.time, app.RUNNING)
    app.create_video_source = lambda item, cfg, size: FileSource(item, cfg, size, clock, app)
    app.time = clock
    app.RUNNING = True
    worker = None
    try:
        worker = app.StreamWorker(local_config, stream, publisher)
        worker.run()
    finally:
        app.create_video_source, app.time, app.RUNNING = originals
    if worker.exception is not None:
        raise RuntimeError(f"offline worker failed for {clip.path}") from worker.exception
    temporal = []
    deployed = []
    inference = []
    pipeline = []
    coverage = []
    for payload in publisher.payloads:
        persons = payload.get("persons", [])
        temporal.append(any(p.get("temporal", {}).get("positive", False) for p in persons))
        deployed.append(any(p.get("state") in ("fallen", "recovering") for p in persons))
        inference.append(float(payload.get("inference_ms", payload.get("inference_time_ms", 0))))
        pipeline.append(float(payload.get("pipeline_ms", 0)))
        coverage.append(bool(payload.get("person_count", 0)))
    if not publisher.payloads:
        raise RuntimeError(f"decoded zero sampled frames: {clip.path}")
    backends = {payload.get("postprocess_backend") for payload in publisher.payloads}
    if backends != {"cpp"}:
        raise RuntimeError(f"C++ postprocess is required for evaluation, got {sorted(backends, key=str)}")
    return {"clip_id": clip.clip_id, "path": str(clip.path), "label": clip.label,
            "onset_sec": None if not np.isfinite(clip.onset_sec) else clip.onset_sec,
            "frames": len(publisher.payloads),
            "temporal_gate_trigger_sec": _first_trigger(temporal),
            "deployed_alert_trigger_sec": _first_trigger(deployed),
            "pose_coverage": sum(coverage) / len(coverage),
            "mean_inference_ms": float(np.mean(inference)),
            "p95_inference_ms": float(np.percentile(inference, 95, method="nearest")),
            "mean_pipeline_ms": float(np.mean(pipeline)),
            "max_person_count": max((p.get("person_count", 0) for p in publisher.payloads), default=0)}


def make_freeze_manifest(args) -> dict:
    if args.dataset is None:
        raise SystemExit("--dataset is required to freeze the Subject 4 corpus identity")
    required = {"platform": args.platform, "model_sha256": sha256(args.model),
                "temporal_sha256": sha256(args.temporal_model) if args.temporal_model else None,
                "config_sha256": sha256(args.config), "runtime_source_hashes": source_hashes(),
                "holdout_corpus": corpus_identity(args.dataset, 4)}
    return {"schema_version": 1, "kind": "rknn-evaluation-freeze", "platform": args.platform,
            "subject": args.subject, "identity": required,
            "model_sha256": required["model_sha256"],
            "temporal_sha256": required["temporal_sha256"],
            "config_sha256": required["config_sha256"],
            "runtime_source_hashes": required["runtime_source_hashes"],
            "holdout_corpus": required["holdout_corpus"],
            "created_utc": _real_time.strftime("%Y-%m-%dT%H:%M:%SZ", _real_time.gmtime()),
            "immutable": True}


def verify_freeze(args):
    if not args.freeze_manifest:
        raise SystemExit("Subject 4 requires --freeze-manifest")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    expected = make_freeze_manifest(args)["identity"]
    if freeze.get("identity") != expected:
        raise SystemExit("freeze manifest identity mismatch; model/temporal/config/runtime/holdout corpus changed")
    if freeze.get("immutable") is not True:
        raise SystemExit("freeze manifest must be immutable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--temporal-model", type=Path, required=True,
                    help="explicit frozen FP32 temporal profile; never silently disable the gate")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--subject", type=int, choices=(3, 4), default=3)
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--freeze-manifest", type=Path)
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.freeze_only:
        if args.report.exists():
            raise SystemExit("refusing to overwrite existing freeze/report file")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(make_freeze_manifest(args), indent=2) + "\n", encoding="utf-8")
        return 0
    if args.dataset is None:
        raise SystemExit("--dataset is required for evaluation")
    if args.subject == 4:
        if not args.allow_holdout:
            raise SystemExit("Subject 4 is frozen holdout: pass --allow-holdout")
        verify_freeze(args)
    sys.path.insert(0, str(ROOT / "platforms/rknn"))
    app = load_module(ROOT / "platforms/rknn/app.py", "rknn_eval_app")
    metrics = load_jetson_metrics()
    clips = metrics.gmdcsa_clips(args.dataset, args.subject)
    if args.limit > 0:
        clips = clips[:args.limit]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["model_path"] = str(args.model)
    rows = [run_clip(app, config, clip, args.temporal_model) for clip in clips]
    report = {"protocol": "production RK StreamWorker; 15fps offline OpenCV; fresh tracker/temporal per clip; strict C++ postprocess",
              "platform": args.platform, "dataset": f"gmdcsa24-subject{args.subject}",
              "subject": args.subject, "model": str(args.model), "model_sha256": sha256(args.model),
              "temporal_model": str(args.temporal_model) if args.temporal_model else None,
              "temporal_sha256": sha256(args.temporal_model) if args.temporal_model else None,
              "config_sha256": sha256(args.config), "runtime_source_hashes": source_hashes(),
              "postprocess": {"backend": "cpp", "strict": True, "fallback": "none"},
              "temporal_gate": metrics.metric_summary(clips, "temporal_gate_trigger_sec", rows),
              "deployed_alert": metrics.metric_summary(clips, "deployed_alert_trigger_sec", rows),
              "pose_coverage": float(np.mean([r["pose_coverage"] for r in rows])),
              "mean_inference_ms": float(np.mean([r["mean_inference_ms"] for r in rows])),
              "mean_pipeline_ms": float(np.mean([r["mean_pipeline_ms"] for r in rows])), "clips": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "temporal_gate", "deployed_alert", "pose_coverage", "mean_inference_ms")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
