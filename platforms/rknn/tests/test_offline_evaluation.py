"""Offline RK evaluation harness tests; no RKNN runtime or dataset is required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "platforms/rknn"))
from tools import evaluate_videos as evaluator  # noqa: E402
from tools import extract_gmdcsa_traces as extractor  # noqa: E402
import app  # noqa: E402


def _clip(tmp_path: Path, frames: int = 5) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 24))
    for index in range(frames):
        writer.write(np.full((24, 32, 3), index * 10, np.uint8))
    writer.release()
    return path


def _detection(offset: float = 0.0):
    kp = [[16 + offset, 5, 1.0] for _ in range(17)]
    kp[5] = [12 + offset, 7, 1.0]; kp[6] = [20 + offset, 7, 1.0]
    kp[11] = [12 + offset, 16, 1.0]; kp[12] = [20 + offset, 16, 1.0]
    return {"box": [4 + offset, 2, 28 + offset, 22], "score": .9, "keypoints": kp}


class FakePose:
    closed = 0
    def __init__(self, *_args, **_kwargs): pass
    def infer(self, _frame): return ([], 2.0)
    def close(self): type(self).closed += 1


class FakeDecoder:
    active_backend = "cpp"
    configs = []
    def __init__(self, config=None, *_args, **_kwargs): self.configs.append(config)
    def decode(self, _outputs, *_args, **_kwargs): return [_detection(), _detection(2)]


class EmptyDecoder(FakeDecoder):
    def decode(self, _outputs, *_args, **_kwargs): return []


class FailingPose(FakePose):
    def __init__(self, *_args, **_kwargs): self.calls = 0
    def infer(self, _frame):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("inference broke after a partial result")
        return ([], 2.0)


def test_multi_person_file_source_and_eof_restore(tmp_path):
    path = _clip(tmp_path)
    original = (app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time)
    app.RKNNPose, app.PoseDecoder = FakePose, FakeDecoder
    cfg = {"model_path": "fake", "input_size": 32, "mqtt": {"topic": "x/{stream_id}"},
           "tracker": {"iou_threshold": .1, "max_lost_sec": .75}}
    clip = type("Clip", (), {"path": path, "clip_id": "test/clip", "label": 0, "onset_sec": np.inf})()
    try:
        row = evaluator.run_clip(app, cfg, clip, None)
    finally:
        app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time = original
    assert (app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time) == original
    assert FakePose.closed == 1
    assert row["frames"] == 5
    assert row["max_person_count"] == 2
    assert row["pose_coverage"] == 1.0
    assert FakeDecoder.configs[-1] == {"backend": "cpp", "strict": True, "fallback": "none"}


def test_blank_frames_complete_without_persons(tmp_path):
    path = _clip(tmp_path)
    original = (app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time)
    app.RKNNPose, app.PoseDecoder = FakePose, EmptyDecoder
    cfg = {"model_path": "fake", "input_size": 32, "mqtt": {"topic": "x/{stream_id}"},
           "tracker": {"iou_threshold": .1, "max_lost_sec": .75}}
    clip = type("Clip", (), {"path": path, "clip_id": "test/blank", "label": 0, "onset_sec": np.inf})()
    try:
        row = evaluator.run_clip(app, cfg, clip, None)
    finally:
        app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time = original
    assert row["frames"] == 5
    assert row["pose_coverage"] == 0.0


def test_partial_worker_failure_is_not_reported_as_success(tmp_path):
    path = _clip(tmp_path, frames=3)
    original = (app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time)
    app.RKNNPose, app.PoseDecoder = FailingPose, EmptyDecoder
    cfg = {"model_path": "fake", "input_size": 32, "mqtt": {"topic": "x/{stream_id}"},
           "tracker": {"iou_threshold": .1, "max_lost_sec": .75}}
    clip = type("Clip", (), {"path": path, "clip_id": "test/fail", "label": 0,
                              "onset_sec": np.inf})()
    try:
        with pytest.raises(RuntimeError, match="offline worker failed") as exc:
            evaluator.run_clip(app, cfg, clip, None)
    finally:
        app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time = original
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "partial result" in str(exc.value.__cause__)


def test_file_source_releases_capture_at_eof(tmp_path):
    path = _clip(tmp_path, frames=1)
    clock = evaluator.SyntheticClock()
    source = evaluator.FileSource({"path": str(path)}, {}, 32, clock, type("A", (), {"RUNNING": True})())
    assert source.read() is not None
    assert source.read() is None
    assert source.capture is None


def test_temporal_positive_drives_deployed_alert(tmp_path):
    path = _clip(tmp_path, frames=2)
    temporal = tmp_path / "temporal.npz"
    np.savez(temporal, frame_mask=np.ones(56, np.float32), mean=np.zeros(504, np.float32),
             scale=np.ones(504, np.float32), w1=np.ones((504, 1), np.float32),
             b1=np.zeros(1, np.float32), w2=np.ones(1, np.float32), b2=10.0,
             threshold=.5, consecutive=1, window=48)
    original = (app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time)
    app.RKNNPose, app.PoseDecoder = FakePose, FakeDecoder
    cfg = {"model_path": "fake", "input_size": 32, "mqtt": {"topic": "x/{stream_id}"},
           "tracker": {"iou_threshold": .1, "max_lost_sec": .75}}
    clip = type("Clip", (), {"path": path, "clip_id": "test/fall", "label": 1, "onset_sec": 0.0})()
    try:
        row = evaluator.run_clip(app, cfg, clip, temporal)
    finally:
        app.RKNNPose, app.PoseDecoder, app.RUNNING, app.time = original
    assert row["temporal_gate_trigger_sec"] == 0.0
    assert row["deployed_alert_trigger_sec"] == 0.0


def test_freeze_manifest_rejects_changed_identity(tmp_path):
    model = tmp_path / "model.rknn"; model.write_bytes(b"model")
    config = tmp_path / "config.json"; config.write_text("{}\n")
    dataset = tmp_path / "dataset"; (dataset / "subject-4").mkdir(parents=True)
    (dataset / "subject-4/Fall.csv").write_text("header\n")
    args = type("Args", (), {"platform": "rk3576", "subject": 3, "model": model,
                              "temporal_model": None, "config": config,
                              "dataset": dataset})()
    manifest = evaluator.make_freeze_manifest(args)
    manifest["identity"]["config_sha256"] = "changed"
    freeze = tmp_path / "freeze.json"; freeze.write_text(json.dumps(manifest))
    args.freeze_manifest = freeze
    with pytest.raises(SystemExit, match="identity mismatch"):
        evaluator.verify_freeze(args)


def test_subject4_requires_holdout_and_manifest(tmp_path):
    model = tmp_path / "model.rknn"; model.write_bytes(b"model")
    config = tmp_path / "config.json"; config.write_text("{}\n")
    dataset = tmp_path / "dataset"; (dataset / "subject-4").mkdir(parents=True)
    (dataset / "subject-4/Fall.csv").write_text("header\n")
    args = type("Args", (), {"platform": "rk3576", "subject": 4, "model": model,
                              "temporal_model": None, "config": config,
                              "freeze_manifest": None, "dataset": dataset})()
    with pytest.raises(SystemExit, match="requires --freeze-manifest"):
        evaluator.verify_freeze(args)


def test_freeze_only_does_not_initialize_hardware(tmp_path):
    model = tmp_path / "model.rknn"; model.write_bytes(b"model")
    config = tmp_path / "config.json"; config.write_text("{}\n")
    report = tmp_path / "freeze.json"
    dataset = tmp_path / "dataset"; (dataset / "subject-4").mkdir(parents=True)
    (dataset / "subject-4/Fall.csv").write_text("header\n")
    args = type("Args", (), {"platform": "rk3576", "subject": 3, "model": model,
                              "temporal_model": None, "config": config,
                              "report": report, "dataset": dataset})()
    result = evaluator.make_freeze_manifest(args)
    report.write_text(json.dumps(result))
    assert json.loads(report.read_text())["immutable"] is True


def test_freeze_rejects_changed_holdout_file(tmp_path):
    model = tmp_path / "model.rknn"; model.write_bytes(b"model")
    config = tmp_path / "config.json"; config.write_text("{}\n")
    dataset = tmp_path / "dataset"; (dataset / "subject-4/Fall").mkdir(parents=True)
    clip_file = dataset / "subject-4/Fall/01.mp4"; clip_file.write_bytes(b"video-v1")
    args = type("Args", (), {"platform": "rk3576", "subject": 4, "model": model,
                              "temporal_model": None, "config": config,
                              "dataset": dataset})()
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(evaluator.make_freeze_manifest(args)))
    args.freeze_manifest = freeze
    clip_file.write_bytes(b"video-v2")
    with pytest.raises(SystemExit, match="identity mismatch"):
        evaluator.verify_freeze(args)


def test_trace_source_inventory_changes_with_source_content(tmp_path):
    dataset = tmp_path / "dataset"
    source = dataset / "subject-1/ADL/01.mp4"
    source.parent.mkdir(parents=True); source.write_bytes(b"video-v1")
    sources = [(1, "ADL", source, tmp_path / "trace.jsonl")]
    first = extractor.source_inventory(sources, dataset)
    source.write_bytes(b"video-v2")
    second = extractor.source_inventory(sources, dataset)
    assert first != second
    assert list(first) == ["subject-1/ADL/01.mp4"]
