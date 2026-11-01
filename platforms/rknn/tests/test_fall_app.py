"""Tests for FallApp BASE-1 unified vision base hooks."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Add platforms/rknn and contracts to sys.path
RKNN_DIR = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
sys.path.insert(0, str(RKNN_DIR))
sys.path.insert(0, str(CONTRACTS_DIR))

from fall_app import FallApp, AppHooks, Outgoing
from validate_payload import validate


@dataclass
class MockDetection:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id: int = 0
    track_id: int = 1
    keypoints: tuple[float, ...] = ()


@dataclass
class MockFrameResult:
    stream_id: str
    seq: int
    wall_ms: float
    inference_ms: float
    detections: list[MockDetection]
    queue_delay_ms: float = 0.0


class MockStreamContext:
    def __init__(self, stream_id: str = "cam-01"):
        self.stream_id = stream_id
        self.state: dict = {}


def _make_keypoints(hip_y: float, lying: bool = False) -> tuple[float, ...]:
    kpts = [0.5, 0.5, 0.9] * 17
    if not lying:
        kpts[5 * 3 : 5 * 3 + 3] = [0.45, hip_y - 0.2, 0.9]
        kpts[6 * 3 : 6 * 3 + 3] = [0.55, hip_y - 0.2, 0.9]
        kpts[11 * 3 : 11 * 3 + 3] = [0.47, hip_y, 0.9]
        kpts[12 * 3 : 12 * 3 + 3] = [0.53, hip_y, 0.9]
    else:
        kpts[5 * 3 : 5 * 3 + 3] = [0.3, hip_y - 0.02, 0.9]
        kpts[6 * 3 : 6 * 3 + 3] = [0.3, hip_y - 0.02, 0.9]
        kpts[11 * 3 : 11 * 3 + 3] = [0.7, hip_y, 0.9]
        kpts[12 * 3 : 12 * 3 + 3] = [0.7, hip_y, 0.9]
    return tuple(kpts)


def test_fall_app_protocol_and_configure():
    app = FallApp()
    assert isinstance(app, AppHooks)
    assert app.name == "fall-detection"
    assert app.wants_frames is True
    assert app.analyzers(None) == []  # type: ignore[arg-type]
    assert app.plugins() == []

    app.configure(
        {
            "topic_template": "alerts/{stream_id}/fall",
            "keypoint_threshold": 0.3,
            "fall": {
                "torso_angle": 60.0,
            },
        },
        device_id="rk3588-01",
    )
    assert app.topic_template == "alerts/{stream_id}/fall"
    assert app.keypoint_threshold == 0.3
    assert app.device_id == "rk3588-01"
    assert app.fall_config.torso_angle == 60.0


def test_on_stream_lifecycle():
    app = FallApp()
    ctx = MockStreamContext("test-stream")
    app.on_stream_added(ctx)

    assert "detectors" in ctx.state
    assert "temporal" in ctx.state
    assert ctx.state["frame_id"] == 0
    assert ctx.state["global_event_id"] == 0

    assert app.on_event(ctx, None) == []  # type: ignore[arg-type]

    app.on_stream_removed(ctx)
    assert len(ctx.state) == 0


def test_on_frame_empty_detections():
    app = FallApp()
    ctx = MockStreamContext("cam-lobby")
    app.on_stream_added(ctx)

    res = MockFrameResult(
        stream_id="cam-lobby",
        seq=1,
        wall_ms=1723456789125.0,
        inference_ms=15.4,
        detections=[],
        queue_delay_ms=2.1,
    )

    outs = app.on_frame(ctx, res)
    assert len(outs) == 1
    out = outs[0]
    assert isinstance(out, Outgoing)
    assert out.topic_suffix == "results/cam-lobby"

    payload = out.payload
    assert isinstance(payload, dict)
    assert payload["stream_id"] == "cam-lobby"
    assert payload["frame_id"] == 1
    assert payload["timestamp"] == 1723456789125
    assert payload["person_detected"] is False
    assert payload["person_count"] == 0
    assert payload["fallen_count"] == 0
    assert payload["state"] == "normal"
    assert payload["fall_detected"] is False
    assert payload["fall_event"] is False
    assert payload["tracking"] is False
    assert payload["pose17"] == []
    assert payload["persons"] == []

    # Strict contract v1 validation
    validate(payload)


def test_on_frame_single_person_normal():
    app = FallApp()
    ctx = MockStreamContext("cam-01")
    app.on_stream_added(ctx)

    kpts = _make_keypoints(hip_y=0.4, lying=False)
    det = MockDetection(
        cx=0.5,
        cy=0.45,
        w=0.2,
        h=0.6,
        score=0.92,
        class_id=0,
        track_id=1,
        keypoints=kpts,
    )
    res = MockFrameResult(
        stream_id="cam-01",
        seq=1,
        wall_ms=1723456789125.0,
        inference_ms=12.5,
        detections=[det],
        queue_delay_ms=1.0,
    )

    outs = app.on_frame(ctx, res)
    assert len(outs) == 1
    payload = outs[0].payload

    assert payload["person_detected"] is True
    assert payload["person_count"] == 1
    assert payload["fallen_count"] == 0
    assert payload["state"] == "normal"
    assert payload["tracking"] is True
    assert len(payload["pose17"]) == 17
    assert len(payload["persons"]) == 1

    person = payload["persons"][0]
    assert person["track_id"] == 1
    assert person["person_detected"] is True
    assert person["state"] == "normal"
    assert len(person["pose17"]) == 17
    assert len(person["bbox"]) == 4

    validate(payload)


def test_on_frame_fall_sequence():
    app = FallApp(
        {
            "fall": {
                "temporal_confirmation_required": False,
                "hip_drop_speed": 0.1,
                "hip_drop_distance": 0.01,
                "confirmation": 0.3,
                "suspected_timeout": 2.0,
            }
        }
    )
    ctx = MockStreamContext("cam-fall")
    app.on_stream_added(ctx)

    # Frame 1: Standing upright at t=0.0s
    det1 = MockDetection(
        cx=0.5,
        cy=0.4,
        w=0.2,
        h=0.6,
        score=0.95,
        track_id=1,
        keypoints=_make_keypoints(0.35, lying=False),
    )
    res1 = MockFrameResult("cam-fall", 1, 0.0, 10.0, [det1])
    out1 = app.on_frame(ctx, res1)[0]
    validate(out1.payload)
    assert out1.payload["state"] == "normal"
    assert out1.payload["fall_event"] is False

    # Frame 2: Rapid drop to lying down at t=0.1s -> suspected
    det2 = MockDetection(
        cx=0.5,
        cy=0.6,
        w=0.6,
        h=0.2,
        score=0.93,
        track_id=1,
        keypoints=_make_keypoints(0.65, lying=True),
    )
    res2 = MockFrameResult("cam-fall", 2, 100.0, 10.0, [det2])
    out2 = app.on_frame(ctx, res2)[0]
    validate(out2.payload)
    assert out2.payload["state"] == "suspected"
    assert out2.payload["fall_event"] is False

    # Frame 3: Lying persists past confirmation (0.3s) at t=0.5s -> fallen!
    det3 = MockDetection(
        cx=0.5,
        cy=0.6,
        w=0.6,
        h=0.2,
        score=0.94,
        track_id=1,
        keypoints=_make_keypoints(0.65, lying=True),
    )
    res3 = MockFrameResult("cam-fall", 3, 500.0, 10.0, [det3])
    out3 = app.on_frame(ctx, res3)[0]
    payload3 = out3.payload

    validate(payload3)
    assert payload3["state"] == "fallen"
    assert payload3["fall_detected"] is True
    assert payload3["fall_event"] is True
    assert payload3["event_id"] == 1
    assert payload3["global_event_id"] == 1
    assert payload3["fallen_count"] == 1
    assert len(payload3["events"]) == 1

    event = payload3["events"][0]
    assert event["kind"] == "fall"
    assert event["track_id"] == 1
    assert event["event_id"] == 1
    assert event["global_event_id"] == 1


def test_on_frame_track_id_fallback():
    app = FallApp()
    ctx = MockStreamContext("cam-track-test")
    app.on_stream_added(ctx)

    # Detection without track_id (0)
    det0 = MockDetection(
        cx=0.5,
        cy=0.5,
        w=0.2,
        h=0.6,
        score=0.88,
        track_id=0,
        keypoints=_make_keypoints(0.4, lying=False),
    )
    res = MockFrameResult("cam-track-test", 1, 1000.0, 11.0, [det0])
    out = app.on_frame(ctx, res)[0]

    validate(out.payload)
    person = out.payload["persons"][0]
    assert person["track_id"] >= 1


def test_on_frame_multiple_persons():
    app = FallApp()
    ctx = MockStreamContext("cam-multi")
    app.on_stream_added(ctx)

    det1 = MockDetection(
        cx=0.25,
        cy=0.5,
        w=0.2,
        h=0.6,
        score=0.91,
        track_id=1,
        keypoints=_make_keypoints(0.4, lying=False),
    )
    det2 = MockDetection(
        cx=0.75,
        cy=0.5,
        w=0.2,
        h=0.6,
        score=0.85,
        track_id=2,
        keypoints=_make_keypoints(0.4, lying=False),
    )
    res = MockFrameResult("cam-multi", 1, 2000.0, 14.0, [det1, det2])
    out = app.on_frame(ctx, res)[0]

    validate(out.payload)
    assert out.payload["person_count"] == 2
    assert len(out.payload["persons"]) == 2
    assert {p["track_id"] for p in out.payload["persons"]} == {1, 2}
