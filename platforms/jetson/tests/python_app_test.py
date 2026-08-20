#!/usr/bin/env python3
"""Host-only checks for the Python per-track fall state machine.

The fake bridge exposes a deterministic temporal gate, so these tests exercise
the state transitions without OpenCV, CUDA, TensorRT, or an RTSP source.
"""

import importlib.util
import json
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("jetson_fall_app", Path(__file__).parents[1] / "app.py")
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


class FakeBridge:
    """Controllable temporal bridge; each handle is one isolated track."""

    def __init__(self, positive_after=None):
        self.positive_after = positive_after

    def create_temporal(self):
        return {"calls": 0}

    def close_temporal(self, _handle):
        pass

    def temporal_update(self, handle, *_args, **_kwargs):
        handle["calls"] += 1
        positive = self.positive_after is not None and handle["calls"] >= self.positive_after
        return APP.CTemporalResult(1, int(positive), 0.95 if positive else 0.05)


def upright_detection(x=0.5, y=0.5):
    points = [APP.Keypoint(x, y, 0.9) for _ in range(17)]
    points[APP.LEFT_SHOULDER] = APP.Keypoint(x - 0.02, y - 0.15, 0.9)
    points[APP.RIGHT_SHOULDER] = APP.Keypoint(x + 0.02, y - 0.15, 0.9)
    points[APP.LEFT_HIP] = APP.Keypoint(x - 0.02, y + 0.15, 0.9)
    points[APP.RIGHT_HIP] = APP.Keypoint(x + 0.02, y + 0.15, 0.9)
    return APP.Detection(x, y, 0.22, 0.50, 0.9, points)


def lying_detection(x=0.5, y=0.72):
    # Shift the shoulder midpoint sideways from the hip midpoint: this is the
    # normalized horizontal torso geometry used by Track._features().
    points = [APP.Keypoint(x, y, 0.9) for _ in range(17)]
    points[APP.LEFT_SHOULDER] = APP.Keypoint(x - 0.30, y - 0.03, 0.9)
    points[APP.RIGHT_SHOULDER] = APP.Keypoint(x - 0.18, y - 0.03, 0.9)
    points[APP.LEFT_HIP] = APP.Keypoint(x + 0.18, y + 0.03, 0.9)
    points[APP.RIGHT_HIP] = APP.Keypoint(x + 0.30, y + 0.03, 0.9)
    return APP.Detection(x, y, 0.78, 0.20, 0.9, points)


def config():
    return {
        "keypoint_threshold": 0.25,
        "tracker": {"iou_threshold": 0.1, "center_distance_threshold": 0.25, "max_missed_frames": 3},
        "fall": {
            "temporal_confirmation_required": True,
            "confirmation_sec": 0.25,
            "cooldown_sec": 1.0,
            "recovery_window_sec": 0.2,
            "suspected_timeout_sec": 2.0,
        },
    }


def test_first_frame_lying_does_not_report():
    # Even a future/fake bridge that optimistically says positive on its first
    # call cannot bypass the no-history invariant.
    tracker = APP.MultiPersonTracker(FakeBridge(positive_after=1), config())
    tracks = tracker.update([lying_detection()], 0.0, 640, 480)
    assert len(tracks) == 1
    assert tracks[0].state == "normal"
    assert not tracks[0].fall_event


def test_geometry_only_does_not_confirm():
    tracker = APP.MultiPersonTracker(FakeBridge(), config())
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracks = tracker.update([lying_detection()], 0.1, 640, 480)
    assert tracks[0].state == "suspected"
    for timestamp in (0.4, 0.8, 1.2):
        tracks = tracker.update([lying_detection()], timestamp, 640, 480)
    assert tracks[0].state == "suspected"
    assert not tracks[0].fall_event


def test_temporal_positive_confirms_and_event_is_edge():
    tracker = APP.MultiPersonTracker(FakeBridge(positive_after=3), config())
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracker.update([lying_detection()], 0.1, 640, 480)
    tracks = tracker.update([lying_detection()], 0.2, 640, 480)
    assert tracks[0].state == "fallen"
    assert tracks[0].fall_event
    tracks = tracker.update([lying_detection()], 0.3, 640, 480)
    assert tracks[0].state == "fallen"
    assert not tracks[0].fall_event


def test_temporal_positive_waits_for_reacquisition():
    tracker = APP.MultiPersonTracker(FakeBridge(positive_after=3), config())
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracker.update([lying_detection()], 0.1, 640, 480)
    tracks = tracker.update([], 0.2, 640, 480)
    assert tracks[0].state == "suspected"
    assert not tracks[0].fall_event
    tracks = tracker.update([lying_detection()], 0.3, 640, 480)
    assert tracks[0].state == "fallen"
    assert tracks[0].fall_event


def test_normal_track_cannot_first_trigger_while_missing():
    tracker = APP.MultiPersonTracker(FakeBridge(positive_after=3), config())
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracker.update([upright_detection()], 0.1, 640, 480)
    tracks = tracker.update([], 0.2, 640, 480)
    assert tracks[0].state == "normal"
    assert not tracks[0].fall_event


def test_legacy_geometry_mode_is_explicit():
    legacy = config()
    legacy["fall"]["temporal_confirmation_required"] = False
    tracker = APP.MultiPersonTracker(FakeBridge(), legacy)
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracker.update([lying_detection()], 0.1, 640, 480)
    tracks = tracker.update([lying_detection()], 0.4, 640, 480)
    assert tracks[0].state == "fallen"
    assert tracks[0].fall_event


def test_recovery_returns_to_normal():
    tracker = APP.MultiPersonTracker(FakeBridge(positive_after=3), config())
    tracker.update([upright_detection()], 0.0, 640, 480)
    tracker.update([lying_detection()], 0.1, 640, 480)
    tracker.update([lying_detection()], 0.2, 640, 480)
    tracks = tracker.update([upright_detection()], 0.3, 640, 480)
    assert tracks[0].state == "recovering"
    tracks = tracker.update([upright_detection()], 0.6, 640, 480)
    assert tracks[0].state == "normal"
    assert not tracks[0].fall_event


def test_payload_matches_cross_platform_contract():
    tracker = APP.MultiPersonTracker(FakeBridge(), config())
    tracks = tracker.update([upright_detection()], 1.0, 640, 480)
    worker = APP.StreamWorker({"id": "jetson-test"}, config(), None)
    payload = worker._payload(tracks, 1, 12.5)
    validator_path = Path(__file__).resolve().parents[3] / "contracts" / "validate_payload.py"
    validator_spec = importlib.util.spec_from_file_location("mqtt_contract", validator_path)
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    validator.validate(payload)
    assert payload["event_id"] == payload["global_event_id"] == 0
    assert payload["event_id_scope"] == "stream_global_event_id"
    assert len(payload["pose17"]) == len(payload["keypoints"]) == 17
    assert payload["persons"][0]["missed_frames"] == 0

    fixture = json.loads((Path(__file__).parent / "fixtures" / "mqtt_payload.json").read_text())
    validator.validate(fixture)


def test_worker_count_shards_by_calibrated_capacity():
    """`auto` divides streams by the per-device single-process capacity."""
    def config_with(streams, **runtime):
        return {"runtime": runtime, "streams": streams}

    seven = {"workers": "auto", "max_streams_per_worker": 7}
    assert APP.worker_count(config_with([], **seven), 7) == 1
    assert APP.worker_count(config_with([], **seven), 8) == 2
    assert APP.worker_count(config_with([], **seven), 14) == 2
    assert APP.worker_count(config_with([], **seven), 15) == 3

    # Absent or zeroed calibration keeps the historical single process.
    assert APP.worker_count({}, 20) == 1
    assert APP.worker_count(config_with([], workers="auto", max_streams_per_worker=0), 20) == 1

    # An explicit count wins, but never exceeds the number of streams.
    assert APP.worker_count(config_with([], workers=3), 16) == 3
    assert APP.worker_count(config_with([], workers=99), 4) == 4


def test_shard_streams_is_balanced_and_lossless():
    streams = [{"id": f"cam-{index}"} for index in range(16)]
    shards = APP.shard_streams(streams, 3)
    sizes = sorted(len(shard) for shard in shards)
    assert sizes == [5, 5, 6], sizes
    flattened = [stream["id"] for shard in shards for stream in shard]
    assert sorted(flattened) == sorted(stream["id"] for stream in streams)
    assert len(set(flattened)) == len(streams)


def test_runtime_section_is_validated():
    import tempfile, os
    base = json.loads((Path(__file__).parents[1] / "config" / "config.json").read_text())
    for bad in ({"workers": 0}, {"workers": "many"}, {"workers": True},
                {"max_streams_per_worker": -1}):
        candidate = dict(base)
        candidate["runtime"] = {"workers": "auto", "max_streams_per_worker": 7, **bad}
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(candidate, handle)
        handle.close()
        try:
            APP.load_config(handle.name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"runtime{bad} should be rejected")
        finally:
            os.unlink(handle.name)


def main():
    test_first_frame_lying_does_not_report()
    test_geometry_only_does_not_confirm()
    test_temporal_positive_confirms_and_event_is_edge()
    test_temporal_positive_waits_for_reacquisition()
    test_normal_track_cannot_first_trigger_while_missing()
    test_legacy_geometry_mode_is_explicit()
    test_recovery_returns_to_normal()
    test_payload_matches_cross_platform_contract()
    test_worker_count_shards_by_calibrated_capacity()
    test_shard_streams_is_balanced_and_lossless()
    test_runtime_section_is_validated()
    print("python_app_test passed")


if __name__ == "__main__":
    main()
