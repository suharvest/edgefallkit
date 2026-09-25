#!/usr/bin/env python3
"""Scenario tests for the Jetson Python fall state machine (app.py Track).

The fake bridge exposes a deterministic temporal gate (positive from a chosen
timestamp on), so the state transitions are exercised without OpenCV, CUDA,
TensorRT, MQTT, or an RTSP source.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("jetson_fall_app", Path(__file__).parents[1] / "app.py")
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


class FakeBridge:
    """Temporal gate stub: positive from ``positive_from`` onward."""

    def __init__(self, positive_from=None):
        self.positive_from = positive_from

    def create_temporal(self):
        return {}

    def close_temporal(self, _handle):
        pass

    def temporal_update(self, _handle, *args):
        timestamp = args[-1]
        positive = self.positive_from is not None and timestamp >= self.positive_from
        return APP.CTemporalResult(1, int(positive), 0.95 if positive else 0.05)


def make_config(**fall):
    return {
        "keypoint_threshold": 0.25,
        "tracker": {"iou_threshold": 0.1, "center_distance_threshold": 0.35,
                    "max_missed_frames": 200},
        "fall": dict(fall),
    }


def detection(hip_y=0.45, torso=8.0, aspect=0.45):
    """Valid COCO-17 pose with the requested hip height, torso angle, and box
    aspect (width / height), centred at x=0.5."""
    dy = 0.12
    dx = dy * math.tan(math.radians(torso))
    shoulder_x, shoulder_y = 0.5 - dx, hip_y - dy
    points = [APP.Keypoint(0.5, hip_y, 0.9) for _ in range(17)]
    points[APP.LEFT_SHOULDER] = APP.Keypoint(shoulder_x - 0.02, shoulder_y, 0.9)
    points[APP.RIGHT_SHOULDER] = APP.Keypoint(shoulder_x + 0.02, shoulder_y, 0.9)
    points[APP.LEFT_HIP] = APP.Keypoint(0.48, hip_y, 0.9)
    points[APP.RIGHT_HIP] = APP.Keypoint(0.52, hip_y, 0.9)
    height = 0.40
    width = height * aspect
    # Detection x/y is the box centre.
    return APP.Detection(0.5, hip_y, width, height, 0.9, points)


_STAND = (8.0, 0.45)
_LIE = (80.0, 2.45)


def fall_frames(t_fall, n, fall_sec=0.4, hip0=0.45, hip1=0.75,
                stand_up_at=None, gap=None):
    """Standing, a ``fall_sec`` fall at ``t_fall``, then lying.  ``gap=(a,b)``
    marks frames in [a, b) as invalid pose; ``stand_up_at`` goes back upright."""
    for i in range(n):
        ts = i / 15.0
        if stand_up_at is not None and ts >= stand_up_at:
            hip, (torso, aspect) = hip0, _STAND
        elif ts < t_fall:
            hip, (torso, aspect) = hip0, _STAND
        elif fall_sec > 0 and ts < t_fall + fall_sec:
            k = (ts - t_fall) / fall_sec
            hip = hip0 + (hip1 - hip0) * k
            torso = _STAND[0] + (_LIE[0] - _STAND[0]) * k
            aspect = _STAND[1] + (_LIE[1] - _STAND[1]) * k
        else:
            hip, (torso, aspect) = hip1, _LIE
        yield ts, hip, torso, aspect, not (gap and gap[0] <= ts < gap[1])


def replay(tracker, frames):
    """Feed one person; returns per-frame snapshots (state, event_id,
    hip_drop_speed) plus the per-frame fall-event flags.  Snapshots are
    copied because Track objects are mutated in place."""
    snapshots, events = [], []
    for ts, hip, torso, aspect, valid in frames:
        detections = [detection(hip, torso, aspect)] if valid else []
        tracks = tracker.update(detections, ts, 640, 480)
        first = tracks[0] if tracks else None
        snapshots.append(None if first is None else
                         (first.state, first.event_id, first.features.hip_drop_speed))
        events.append(any(t.fall_event for t in tracks))
    return snapshots, events


class AppFallTest(unittest.TestCase):
    def test_upright_person_with_temporal_positive_never_alarms(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=0.0), make_config())
        frames = [(i / 15.0, 0.45, 8.0, 0.45, True) for i in range(150)]
        tracks, events = replay(tracker, frames)
        self.assertFalse(any(events))
        self.assertEqual({s[0] for s in tracks}, {"normal"})

    def test_temporal_positive_on_invalid_frame_in_normal_never_alarms(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=0.0), make_config())
        frames = [(0.0, 0.45, 8.0, 0.45, True),
                  (0.1, 0.45, 8.0, 0.45, True),
                  (0.2, 0.0, 0.0, 0.0, False),
                  (0.3, 0.45, 8.0, 0.45, True)]
        tracks, events = replay(tracker, frames)
        self.assertFalse(any(events))
        self.assertEqual({s[0] for s in tracks}, {"normal"})

    def test_sit_down_with_temporal_positive_never_alarms(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=1.2), make_config())
        frames = []
        for i in range(150):
            ts = i / 15.0
            if ts < 1.0:
                hip, torso, aspect = 0.45, 8.0, 0.45
            elif ts < 1.8:
                k = (ts - 1.0) / 0.8
                hip, torso, aspect = 0.45 + 0.2 * k, 8.0 + 22.0 * k, 0.45 + 0.9 * k
            else:
                hip, torso, aspect = 0.65, 30.0, 1.35   # seated, wide box
            frames.append((ts, hip, torso, aspect, True))
        tracks, events = replay(tracker, frames)
        self.assertFalse(any(events))
        self.assertNotIn("fallen", {s[0] for s in tracks})

    def test_real_fall_with_temporal_positive_alarms_once(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=1.6), make_config())
        tracks, events = replay(tracker, fall_frames(1.0, 150))
        self.assertEqual(sum(events), 1)
        edge = events.index(True)
        self.assertEqual(tracks[edge][1], 1)
        # FALLEN was reached through SUSPECTED, never straight from NORMAL.
        self.assertEqual(tracks[edge - 1][0], "suspected")

    def test_occluded_fall_arms_by_displacement_and_alarms_once(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=2.4), make_config())
        frames = list(fall_frames(1.0, 150, fall_sec=0.0, hip0=0.40, hip1=0.60,
                                  gap=(1.0, 2.0)))
        tracks, events = replay(tracker, frames)
        speeds = [s[2] for s in tracks if s is not None]
        self.assertLess(max(speeds), 0.25)
        self.assertEqual(sum(events), 1)

    def test_late_temporal_positive_while_still_lying_confirms_once(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=4.0), make_config())
        tracks, events = replay(tracker, fall_frames(1.0, 150))
        self.assertEqual(sum(events), 1)

    def test_late_temporal_positive_after_standing_up_does_not_alarm(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=4.0), make_config())
        tracks, events = replay(tracker, fall_frames(1.0, 150, stand_up_at=3.0))
        self.assertFalse(any(events))
        self.assertEqual(tracks[-1][0], "normal")

    def test_min_features_three_confirms_stationary_lying_victim(self):
        tracker = APP.MultiPersonTracker(FakeBridge(positive_from=1.9),
                                         make_config(min_suspected_features=3))
        tracks, events = replay(tracker, fall_frames(1.0, 150))
        self.assertEqual(sum(events), 1)
        edge = events.index(True)
        self.assertEqual(tracks[edge][2], 0.0)


if __name__ == "__main__":
    unittest.main()
