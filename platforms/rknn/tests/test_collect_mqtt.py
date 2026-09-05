import sys
import unittest
import io
import queue
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from collect_mqtt import _reader, summarize, parse_args


def payload(stream, frame, inference=10.0, pipeline=20.0, timestamp=None):
    now = frame * 1000 if timestamp is None else timestamp
    features = {"valid": False, "hip_drop_speed": 0, "hip_drop_distance": 0,
                "torso_angle_deg": 0, "bbox_aspect_ratio": 0}
    return {"timestamp": now, "frame_id": frame, "inference_time_ms": int(inference),
            "stream_id": stream, "fall_detected": False, "fall_event": False,
            "event_id": 0, "global_event_id": 0, "event_id_scope": "stream_global_event_id",
            "state": "normal", "person_detected": False, "person_count": 0,
            "fallen_count": 0, "tracking": False, "features": features, "keypoints": [],
            "pose17": [], "persons": [], "inference_ms": inference, "pipeline_ms": pipeline}


def record(stream, frame, ns):
    return {"received_ms": ns // 1_000_000, "received_monotonic_ns": ns,
            "payload": payload(stream, frame)}


class CollectorSummaryTest(unittest.TestCase):
    def test_reader_records_both_clocks_at_receive_time(self):
        output = queue.Queue()
        with patch("collect_mqtt.time.monotonic_ns", return_value=123), \
             patch("collect_mqtt.time.time_ns", return_value=456_000_000):
            _reader(io.StringIO("{}\n"), output)
        self.assertEqual(output.get(), ("{}\n", 123, 456_000_000))
        self.assertIsNone(output.get())

    def test_zero_missing_and_low_fps_fail(self):
        result = summarize([], ["bad"], 0, window_seconds=10,
                           expected_streams=["a", "b"], min_fps=14.5)
        self.assertFalse(result["pass"])
        self.assertIn("zero valid messages", result["pass_reasons"])
        self.assertEqual(result["missing_streams"], ["a", "b"])
        result = summarize([record("a", i, i * 1_000_000_000) for i in range(10)], [], 0,
                           window_seconds=1, expected_streams=["a"], min_fps=14.5)
        self.assertFalse(result["pass"])
        self.assertAlmostEqual(result["streams"]["a"]["window_fps"], 10)

    def test_exact_fps_passes_and_warmup_is_represented_by_input_scope(self):
        result = summarize([record("a", i, i * 1_000_000_000) for i in range(15)], [], 0,
                           window_seconds=1, expected_streams=["a"], min_fps=14.5)
        self.assertTrue(result["pass"])
        self.assertEqual(result["scope"], "fixed monotonic wall-clock window")
        self.assertIn("inference_ms", result["streams"]["a"])
        self.assertIn("p95", result["streams"]["a"]["pipeline_ms"])

    def test_legacy_fields_and_observed_span_remain(self):
        result = summarize([record("a", 1, 0), record("a", 2, 2_000_000_000)], [], 0,
                           elapsed_sec=2, mode="count")
        self.assertEqual(result["scope"], "legacy count")
        self.assertEqual(result["streams"]["a"]["mqtt_observed_fps"], .5)
        self.assertIsNone(result["window_seconds"])

    def test_cli_validation(self):
        base = ["--topic", "x", "--raw-out", "r", "--summary-out", "s"]
        with self.assertRaises(SystemExit):
            parse_args(base + ["--count", "1", "--expected-streams", "a"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--duration-seconds", "1", "--min-fps", "-1"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--duration-seconds", "1", "--port", "0"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--count", "2", "--warmup-seconds", "1"])
        args = parse_args(base + ["--duration-seconds", "2", "--warmup-seconds", "1",
                                  "--expected-streams", "a,b", "--port", "1884"])
        self.assertEqual(args.expected_streams, ["a", "b"])
        self.assertEqual(args.port, 1884)


if __name__ == "__main__":
    unittest.main()
