import hashlib
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import benchmark


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            value = self.value
            self.value += 0.00001
            return value


class FakeRunner:
    made = []

    def __init__(self, model, core_mask):
        self.model, self.core_mask, self.closed, self.calls = model, core_mask, False, 0
        self.made.append(self)

    def infer(self, frame):
        self.calls += 1
        outputs = []
        for grid in (80, 40, 20):
            for channels in (64, 1, 51):
                outputs.append(np.ones((1, channels, grid, grid), dtype=np.float32))
        return outputs, 2.5

    def close(self):
        self.closed = True


class FakeDecoder:
    def __init__(self, config):
        self.active_backend = "fake"

    def decode(self, outputs):
        return [{"score": 1.0}]


def args_for(model, **overrides):
    values = dict(model=model, contexts=1, iterations=4, iterations_explicit=True,
                  warmup=1, warmup_seconds=None, duration_seconds=None, repetitions=1,
                  core_mask="auto", core_masks=None, input_npy=None, image=None, postprocess_backend="numpy")
    values.update(overrides)
    return SimpleNamespace(**values)


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        FakeRunner.made = []

    def model(self, directory):
        path = Path(directory) / "model.rknn"
        path.write_bytes(b"fake-rknn")
        return str(path)

    def test_legacy_iterations_and_each_context_has_own_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark.run_benchmark(
                args_for(self.model(directory), contexts=3, iterations=7),
                FakeRunner, FakeDecoder, FakeClock())
        self.assertEqual(result["iterations"], 7)
        self.assertEqual(result["samples"], 7)
        self.assertEqual(sorted(r.calls for r in FakeRunner.made), [3, 3, 4])  # warmup + work
        self.assertTrue(all(r.closed for r in FakeRunner.made))
        self.assertEqual(result["core_mask"], "auto")
        self.assertEqual(len(result["repetitions_detail"]), 1)
        self.assertIn("rknn_call_ms", result["metric_scopes"])
        self.assertIn("input_array_sha256", result)

    def test_duration_repetitions_and_core_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark.run_benchmark(
                args_for(self.model(directory), contexts=1, iterations=None,
                         iterations_explicit=False, warmup_seconds=.03,
                         duration_seconds=.05, repetitions=3, core_mask="0x3"),
                FakeRunner, FakeDecoder, FakeClock())
        self.assertEqual(result["repetitions"], 3)
        self.assertEqual(result["contexts"], 1)
        self.assertGreater(result["samples"], 0)
        self.assertEqual(len(result["repetitions_detail"]), 3)
        self.assertTrue(all(r.core_mask == 3 for r in FakeRunner.made))
        self.assertTrue(all(r.closed for r in FakeRunner.made))

    def test_per_context_core_masks_are_assigned_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark.run_benchmark(
                args_for(self.model(directory), contexts=2, iterations=4,
                         core_masks=[1, 2]), FakeRunner, FakeDecoder, FakeClock())
        self.assertEqual(sorted(runner.core_mask for runner in FakeRunner.made), [1, 2])
        self.assertEqual(result["core_masks"], [1, 2])
        self.assertEqual([c["core_mask"] for c in result["repetitions_detail"][0]["contexts"]], [1, 2])

    def test_close_on_inference_failure_and_rejects_nonfinite_or_empty(self):
        class Broken(FakeRunner):
            def infer(self, frame):
                self.calls += 1
                return [], 1.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exactly 9"):
                benchmark.run_benchmark(args_for(self.model(directory), warmup=0),
                                        Broken, FakeDecoder, FakeClock())
        self.assertTrue(FakeRunner.made[0].closed)

    def test_invalid_values_and_duration_iteration_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)
            for kwargs in ({"contexts": 0}, {"iterations": 0}, {"repetitions": 0},
                           {"contexts": 2, "iterations": 1},
                           {"duration_seconds": 0, "iterations": None, "iterations_explicit": False},
                           {"core_mask": "0"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    benchmark.run_benchmark(args_for(model, **kwargs), FakeRunner, FakeDecoder, FakeClock())
            with self.assertRaisesRegex(ValueError, "one positive integer"):
                benchmark.run_benchmark(args_for(model, contexts=2, core_masks=[1]),
                                        FakeRunner, FakeDecoder, FakeClock())
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                benchmark.run_benchmark(args_for(model, contexts=2, core_mask="0x3",
                                                  core_masks=[1, 2]), FakeRunner, FakeDecoder, FakeClock())
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                benchmark.run_benchmark(args_for(model, duration_seconds=1),
                                        FakeRunner, FakeDecoder, FakeClock())

    def test_hashes_and_help_do_not_import_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)
            result = benchmark.run_benchmark(args_for(model, warmup=0), FakeRunner, FakeDecoder, FakeClock())
        self.assertEqual(result["model_sha256"], hashlib.sha256(b"fake-rknn").hexdigest())
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertEqual(result["input_sha256"], result["input_array_sha256"])
        self.assertIsNone(result["input_file_sha256"])
        proc = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--help"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--duration-seconds", proc.stdout)

    def test_rejects_duplicate_or_wrong_pose_heads(self):
        values = []
        for grid in (80, 40, 20):
            for channels in (64, 1, 51):
                values.append(np.ones((1, channels, grid, grid), np.float32))
        with self.assertRaisesRegex(RuntimeError, "pose head set mismatch"):
            benchmark._output_contract(values[:-1] + [values[-2]])


if __name__ == "__main__":
    unittest.main()
