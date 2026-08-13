import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from rknn_pose import PoseDecoder, decode_pose_numpy


def synthetic_outputs(layout="nchw"):
    outputs = []
    for level, grid in enumerate((8, 4, 2)):
        cell = min(level, grid - 1)
        box = np.zeros((1, 64, grid, grid), np.float32)
        score = np.zeros((1, 1, grid, grid), np.float32)
        keypoints = np.zeros((1, 51, grid, grid), np.float32)
        score[0, 0, cell, cell] = 0.91 - level * 0.1
        keypoints[0, 2::3, cell, cell] = 2.0 - level
        triplet = (box, score, keypoints)
        if layout == "nhwc":
            triplet = tuple(x.transpose(0, 2, 3, 1) for x in triplet)
        outputs.extend(triplet)
    return outputs


class PostprocessTest(unittest.TestCase):
    def assert_detections_close(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a["score"], b["score"], places=5)
            np.testing.assert_allclose(a["box"], b["box"], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(a["keypoints"], b["keypoints"], rtol=2e-5, atol=2e-5)

    def test_cpp_matches_numpy_nchw_and_nhwc(self):
        try:
            decoder = PoseDecoder({"backend": "cpp", "strict": True, "fallback": "none"})
        except RuntimeError as exc:
            self.skipTest(str(exc))
        for layout in ("nchw", "nhwc"):
            outputs = synthetic_outputs(layout)
            expected = decode_pose_numpy(outputs, 0.35, 0.45, 64)
            actual = decoder.decode(outputs, 0.35, 0.45, 64)
            self.assert_detections_close(expected, actual)

    def test_cpp_runtime_failure_falls_back_to_numpy(self):
        previous = sys.modules.get("rknn_postprocess")
        fake = types.SimpleNamespace(decode_pose=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
        sys.modules["rknn_postprocess"] = fake
        try:
            decoder = PoseDecoder({"backend": "cpp", "strict": False, "fallback": "numpy"})
            outputs = synthetic_outputs()
            self.assertEqual(decoder.decode(outputs, input_size=64), decode_pose_numpy(outputs, input_size=64))
            self.assertEqual(decoder.active_backend, "numpy")
        finally:
            if previous is None:
                sys.modules.pop("rknn_postprocess", None)
            else:
                sys.modules["rknn_postprocess"] = previous

    def test_strict_cpp_runtime_failure_is_raised(self):
        previous = sys.modules.get("rknn_postprocess")
        sys.modules["rknn_postprocess"] = types.SimpleNamespace(
            decode_pose=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            decoder = PoseDecoder({"backend": "cpp", "strict": True, "fallback": "none"})
            with self.assertRaisesRegex(RuntimeError, "boom"):
                decoder.decode(synthetic_outputs(), input_size=64)
        finally:
            if previous is None:
                sys.modules.pop("rknn_postprocess", None)
            else:
                sys.modules["rknn_postprocess"] = previous


if __name__ == "__main__":
    unittest.main()
