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


def absolute_outputs(layout="nchw", gx=3, gy=5, grid=8, input_size=64):
    """One detection at cell (gx, gy): DFL distances all equal 2 and every raw
    keypoint offset 0.25, so the keypoints must land exactly on the box centre
    ((gx+0.5)*stride, (gy+0.5)*stride)."""
    box = np.zeros((1, 64, grid, grid), np.float32)
    box[0, 2::16, gy, gx] = 10.0            # softmax -> bin 2 -> distance 2.0
    score = np.zeros((1, 1, grid, grid), np.float32)
    score[0, 0, gy, gx] = 0.91
    keypoints = np.full((1, 51, grid, grid), 0.25, np.float32)
    keypoints[0, 2::3] = 0.0
    keypoints[0, 2::3, gy, gx] = 2.0
    outputs = (box, score, keypoints)
    if layout == "nhwc":
        outputs = tuple(x.transpose(0, 2, 3, 1) for x in outputs)
    return outputs, (gx + 0.5) * (input_size // grid), (gy + 0.5) * (input_size // grid)


class AbsoluteDecodeTest(unittest.TestCase):
    def assert_absolute(self, detections, cx, cy):
        self.assertEqual(len(detections), 1)
        det = detections[0]
        x1, y1, x2, y2 = det["box"]
        self.assertAlmostEqual((x1 + x2) / 2, cx, places=4)
        self.assertAlmostEqual((y1 + y2) / 2, cy, places=4)
        for kx, ky, kc in det["keypoints"]:
            self.assertAlmostEqual(kx, cx, places=4)
            self.assertAlmostEqual(ky, cy, places=4)

    def test_numpy_keypoints_land_on_box_centre(self):
        for layout in ("nchw", "nhwc"):
            outputs, cx, cy = absolute_outputs(layout)
            self.assert_absolute(decode_pose_numpy(outputs, 0.35, 0.45, 64), cx, cy)

    def test_cpp_keypoints_land_on_box_centre(self):
        try:
            decoder = PoseDecoder({"backend": "cpp", "strict": True, "fallback": "none"})
        except RuntimeError as exc:
            self.skipTest(str(exc))
        for layout in ("nchw", "nhwc"):
            outputs, cx, cy = absolute_outputs(layout)
            self.assert_absolute(decoder.decode(outputs, 0.35, 0.45, 64), cx, cy)


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
