import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from video_source import (FallbackVideoSource, aspect_fit_geometry,
                          copy_strided_rgb_to_letterbox, pad_scaled_rgb,
                          validate_backend_config)


class FakeSource:
    def __init__(self, name, values):
        self.backend_name = name
        self.values = iter(values)
        self.closed = False

    def read(self):
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class BackendAndComposeTest(unittest.TestCase):
    def test_aspect_fit_geometry_matches_offline_letterbox(self):
        expected = {
            (1280, 720): (640, 360, 0, 140),
            (640, 480): (640, 480, 0, 80),
            (640, 640): (640, 640, 0, 0),
        }
        for source, geometry in expected.items():
            with self.subTest(source=source):
                self.assertEqual(aspect_fit_geometry(*source, 640), geometry)

    def test_scaled_rgb_is_padded_without_stretch_or_extra_frame_copy(self):
        for source_w, source_h in ((1280, 720), (640, 480), (640, 640)):
            with self.subTest(source=(source_w, source_h)):
                scaled_w, scaled_h, left, top = aspect_fit_geometry(
                    source_w, source_h, 640)
                scaled = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
                scaled[0, 0] = (1, 2, 3)
                scaled[-1, -1] = (4, 5, 6)
                canvas = pad_scaled_rgb(scaled, 640)
                self.assertEqual(canvas.shape, (640, 640, 3))
                self.assertEqual(canvas.dtype, np.uint8)
                np.testing.assert_array_equal(canvas[top, left], (1, 2, 3))
                np.testing.assert_array_equal(
                    canvas[top + scaled_h - 1, left + scaled_w - 1], (4, 5, 6))
                if top:
                    self.assertTrue(np.all(canvas[:top] == 114))
                    self.assertTrue(np.all(canvas[top + scaled_h:] == 114))

    def test_strided_mapped_rgb_is_copied_once_into_owned_canvas(self):
        width, height, stride = 5, 3, 20
        mapped = bytearray(stride * height)
        view = np.ndarray((height, width, 3), dtype=np.uint8, buffer=mapped,
                          strides=(stride, 3, 1))
        view[:] = (7, 8, 9)
        canvas = copy_strided_rgb_to_letterbox(mapped, width, height, stride, 8)
        mapped[:] = b"\x00" * len(mapped)
        np.testing.assert_array_equal(canvas[2, 1], (7, 8, 9))
        self.assertTrue(np.all(canvas[:2] == 114))
        self.assertTrue(np.all(canvas[5:] == 114))

    def test_video_fallback_after_configured_failures(self):
        primary = FakeSource("gstreamer_mpp", [None, None])
        fallback = FakeSource("opencv_ffmpeg", ["frame"])
        source = FallbackVideoSource(primary, fallback, strict=False, failure_limit=2)
        self.assertIsNone(source.read())
        self.assertIsNone(source.read())
        self.assertEqual(source.active_backend, "opencv_ffmpeg")
        self.assertEqual(source.read(), "frame")
        self.assertTrue(primary.closed)

    def test_strict_video_does_not_fallback(self):
        source = FallbackVideoSource(FakeSource("gstreamer_mpp", [RuntimeError("bad")]),
                                     FakeSource("opencv_ffmpeg", ["frame"]), strict=True)
        with self.assertRaisesRegex(RuntimeError, "bad"):
            source.read()

    def test_config_and_compose_declare_backends_and_host_abi(self):
        root = Path(__file__).parents[2]
        for platform in ("rk3576", "rk3588"):
            cfg = json.loads((root / platform / "config/config.json").read_text())
            validate_backend_config(cfg)
            self.assertEqual(cfg["video"]["backend"], "gstreamer_mpp")
            self.assertEqual(cfg["postprocess"]["backend"], "cpp")
            compose = (root / platform / "docker-compose.yml").read_text()
            self.assertIn("${FALL_RK_IMAGE:-sensecraft-missionpack.seeed.cn/solution/"
                          "fall-detection-rknn:0.1.0-rc1}", compose)
            for library in ("libgstrockchipmpp.so", "libgstvideoparsersbad.so",
                            "libgstcodecparsers-1.0.so.0", "librockchip_mpp.so.1", "librga.so.2"):
                self.assertIn(library, compose)

    def test_runtime_image_contains_only_built_extension(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()
        self.assertIn("AS builder", dockerfile)
        self.assertIn("/build/rknn_postprocess*.so", dockerfile)
        runtime = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)[1]
        self.assertNotIn("cpp/rknn_postprocess.cpp", runtime)
        self.assertNotIn("pybind11", runtime)
        self.assertNotIn("Dockerfile.model-builder", runtime)
        self.assertNotIn("export_pose_rawhead_onnx.py", runtime)


if __name__ == "__main__":
    unittest.main()
