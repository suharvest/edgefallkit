import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from video_source import FallbackVideoSource, validate_backend_config


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
