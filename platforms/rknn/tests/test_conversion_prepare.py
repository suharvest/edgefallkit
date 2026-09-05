import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConversionPrepareTests(unittest.TestCase):
    def test_aligned_onnx_sidecar_has_overwrite_guard(self):
        mod = load("prepare_aligned_sidecar_guard", "tools/prepare_aligned_onnx.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.onnx"; source.write_bytes(b"not parsed")
            target = root / "target.onnx"
            Path(str(target) + ".json").write_text("old")
            argv = ["prepare_aligned_onnx.py", "--onnx", str(source), "--out", str(target)]
            with patch.object(sys, "argv", argv), \
                 self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                mod.main()

    def test_strict_mode_requires_sidecar_before_runtime_import(self):
        mod = load("convert_strict_missing", "tools/convert_pose_rknn.py")
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "model.onnx"
            source.write_bytes(b"no sidecar")
            with self.assertRaisesRegex(SystemExit, "requires prepare_aligned_onnx sidecar"):
                mod.main(["--onnx", str(source), "--out", str(Path(td) / "out.rknn"),
                          "--platform", "rk3576", "--strict-9-head"])

    def test_calibration_strict_identity_and_mutations(self):
        import cv2
        import numpy as np
        prepare = load("prepare_strict_fixture", "tools/prepare_calibration.py")
        converter = load("convert_strict_fixture", "tools/convert_pose_rknn.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "subject-1-ADL-01.png"
            cv2.imwrite(str(source), np.zeros((16, 32, 3), np.uint8))
            listing = root / "sources.txt"
            listing.write_text("# relative paths use the list directory\n" + source.name + "\n")
            prepare.main(["--source-list", str(listing), "--out-dir", str(root / "out")])
            calibration = root / "out/calibration.txt"
            sidecar = Path(str(calibration) + ".json")
            original = sidecar.read_text()
            converter._validate_calibration(calibration)
            derived = Path(json.loads(original)["images"][0]["derived"])
            derived.write_bytes(b"changed")
            with self.assertRaisesRegex(SystemExit, "image hash mismatch"):
                converter._validate_calibration(calibration)
            # Even a newly hashed list cannot detach actual rows from records.
            calibration.write_text(str(source) + "\n")
            metadata = json.loads(original)
            metadata["calibration_manifest_sha256"] = converter.sha256_file(calibration)
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(SystemExit, "rows do not match"):
                converter._validate_calibration(calibration)
            calibration.write_text(str(derived) + "\n")
            s4 = root / "subject-4-Fall-01.png"
            s4.write_bytes(source.read_bytes())
            listing.write_text(str(s4) + "\n")
            metadata = json.loads(original)
            metadata["source_list_sha256"] = converter.sha256_file(listing)
            metadata["images"][0]["source"] = str(s4)
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(SystemExit, "Subject 4"):
                converter._validate_calibration(calibration)

    def test_convert_help_without_rknn(self):
        mod = load("convert_pose_rknn", "tools/convert_pose_rknn.py")
        with self.assertRaises(SystemExit) as ctx:
            mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_precision_and_overwrite_guards_happen_before_import(self):
        mod = load("convert_pose_rknn_guard", "tools/convert_pose_rknn.py")
        with tempfile.TemporaryDirectory() as td:
            onnx = Path(td) / "model.onnx"
            out = Path(td) / "out.rknn"
            onnx.write_bytes(b"not-a-real-onnx")
            with self.assertRaisesRegex(SystemExit, "requires --dataset"):
                mod.main(["--onnx", str(onnx), "--out", str(out), "--platform", "rk3576",
                          "--precision", "int8"])
            with self.assertRaisesRegex(SystemExit, "cannot be combined"):
                mod.main(["--onnx", str(onnx), "--out", str(out), "--platform", "rk3576",
                          "--precision", "fp16", "--dataset", str(onnx)])

    def test_calibration_rejects_s4_and_unparseable(self):
        mod = load("prepare_calibration_reject", "tools/prepare_calibration.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "subject-4-Fall-01.jpg"
            image.write_bytes(b"x")
            listing = root / "sources.txt"
            listing.write_text(str(image) + "\n")
            with self.assertRaisesRegex(SystemExit, "Subject 4"):
                mod.main(["--source-list", str(listing), "--out-dir", str(root / "out")])
            image2 = root / "mystery.jpg"
            image2.write_bytes(b"x")
            listing.write_text(str(image2) + "\n")
            with self.assertRaisesRegex(SystemExit, "cannot identify"):
                mod.main(["--source-list", str(listing), "--out-dir", str(root / "out2")])

    def test_calibration_records_hashes_and_bgr(self):
        mod = load("prepare_calibration_ok", "tools/prepare_calibration.py")
        fake_cv2 = types.SimpleNamespace(
            IMREAD_COLOR=1, INTER_LINEAR=2, BORDER_CONSTANT=3,
            imread=lambda *_: types.SimpleNamespace(shape=(480, 640, 3), size=921600),
            resize=lambda image, size, interpolation: types.SimpleNamespace(shape=(size[1], size[0], 3)),
            copyMakeBorder=lambda image, *args, **kwargs: types.SimpleNamespace(shape=(640, 640, 3)),
            imwrite=lambda path, image: (Path(path).write_bytes(b"png") or True),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "subject-1-ADL-01.jpg"
            image.write_bytes(b"source")
            listing = root / "sources.txt"
            listing.write_text(str(image) + "\n")
            old = sys.modules.get("cv2")
            sys.modules["cv2"] = fake_cv2
            try:
                mod.main(["--source-list", str(listing), "--out-dir", str(root / "out")])
            finally:
                if old is None:
                    sys.modules.pop("cv2", None)
                else:
                    sys.modules["cv2"] = old
            metadata = json.loads((root / "out/calibration.txt.json").read_text())
            self.assertEqual(metadata["subjects"], [1])
            self.assertEqual(metadata["images"][0]["derived_color"], "BGR")
            self.assertEqual(metadata["images"][0]["size"], [640, 640])
            self.assertTrue(metadata["images"][0]["derived_sha256"])


if __name__ == "__main__":
    unittest.main()
