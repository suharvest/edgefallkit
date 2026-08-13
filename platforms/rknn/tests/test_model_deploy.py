import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PREPARE = ROOT / "prepare_model.sh"
DEPLOY = ROOT / "deploy.sh"


def run(script, *args, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(["sh", str(script), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=merged)


class ModelPreparationTest(unittest.TestCase):
    def test_license_gate_precedes_work(self):
        result = run(PREPARE, "--platform", "rk3576", "--dry-run")
        self.assertEqual(result.returncode, 3)
        self.assertIn("license gate", result.stderr)

    def test_verified_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "yolo11n_pose_rawhead_fp16.rk3576.rknn"
            payload = b"cached-rknn"
            (root / name).write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (root / "SHA256SUMS").write_text(f"{digest}  {name}\n")
            result = run(PREPARE, "--platform", "rk3576", "--output-dir", directory,
                         "--accept-upstream-license", "--offline")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cache hit", result.stdout)

    def test_offline_missing_model_fails_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(PREPARE, "--platform", "rk3588", "--output-dir", directory,
                         "--accept-upstream-license", "--offline")
            self.assertEqual(result.returncode, 4)
            self.assertIn("offline mode", result.stderr)

    def test_model_url_requires_https_and_known_artifact_type(self):
        result = run(PREPARE, "--platform", "rk3588", "--accept-upstream-license",
                     "--model-url", "http://example.test/model.rknn", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must use HTTPS", result.stderr)
        result = run(PREPARE, "--platform", "rk3588", "--accept-upstream-license",
                     "--model-url", "https://example.test/model.bin", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must end in .rknn or .onnx", result.stderr)
        result = run(PREPARE, "--platform", "rk3588", "--accept-upstream-license",
                     "--model-url", "https://huggingface.co/acme/model.rknn", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("hf-mirror.com", result.stderr)

    def test_x86_dry_run_describes_official_export(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(PREPARE, "--platform", "rk3588", "--output-dir", directory,
                         "--accept-upstream-license", "--dry-run",
                         env={"RKNN_TEST_ARCH": "x86_64"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("official yolo11n-pose.pt", result.stdout)
            self.assertIn("RKNN Toolkit 2.3.2", result.stdout)

    def test_prebuilt_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.rknn"
            source.write_bytes(b"custom")
            output = root / "out"
            result = run(PREPARE, "--platform", "rk3576", "--output-dir", str(output),
                         "--accept-upstream-license", "--model-file", str(source),
                         "--model-sha256", "0" * 64)
            self.assertEqual(result.returncode, 6)
            self.assertIn("SHA256 mismatch", result.stderr)

    def test_new_prebuilt_preserves_temporal_manifest_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.rknn"
            source.write_bytes(b"custom")
            output = root / "out"
            output.mkdir()
            manifest = output / "SHA256SUMS"
            manifest.write_text("abc  temporal-rk3576.npz\n")
            result = run(PREPARE, "--platform", "rk3576", "--output-dir", str(output),
                         "--accept-upstream-license", "--model-file", str(source))
            self.assertEqual(result.returncode, 0, result.stderr)
            text = manifest.read_text()
            self.assertIn("abc  temporal-rk3576.npz", text)
            self.assertIn("yolo11n_pose_rawhead_fp16.rk3576.rknn", text)

    def test_arm_refuses_local_conversion_with_actionable_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(PREPARE, "--platform", "rk3576", "--output-dir", directory,
                         "--accept-upstream-license",
                         env={"RKNN_TEST_ARCH": "aarch64"})
            self.assertEqual(result.returncode, 5)
            self.assertIn("x86_64-only", result.stderr)
            self.assertIn("--builder-host X86_FLEET_DEVICE", result.stderr)

    def test_arm_remote_builder_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(PREPARE, "--platform", "rk3576", "--output-dir", directory,
                         "--accept-upstream-license", "--builder-host", "builder-x86",
                         "--dry-run", env={"RKNN_TEST_ARCH": "aarch64"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("fleet exec builder-x86", result.stdout)
            self.assertIn("fleet pull builder-x86", result.stdout)

    def test_deploy_dry_run_uses_fleet_and_pinned_runtime(self):
        result = run(DEPLOY, "--platform", "rk3576", "--device", "cat-remote",
                     "--accept-upstream-license", "--dry-run", "--no-up")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verify remote SHA256 or fleet push cat-remote", result.stdout)
        self.assertIn("fall-detection-rknn:0.1.0-rc2", result.stdout)
        self.assertIn("verify RepoDigests", result.stdout)
        self.assertNotIn("compose up", result.stdout)

    def test_offline_no_up_executes_full_flow_with_pinned_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            pose_name = "yolo11n_pose_rawhead_fp16.rk3588.rknn"
            temporal_name = "temporal-rk3588.npz"
            pose = b"isolated-rknn-cache"
            temporal = b"isolated-temporal-cache"
            (models / pose_name).write_bytes(pose)
            (models / temporal_name).write_bytes(temporal)
            (models / "SHA256SUMS").write_text(
                f"{hashlib.sha256(pose).hexdigest()}  {pose_name}\n"
                f"{hashlib.sha256(temporal).hexdigest()}  {temporal_name}\n"
            )
            log = root / "fleet.log"
            fake = root / "fleet"
            fake.write_text(
                "#!/bin/sh\n"
                f"echo \"$*\" >> '{log}'\n"
                "case \"$*\" in *\"docker image inspect\"*) "
                "echo '[\"sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn@"
                "sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8\"]';; esac\n")
            fake.chmod(0o755)
            result = run(DEPLOY, "--platform", "rk3588", "--device", "radxa",
                         "--accept-upstream-license", "--offline", "--no-up",
                         "--models-dir", str(models),
                         env={"FLEET_BIN": str(fake)})
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text()
            self.assertIn("docker image inspect", calls)
            self.assertIn("docker compose", calls)
            self.assertIn("--validate", calls)
            self.assertNotIn("up -d", calls)


if __name__ == "__main__":
    unittest.main()
