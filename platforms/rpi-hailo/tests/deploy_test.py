#!/usr/bin/env python3
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy.sh"
PRODUCTION_SHA256 = "e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d"


class DeployTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.compose = self.root / "compose.yml"
        self.compose.write_text("services:\n  fall-detection:\n    image: example:test\n")
        self.good = b"test-only-hef-content"
        self.expected = hashlib.sha256(self.good).hexdigest()
        self.deploy = self.root / "deploy.sh"
        source = DEPLOY.read_text()
        self.assertIn(PRODUCTION_SHA256, source)
        self.deploy.write_text(source.replace(PRODUCTION_SHA256, self.expected, 1))
        self.deploy.chmod(0o755)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.docker_log = self.root / "docker.log"
        docker = self.bin / "docker"
        docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n")
        docker.chmod(0o755)
        curl = self.bin / "curl"
        curl.write_text("""#!/bin/sh
set -eu
out=
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then out=$2; shift; fi
  shift
done
cp "$CURL_FIXTURE" "$out"
""")
        curl.chmod(0o755)
        self.env = os.environ | {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "FALL_HAILO_MODEL_DIR": str(self.models),
            "FALL_HAILO_COMPOSE_FILE": str(self.compose),
            "DOCKER_LOG": str(self.docker_log),
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_deploy(self, *args, env=None):
        return subprocess.run(
            [str(self.deploy), *args], text=True, capture_output=True,
            env=self.env if env is None else env, check=False)

    def docker_calls(self):
        return self.docker_log.read_text().splitlines() if self.docker_log.exists() else []

    def test_license_gate_prevents_all_actions(self):
        result = self.run_deploy("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--accept-upstream-license", result.stderr)
        self.assertEqual(self.docker_calls(), [])

    def test_dry_run_has_no_side_effects(self):
        result = self.run_deploy("--accept-upstream-license", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN", result.stdout)
        self.assertIn("hailo-model-zoo.s3.eu-west-2.amazonaws.com", result.stdout)
        self.assertFalse((self.models / "yolov8s_pose.hef").exists())
        self.assertEqual(self.docker_calls(), [])

    def test_verified_cache_skips_download_and_starts_offline(self):
        (self.models / "yolov8s_pose.hef").write_bytes(self.good)
        result = self.run_deploy("--accept-upstream-license", "--offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache hit", result.stdout)
        self.assertEqual(self.docker_calls(), [
            f"compose -f {self.compose} config --quiet",
            f"compose -f {self.compose} up -d --pull never --no-build fall-detection",
        ])

    def test_bad_local_sha_never_starts(self):
        bad = self.root / "bad.hef"
        bad.write_bytes(b"wrong")
        result = self.run_deploy(
            "--accept-upstream-license", "--offline", "--hef", str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertEqual(self.docker_calls(), [])
        self.assertFalse((self.models / "yolov8s_pose.hef").exists())

    def test_download_is_verified_then_atomically_installed(self):
        fixture = self.root / "download.hef"
        fixture.write_bytes(self.good)
        env = self.env | {"CURL_FIXTURE": str(fixture)}
        result = self.run_deploy("--accept-upstream-license", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.models / "yolov8s_pose.hef").read_bytes(), self.good)
        self.assertFalse(any(self.models.glob(".yolov8s_pose.hef.tmp.*")))
        self.assertEqual(self.docker_calls(), [
            f"compose -f {self.compose} pull fall-detection",
            f"compose -f {self.compose} config --quiet",
            f"compose -f {self.compose} up -d --pull never --no-build fall-detection",
        ])


if __name__ == "__main__":
    unittest.main()
