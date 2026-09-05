from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import probe_runtime  # noqa: E402


def heads(layout="NCHW"):
    values = []
    for grid in (80, 40, 20):
        for channels in (64, 1, 51):
            shape = (1, channels, grid, grid) if layout == "NCHW" else (1, grid, grid, channels)
            values.append(np.full(shape, grid + channels, np.float32))
    return values


class Runner:
    closed = False
    def __init__(self, _model): pass
    def infer(self, _frame): return heads(), 3.5
    def close(self): type(self).closed = True


def test_contract_accepts_complete_nchw_and_nhwc():
    assert len(probe_runtime.output_contract(heads("NCHW"))) == 9
    assert {row["layout"] for row in probe_runtime.output_contract(heads("NHWC"))} == {"NHWC"}


@pytest.mark.parametrize("outputs,match", [
    (heads()[:-1], "exactly 9"),
    (heads()[:-1] + [np.ones((1, 7, 20, 20), np.float32)], "unsupported"),
    (heads()[:-1] + [heads()[-2]], "duplicate"),
])
def test_contract_rejects_wrong_head_sets(outputs, match):
    with pytest.raises(RuntimeError, match=match):
        probe_runtime.output_contract(outputs)


def test_run_hashes_files_arrays_and_closes(tmp_path):
    Runner.closed = False
    model = tmp_path / "model.rknn"; model.write_bytes(b"model")
    input_path = tmp_path / "input.npy"; np.save(input_path, np.zeros((640, 640, 3), np.uint8))
    report_path = tmp_path / "report.json"
    args = SimpleNamespace(model=str(model), input_npy=str(input_path), warmup=2, core_mask="auto",
                           iterations=5, report=report_path)
    report = probe_runtime.run(args, Runner)
    assert report["ok"] and report["output_count"] == 9
    assert report["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert report["input_file_sha256"] != report["input_array_sha256"]
    assert report["rknn_call_ms"]["samples"] == 5
    assert report["steady_output_hash_stable"]
    assert len(report["output_signature_sha256"]) == 64
    assert Runner.closed


def test_help_does_not_import_hardware():
    process = subprocess.run([sys.executable, str(ROOT / "tools/probe_runtime.py"), "--help"],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.returncode == 0, process.stderr
    assert "--input-npy" in process.stdout
