"""Preflight tests for RK temporal profile trace identity and S1-3 isolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "platforms/rknn"))
from tools import train_temporal_profile as profile  # noqa: E402


COUNTS = {1: 32, 2: 48, 3: 43}


def make_corpus(tmp_path: Path):
    traces = tmp_path / "traces"
    model = tmp_path / "pose.rknn"
    model.write_bytes(b"pose-model")
    completed = {}
    for subject, count in COUNTS.items():
        for index in range(1, count + 1):
            kind = "ADL" if index <= count // 2 else "Fall"
            number = index if kind == "ADL" else index - count // 2
            key = f"subject-{subject}/{kind}/{number:02d}.mp4"
            trace = traces / Path(key).with_suffix(".jsonl")
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("{}\n")
            completed[key] = {"trace_sha256": profile.sha256(trace)}
    manifest = {"platform": "rk3576", "model_sha256": profile.sha256(model),
                "subjects": [1, 2, 3], "complete": True, "failed": {},
                "extraction_identity": {"input_size": 640}, "completed": completed}
    (traces / "extraction-manifest.json").write_text(json.dumps(manifest))
    return traces, model


def test_valid_complete_s1_s3_manifest(tmp_path):
    traces, model = make_corpus(tmp_path)
    manifest = profile.validate_trace_identity(traces, model, "rk3576")
    assert len(manifest["completed"]) == 123


@pytest.mark.parametrize("mutation", ("changed", "missing", "extra", "s4"))
def test_trace_identity_rejects_invalid_inventory(tmp_path, mutation):
    traces, model = make_corpus(tmp_path)
    path = traces / "extraction-manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "changed":
        key = next(iter(manifest["completed"]))
        trace = traces / Path(key).with_suffix(".jsonl")
        trace.write_text("changed\n")
    elif mutation == "missing":
        manifest["completed"].pop(next(iter(manifest["completed"])))
    elif mutation == "extra":
        manifest["completed"]["subject-1/ADL/99.mp4"] = {"trace_sha256": "x"}
    else:
        manifest["subjects"] = [1, 2, 3, 4]
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        profile.validate_trace_identity(traces, model, "rk3576")


def test_wrong_model_rejected(tmp_path):
    traces, model = make_corpus(tmp_path)
    wrong = tmp_path / "wrong.rknn"
    wrong.write_bytes(b"other")
    with pytest.raises(SystemExit, match="platform/model"):
        profile.validate_trace_identity(traces, wrong, "rk3576")


def test_development_loader_does_not_read_subject4_annotations(tmp_path):
    traces = tmp_path / "traces"
    dataset = tmp_path / "dataset"
    for subject in (1, 2, 3):
        path = dataset / f"subject-{subject}" / "Fall.csv"
        path.parent.mkdir(parents=True)
        path.write_text(f"subject-{subject}\n")
    trap = dataset / "subject-4" / "Fall.csv"
    trap.parent.mkdir(parents=True)
    trap.write_text("MUST NOT BE READ\n")

    class Training:
        def load_clips(self, _traces, view, allowed):
            assert allowed == {1, 2, 3}
            assert not (view / "subject-4").exists()
            return ["s1-3"]

    assert profile.load_development_clips(Training(), traces, dataset) == ["s1-3"]
