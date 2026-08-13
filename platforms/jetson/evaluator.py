#!/usr/bin/env python3
"""Offline Jetson evaluator for GMDCSA-24 and RealBiomFall.

The evaluator deliberately reuses the production ``TrtBridge`` and
``MultiPersonTracker`` from :mod:`app`: the only difference is that a video
file replaces the RTSP capture and MQTT is not started.  A video is processed
at a fixed 15 FPS timeline, with a fresh tracker/temporal state for every
clip.  The resulting JSONL contains one record per clip and ``summary.json``
contains the protocol metrics.

No torch/Ultralytics/ONNX Runtime dependency is used.  On Jetson the only
runtime data path is ``libjetson_fall_trt.so`` + the target TensorRT engine;
OpenCV/numpy are used for file decode and frame buffers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import cv2
except ImportError:  # Host fake tests can use a fake capture without OpenCV.
    cv2 = None

try:
    import numpy as np
except ImportError:  # The runtime path requires numpy through app.TrtBridge.
    np = None

try:
    from app import Detection, MultiPersonTracker, TrtBridge
except ImportError:  # Importing by absolute path is useful from an IDE/tests.
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from app import Detection, MultiPersonTracker, TrtBridge


FPS = 15.0
EARLY_MARGIN_SEC = 0.5
SMOKE_TEST_CLIPS = {
    ("ADL", "01"), ("ADL", "05"), ("ADL", "10"), ("ADL", "15"), ("ADL", "20"),
    ("Fall", "01"), ("Fall", "05"), ("Fall", "09"), ("Fall", "13"), ("Fall", "17"),
}
EXPECTED_GMDCSA_CLIPS = {
    "train": 80,
    "validation": 43,
    "freeze": 123,
    "subject4-clean": 27,
    "subject4-smoke": 10,
    "all": 160,
}
EXPECTED_REALBIOM_TESTING = 34


@dataclass(frozen=True)
class ClipSpec:
    path: Path
    label: int
    onset_sec: float = math.inf
    dataset: str = "unknown"
    split: str = "custom"
    subject: Optional[int] = None
    subset: Optional[str] = None
    clip_id: str = ""


@dataclass
class ClipResult:
    clip: str
    path: str
    dataset: str
    split: str
    subject: Optional[int]
    subset: Optional[str]
    label: int
    onset_sec: Optional[float]
    alert_sec: Optional[float] = None
    alert_track_id: Optional[int] = None
    detected: bool = False
    early_alert: bool = False
    latency_sec: Optional[float] = None
    sampled_frames: int = 0
    pose_frames: int = 0
    pose_coverage: float = 0.0
    mean_person_count: float = 0.0
    mean_keypoint_coverage: float = 0.0
    inference_ms_mean: Optional[float] = None
    inference_ms_p50: Optional[float] = None
    inference_ms_max: Optional[float] = None
    duration_sec: float = 0.0
    sample_fps: float = FPS
    error: Optional[str] = None


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    # Match train_temporal_model.py: undefined specificity/precision is 0.0
    # rather than raising or silently dropping the metric.
    return float(numerator) / max(float(denominator), 1.0)


def load_gmdcsa_onsets(root: Path) -> dict[tuple[int, str], float]:
    """Read the published ``Fall.csv`` onset convention used for training."""
    result: dict[tuple[int, str], float] = {}
    pattern = re.compile(r"fall(?:ing)?[^\[]*\[\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    for subject in range(1, 5):
        # download_gmdcsa24.sh writes subject-N/Fall.csv. Accept the nested
        # variant too, since some dataset mirrors keep CSVs under Fall/.
        candidates = [root / f"subject-{subject}" / "Fall.csv",
                      root / f"subject-{subject}" / "Fall" / "Fall.csv"]
        csv_path = next((path for path in candidates if path.is_file()), None)
        if csv_path is None:
            continue
        for raw in csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[1:]:
            name = raw.split(",", 1)[0].strip()
            if not re.fullmatch(r"\d{2}\.mp4", name):
                continue
            starts = [float(match.group(1)) for match in pattern.finditer(raw)]
            if starts:
                result[(subject, name)] = min(starts)
    return result


def _gmdcsa_split_matches(subject: int, category: str, number: str, split: str) -> bool:
    smoke = (category, number) in SMOKE_TEST_CLIPS
    if split == "train":
        return subject in {1, 2}
    if split == "validation":
        return subject == 3
    if split == "freeze":
        return subject in {1, 2, 3}
    if split == "subject4-clean":
        return subject == 4 and not smoke
    if split == "subject4-smoke":
        return subject == 4 and smoke
    if split == "all":
        return True
    raise ValueError(f"unknown GMDCSA split: {split}")


def load_gmdcsa_clips(root: Path, split: str = "subject4-clean",
                      strict: bool = True) -> list[ClipSpec]:
    """Resolve GMDCSA clips using the exact train/validation/clean-test split."""
    onsets = load_gmdcsa_onsets(root)
    clips: list[ClipSpec] = []
    pattern = re.compile(r"subject-(\d+)[/\\](ADL|Fall)[/\\](\d{2})\.mp4$", re.I)
    for path in sorted(root.glob("subject-*/*/*.mp4")):
        match = pattern.search(path.as_posix())
        if match is None:
            continue
        subject = int(match.group(1))
        category = "Fall" if match.group(2).lower() == "fall" else "ADL"
        number = match.group(3)
        if not _gmdcsa_split_matches(subject, category, number, split):
            continue
        label = int(category == "Fall")
        onset = onsets.get((subject, f"{number}.mp4"), math.nan)
        clips.append(ClipSpec(
            path=path, label=label, onset_sec=onset, dataset="gmdcsa24",
            split=split, subject=subject,
            clip_id=f"subject-{subject}/{category}/{number}.mp4"))
    expected = EXPECTED_GMDCSA_CLIPS.get(split)
    if strict and expected is not None and len(clips) != expected:
        raise ValueError(f"expected {expected} clips for GMDCSA {split}, found {len(clips)}")
    return clips


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("clips", "items", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError("manifest must be a JSON list or an object containing clips[]")
    return [row for row in value if isinstance(row, dict)]


def load_realbiomfall_clips(manifest: Path, subset: Optional[str] = "testing",
                            strict: bool = True) -> list[ClipSpec]:
    """Load the safe JSON manifest generated by prepare_realbiomfall.py."""
    clips: list[ClipSpec] = []
    for row in _manifest_rows(manifest):
        if subset is not None and str(row.get("upstream_subset")) != subset:
            continue
        raw_path = Path(str(row.get("path", "")))
        path = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"RealBiomFall video missing: {path}")
        onset = float(row.get("onset_sec", math.nan))
        clips.append(ClipSpec(
            path=path, label=int(row.get("label", 1)), onset_sec=onset,
            dataset="realbiomfall", split=str(subset or "all"),
            subset=str(row.get("upstream_subset", "")), clip_id=path.name))
    if strict and subset == "testing" and len(clips) != EXPECTED_REALBIOM_TESTING:
        raise ValueError(f"expected {EXPECTED_REALBIOM_TESTING} RealBiomFall testing clips, found {len(clips)}")
    return clips


def load_manifest_clips(manifest: Path, strict: bool = True) -> list[ClipSpec]:
    """Load a generic list[{path,label,onset_sec}] for smoke/regression runs."""
    clips: list[ClipSpec] = []
    for row in _manifest_rows(manifest):
        raw_path = Path(str(row.get("path", "")))
        path = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
        if strict and not path.is_file():
            raise FileNotFoundError(path)
        clips.append(ClipSpec(
            path=path, label=int(row.get("label", 0)),
            onset_sec=float(row.get("onset_sec", math.nan)), dataset="manifest",
            split="custom", subset=row.get("subset"), clip_id=path.name))
    return clips


def _capture_fps(capture: Any, fallback: float) -> float:
    if cv2 is None:
        property_id = 5  # CAP_PROP_FPS, avoids requiring cv2 in fake tests.
    else:
        property_id = cv2.CAP_PROP_FPS
    try:
        source_fps = float(capture.get(property_id))
    except (AttributeError, TypeError, ValueError):
        source_fps = fallback
    return source_fps if math.isfinite(source_fps) and source_fps > 0.1 else fallback


def sample_video_frames(capture: Any, sample_fps: float = FPS) -> Iterator[tuple[Any, float]]:
    """Yield source frames on a deterministic ``n/sample_fps`` timeline.

    Reading remains sequential, but a 30 FPS source contributes every second
    frame and a 29.97 FPS source is selected by timestamp. The timestamp sent
    to the tracker is never the decoder clock: it is exactly ``n/15`` (or the
    requested sample rate), matching train_temporal_model.py's 15 FPS/stride
    semantics.
    """
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be > 0")
    source_fps = _capture_fps(capture, sample_fps)
    frame_index = 0
    sample_index = 0
    next_target_sec = 0.0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        source_time = frame_index / source_fps
        if source_time + 1e-9 >= next_target_sec:
            yield frame, sample_index / sample_fps
            sample_index += 1
            next_target_sec = sample_index / sample_fps
        frame_index += 1


def _frame_shape(frame: Any) -> tuple[int, int]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("capture frame has no HxW shape")
    return int(shape[1]), int(shape[0])


def _visible_keypoint_fraction(track: Any, threshold: float) -> float:
    points = getattr(getattr(track, "box", None), "keypoints", [])
    if not points:
        return 0.0
    return sum(float(point.confidence) >= threshold for point in points) / max(len(points), 1)


def _close_tracker(tracker: Any) -> None:
    for track in list(getattr(tracker, "tracks", [])):
        close = getattr(track, "close", None)
        if close is not None:
            close()
    if hasattr(tracker, "tracks"):
        tracker.tracks.clear()


def evaluate_capture(clip: ClipSpec, capture: Any, bridge: Any,
                     runtime_config: dict[str, Any], sample_fps: float = FPS) -> ClipResult:
    """Evaluate one already-open capture; injectable capture/bridge enables fake tests."""
    result = ClipResult(
        clip=clip.clip_id or clip.path.name, path=clip.path.as_posix(),
        dataset=clip.dataset, split=clip.split, subject=clip.subject,
        subset=clip.subset, label=clip.label,
        onset_sec=_finite_or_none(clip.onset_sec), sample_fps=sample_fps)
    tracker = MultiPersonTracker(bridge, runtime_config)
    inference_times: list[float] = []
    person_counts: list[int] = []
    keypoint_coverages: list[float] = []
    alert_sec = math.inf
    alert_track_id: Optional[int] = None
    try:
        for frame, timestamp in sample_video_frames(capture, sample_fps):
            width, height = _frame_shape(frame)
            detections, inference_ms = bridge.infer(frame)
            persons = tracker.update(detections, timestamp, width, height)
            result.sampled_frames += 1
            inference_times.append(float(inference_ms))
            valid_persons = [person for person in persons if getattr(person.features, "valid", False)]
            person_counts.append(len(valid_persons))
            if valid_persons:
                result.pose_frames += 1
                keypoint_coverages.extend(
                    _visible_keypoint_fraction(person, float(runtime_config.get("keypoint_threshold", 0.25)))
                    for person in valid_persons)
            for person in persons:
                if getattr(person, "fall_event", False) and timestamp < alert_sec:
                    alert_sec = timestamp
                    alert_track_id = int(person.track_id)
        result.duration_sec = result.sampled_frames / sample_fps if sample_fps > 0 else 0.0
        # train_temporal_model.py falls back to 35% of the observed clip when
        # a GMDCSA CSV onset is absent; retain that exact convention.
        if result.label and (result.onset_sec is None or not math.isfinite(result.onset_sec)):
            result.onset_sec = result.duration_sec * 0.35
        result.alert_sec = _finite_or_none(alert_sec)
        result.alert_track_id = alert_track_id
        result.early_alert = bool(
            result.label and result.alert_sec is not None and result.onset_sec is not None and
            result.alert_sec < result.onset_sec - EARLY_MARGIN_SEC)
        result.detected = bool(result.alert_sec is not None and not result.early_alert)
        if result.label and result.detected and result.onset_sec is not None:
            result.latency_sec = result.alert_sec - result.onset_sec
        result.pose_coverage = result.pose_frames / max(result.sampled_frames, 1)
        result.mean_person_count = statistics.fmean(person_counts) if person_counts else 0.0
        result.mean_keypoint_coverage = statistics.fmean(keypoint_coverages) if keypoint_coverages else 0.0
        if inference_times:
            result.inference_ms_mean = statistics.fmean(inference_times)
            result.inference_ms_p50 = statistics.median(inference_times)
            result.inference_ms_max = max(inference_times)
    except Exception as error:  # Keep a failed clip in JSONL and continue the split.
        result.error = f"{type(error).__name__}: {error}"
    finally:
        _close_tracker(tracker)
        release = getattr(capture, "release", None)
        if release is not None:
            release()
    return result


def open_video(path: Path, use_gstreamer: bool = False) -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video evaluation")
    backend = cv2.CAP_GSTREAMER if use_gstreamer else cv2.CAP_ANY
    capture = cv2.VideoCapture(str(path), backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open video: {path}")
    return capture


def _result_prediction(result: ClipResult) -> int:
    # Early fall alerts are intentionally not successful predictions, matching
    # trigger_metrics() in train_temporal_model.py.
    if result.label:
        return int(result.detected and not result.early_alert)
    return int(result.alert_sec is not None)


def compute_metrics(results: Iterable[ClipResult]) -> dict[str, Any]:
    rows = list(results)
    truth = [int(row.label) for row in rows]
    prediction = [_result_prediction(row) for row in rows]
    tn = fp = fn = tp = 0
    for expected, actual in zip(truth, prediction):
        if expected == 1 and actual == 1:
            tp += 1
        elif expected == 1:
            fn += 1
        elif actual == 1:
            fp += 1
        else:
            tn += 1
    latencies = [row.latency_sec for row in rows if row.latency_sec is not None]
    pose_frames = sum(row.pose_frames for row in rows)
    sampled_frames = sum(row.sampled_frames for row in rows)
    inference = [row.inference_ms_mean for row in rows if row.inference_ms_mean is not None]
    metrics = {
        "n": len(rows), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "accuracy": _safe_ratio(tp + tn, len(rows)),
        "recall": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "precision": _safe_ratio(tp, tp + fp),
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "early_alerts": sum(row.early_alert for row in rows),
        "early_fall_alerts": sum(row.early_alert for row in rows),
        "mean_detection_latency_sec": statistics.fmean(latencies) if latencies else None,
        "median_detection_latency_sec": statistics.median(latencies) if latencies else None,
        "pose_frames": pose_frames,
        "sampled_frames": sampled_frames,
        "pose_coverage": _safe_ratio(pose_frames, sampled_frames),
        "mean_clip_pose_coverage": statistics.fmean(row.pose_coverage for row in rows) if rows else 0.0,
        "mean_inference_ms": statistics.fmean(inference) if inference else None,
        "misclassified": [row.clip for row, actual in zip(rows, prediction) if actual != row.label],
    }
    return metrics


def write_results(output_dir: Path, results: list[ClipResult], metadata: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "clips.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    summary = dict(metadata)
    summary["metrics"] = compute_metrics(results)
    summary["clips_jsonl"] = jsonl_path.name
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return jsonl_path, summary_path


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.engine is not None:
        config["engine_path"] = str(args.engine)
    if args.library is not None:
        config["trt_library"] = str(args.library)
    if not config.get("engine_path"):
        raise ValueError("--engine or config.engine_path is required")
    config.setdefault("trt_library", "/app/libjetson_fall_trt.so")
    config.setdefault("keypoint_threshold", 0.25)
    return config


def _load_clips(args: argparse.Namespace) -> tuple[list[ClipSpec], str]:
    sources = sum(value is not None for value in (args.gmdcsa_root, args.realbiom_manifest, args.manifest))
    if sources != 1:
        raise ValueError("choose exactly one of --gmdcsa-root, --realbiom-manifest, or --manifest")
    strict = not args.allow_incomplete
    if args.gmdcsa_root is not None:
        split = args.gmdcsa_split
        return load_gmdcsa_clips(args.gmdcsa_root, split, strict=strict), f"gmdcsa24:{split}"
    if args.realbiom_manifest is not None:
        subset = None if args.realbiom_subset == "all" else args.realbiom_subset
        return load_realbiomfall_clips(args.realbiom_manifest, subset, strict=strict), f"realbiomfall:{args.realbiom_subset}"
    return load_manifest_clips(args.manifest, strict=True), "manifest:custom"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson TensorRT offline fall evaluator")
    parser.add_argument("--config", type=Path, help="app config JSON; --engine/--library override it")
    parser.add_argument("--engine", type=Path, help="TensorRT engine built on this Jetson")
    parser.add_argument("--library", type=Path, help="libjetson_fall_trt.so (default /app/libjetson_fall_trt.so)")
    source = parser.add_argument_group("dataset source")
    source.add_argument("--gmdcsa-root", type=Path)
    source.add_argument("--gmdcsa-split", choices=sorted(EXPECTED_GMDCSA_CLIPS), default="subject4-clean")
    source.add_argument("--realbiom-manifest", type=Path)
    source.add_argument("--realbiom-subset", choices=("testing", "training", "validation", "all"), default="testing")
    source.add_argument("--manifest", type=Path, help="generic JSON list with path/label/onset_sec")
    parser.add_argument("--allow-incomplete", action="store_true", help="allow partial downloaded splits")
    parser.add_argument("--sample-fps", type=float, default=FPS)
    parser.add_argument("--gstreamer", action="store_true", help="open files with OpenCV GStreamer backend")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="list clips without loading TRT")
    args = parser.parse_args(argv)
    try:
        clips, protocol = _load_clips(args)
        if args.sample_fps <= 0:
            raise ValueError("--sample-fps must be > 0")
        print(json.dumps({"protocol": protocol, "clips": len(clips)}, indent=2))
        if args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "clips.jsonl").write_text(
                "".join(json.dumps(asdict(clip), sort_keys=True) + "\n" for clip in clips), encoding="utf-8")
            return 0
        runtime_config = _runtime_config(args)
        bridge = TrtBridge(runtime_config)
        results: list[ClipResult] = []
        started = time.monotonic()
        try:
            for index, clip in enumerate(clips, 1):
                print(f"[{index}/{len(clips)}] {clip.clip_id or clip.path.name}", file=sys.stderr)
                try:
                    capture = open_video(clip.path, use_gstreamer=args.gstreamer)
                    result = evaluate_capture(clip, capture, bridge, runtime_config, args.sample_fps)
                except Exception as error:
                    result = ClipResult(
                        clip=clip.clip_id or clip.path.name, path=clip.path.as_posix(),
                        dataset=clip.dataset, split=clip.split, subject=clip.subject,
                        subset=clip.subset, label=clip.label,
                        onset_sec=_finite_or_none(clip.onset_sec), sample_fps=args.sample_fps,
                        error=f"{type(error).__name__}: {error}")
                results.append(result)
        finally:
            bridge.close()
        metadata = {
            "protocol": protocol,
            "sample_fps": args.sample_fps,
            "early_margin_sec": EARLY_MARGIN_SEC,
            "temporal_window_frames": 48,
            "temporal_stride_frames": 3,
            "runtime": {"engine": runtime_config["engine_path"], "library": runtime_config["trt_library"]},
            "elapsed_sec": time.monotonic() - started,
            "errors": [row.clip for row in results if row.error],
        }
        _, summary_path = write_results(args.output_dir, results, metadata)
        print(json.dumps({"summary": str(summary_path), "metrics": compute_metrics(results)}, indent=2))
        return 0 if not metadata["errors"] else 1
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"evaluator error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
