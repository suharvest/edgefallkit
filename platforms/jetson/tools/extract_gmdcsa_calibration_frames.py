#!/usr/bin/env python3
"""Build a disjoint, pose-balanced TensorRT calibration set from GMDCSA.

Subjects 1-3 are supported. ADL videos contribute two interior frames; fall
videos contribute one pre-fall frame and five frames spanning the annotated
fall interval. Subject 4 is rejected so the frozen test split cannot leak into
engine calibration.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - reported by main on target
    cv2 = None


FALL_START = re.compile(r"fall(?:ing)?[^\[]*\[\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def fall_metadata(csv_path: Path) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 2 or not re.fullmatch(r"\d{2}\.mp4", row[0].strip()):
                continue
            duration = float(row[1])
            match = FALL_START.search(",".join(row[2:]))
            if match is None:
                raise RuntimeError(f"missing fall onset in {csv_path}: {row[0]}")
            rows[row[0].strip()] = (duration, float(match.group(1)))
    return rows


def sample_times(kind: str, duration: float, onset: float | None = None) -> list[float]:
    end = max(0.05, duration - 0.10)
    if kind == "ADL":
        return [duration * 0.33, duration * 0.67]
    if onset is None:
        raise ValueError("fall onset is required")
    start = min(max(0.05, onset), end)
    span = max(0.0, end - start)
    return [max(0.05, start - 0.25)] + [
        start + fraction * span for fraction in (0.15, 0.35, 0.55, 0.75, 0.95)
    ]


def video_duration(source: Path) -> float:
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0.0 or frames <= 0.0:
            raise RuntimeError(f"cannot read video metadata: {source}")
        return frames / fps
    finally:
        capture.release()


def extract_frame(source: Path, timestamp: float, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode {source}")
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None or not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f"cannot extract calibration frame: {source} at {timestamp:.3f}s")
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", default="1,2,3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    subjects = sorted({int(value) for value in args.subjects.split(",") if value.strip()})
    if cv2 is None:
        raise SystemExit("python3-opencv is required")
    if not subjects or any(subject not in (1, 2, 3) for subject in subjects):
        raise SystemExit("calibration subjects must be a non-empty subset of 1,2,3")
    frames_dir = args.output / "frames"
    manifest = args.output / "calibration.txt"
    if not args.resume and (manifest.exists() or (frames_dir.exists() and any(frames_dir.iterdir()))):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    destinations: list[Path] = []
    for subject in subjects:
        subject_dir = args.dataset / f"subject-{subject}"
        fall_rows = fall_metadata(subject_dir / "Fall.csv")
        for kind in ("ADL", "Fall"):
            for source in sorted((subject_dir / kind).glob("*.mp4")):
                if kind == "Fall":
                    _, onset = fall_rows[source.name]
                    duration = video_duration(source)
                else:
                    duration, onset = video_duration(source), None
                for index, timestamp in enumerate(sample_times(kind, duration, onset)):
                    destination = frames_dir / (
                        f"subject-{subject}_{kind}_{source.stem}_p{index}_{timestamp:.3f}.jpg"
                    )
                    if not (args.resume and destination.is_file() and destination.stat().st_size > 0):
                        extract_frame(source, timestamp, destination)
                    destinations.append(destination)
                    print(f"{source} t={timestamp:.3f} -> {destination}", flush=True)

    manifest.write_text(
        "".join(f"frames/{path.name}\n" for path in destinations), encoding="utf-8"
    )
    print(f"manifest={manifest} images={len(destinations)} subjects={subjects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
