#!/usr/bin/env python3
"""Convert one stream's MQTT NDJSON capture to the temporal-training trace format."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import TextIO


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def primary_person(payload: dict) -> dict | None:
    candidates = [
        person for person in payload.get("persons", [])
        if person.get("person_detected")
        and person.get("features", {}).get("valid")
        and isinstance(person.get("pose17"), list)
        and len(person["pose17"]) == 17
    ]
    return max(candidates, key=lambda person: float(person.get("person_score", 0.0)),
               default=None)


def convert(source: Path, stream_id: str) -> list[dict]:
    selected: list[dict] = []
    with open_text(source) as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("stream_id") != stream_id:
                continue
            if not isinstance(payload.get("timestamp"), int):
                raise ValueError(f"line {number}: timestamp must be epoch milliseconds")
            if not isinstance(payload.get("frame_id"), int):
                raise ValueError(f"line {number}: frame_id must be an integer")
            selected.append(payload)
    if not selected:
        raise ValueError(f"no messages for stream_id={stream_id!r}")
    if any(current["frame_id"] <= previous["frame_id"]
           for previous, current in zip(selected, selected[1:])):
        raise ValueError("frame_id must be strictly increasing")

    start_ms = selected[0]["timestamp"]
    rows: list[dict] = []
    for payload in selected:
        person = primary_person(payload)
        if person is None:
            rows.append({
                "timestamp": payload["timestamp"] - start_ms,
                "tracking": False,
                "fall_event": False,
                "pose17": [],
                "features": {},
                "inference_time_ms": payload.get("inference_time_ms", 0.0),
            })
            continue
        rows.append({
            "timestamp": payload["timestamp"] - start_ms,
            "tracking": True,
            "fall_event": bool(person.get("fall_event")),
            "track_id": person["track_id"],
            "pose17": person["pose17"],
            "features": person["features"],
            "inference_time_ms": payload.get("inference_time_ms", 0.0),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace {args.output}; pass --overwrite")
    rows = convert(args.input, args.stream_id)
    if len(rows) < args.min_frames:
        raise ValueError(f"only {len(rows)} frames, expected at least {args.min_frames}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    coverage = sum(bool(row["tracking"]) for row in rows) / len(rows)
    print(json.dumps({
        "input": str(args.input),
        "stream_id": args.stream_id,
        "output": str(args.output),
        "frames": len(rows),
        "pose_coverage": coverage,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
