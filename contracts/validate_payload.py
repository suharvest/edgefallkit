#!/usr/bin/env python3
"""Dependency-free MQTT contract-v1 fixture validator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOP = {
    "timestamp", "frame_id", "inference_time_ms", "stream_id",
    "fall_detected", "fall_event", "event_id", "global_event_id",
    "event_id_scope", "state", "person_detected", "person_count",
    "fallen_count", "tracking", "features", "keypoints", "pose17", "persons",
}
PERSON = {
    "track_id", "state", "fall_detected", "fall_event", "event_id",
    "person_detected", "person_score", "tracking", "missed_frames", "bbox",
    "features", "keypoints", "pose17",
}
FEATURES = {
    "valid", "hip_drop_speed", "hip_drop_distance", "torso_angle_deg",
    "bbox_aspect_ratio",
}
STATES = {"normal", "suspected", "fallen", "recovering"}


def _require(obj: dict, keys: set[str], where: str) -> None:
    missing = keys - obj.keys()
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")


def _pose(value: object, where: str) -> None:
    if not isinstance(value, list) or len(value) not in (0, 17):
        raise ValueError(f"{where}: pose17 must be empty or contain 17 points")
    if any(not isinstance(p, list) or len(p) != 3 for p in value):
        raise ValueError(f"{where}: each pose17 point must have 3 numbers")


def validate(payload: dict) -> None:
    _require(payload, TOP, "payload")
    if not isinstance(payload["timestamp"], int) or payload["timestamp"] < 0:
        raise ValueError("payload.timestamp must be Unix epoch milliseconds")
    if payload["event_id_scope"] != "stream_global_event_id":
        raise ValueError("payload.event_id_scope")
    if payload["event_id"] != payload["global_event_id"]:
        raise ValueError("top event_id and global_event_id must match")
    if payload["state"] not in STATES:
        raise ValueError("payload.state")
    if not isinstance(payload["keypoints"], list):
        raise ValueError("payload.keypoints must be an array")
    _require(payload["features"], FEATURES, "payload.features")
    _pose(payload["pose17"], "payload")
    people = payload["persons"]
    if not isinstance(people, list):
        raise ValueError("payload.persons must be an array")
    visible = 0
    fallen = 0
    for index, person in enumerate(people):
        where = f"persons[{index}]"
        _require(person, PERSON, where)
        _require(person["features"], FEATURES, f"{where}.features")
        if person["state"] not in STATES:
            raise ValueError(f"{where}.state")
        if not isinstance(person["keypoints"], list):
            raise ValueError(f"{where}.keypoints must be an array")
        _pose(person["pose17"], where)
        visible += bool(person["person_detected"])
        fallen += person["state"] in ("fallen", "recovering")
    if payload["person_count"] != visible:
        raise ValueError("person_count does not equal visible persons")
    if payload["fallen_count"] != fallen:
        raise ValueError("fallen_count does not equal retained fallen/recovering persons")
    if payload["person_detected"] != (visible > 0):
        raise ValueError("person_detected mismatch")
    if payload["tracking"] != bool(people):
        raise ValueError("tracking mismatch")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PAYLOAD.json")
    validate(json.loads(Path(sys.argv[1]).read_text()))
    print("MQTT contract v1 passed")


if __name__ == "__main__":
    main()
