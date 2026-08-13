#!/usr/bin/env python3
"""Collect MQTT JSON lines, validate every payload, and write reproducible metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "contracts"))
from validate_payload import validate


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, int((len(ordered) - 1) * p))
    return ordered[index]


def stats(values):
    return {"mean": statistics.mean(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99),
            "max": max(values)} if values else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    command = ["mosquitto_sub", "-h", args.host, "-t", args.topic, "-C", str(args.count),
               "-W", str(args.timeout)]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    records, errors = [], []
    assert process.stdout is not None
    for line in process.stdout:
        received_ms = int(time.time() * 1000)
        try:
            payload = json.loads(line)
            validate(payload)
            records.append({"received_ms": received_ms, "payload": payload})
        except Exception as exc:
            errors.append(str(exc))
    return_code = process.wait()
    elapsed = time.monotonic() - started
    Path(args.raw_out).write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in records))
    grouped = defaultdict(list)
    for record in records:
        grouped[record["payload"]["stream_id"]].append(record)
    streams = {}
    for stream_id, items in grouped.items():
        payloads = [x["payload"] for x in items]
        receive_span = max(1e-9, (items[-1]["received_ms"] - items[0]["received_ms"]) / 1000)
        source_span = max(1e-9, (payloads[-1]["timestamp"] - payloads[0]["timestamp"]) / 1000)
        ids = [x["frame_id"] for x in payloads]
        streams[stream_id] = {
            "messages": len(items), "schema_pass": len(items), "schema_fail": 0,
            "first_frame_id": ids[0], "last_frame_id": ids[-1],
            "mqtt_observed_fps": (len(items) - 1) / receive_span if len(items) > 1 else 0,
            "source_frame_id_rate": (ids[-1] - ids[0]) / source_span if len(items) > 1 else 0,
            "inference_ms": stats([float(x["inference_ms"]) for x in payloads]),
            "pipeline_ms": stats([float(x["pipeline_ms"]) for x in payloads]),
            "receive_minus_frame_timestamp_ms": stats([items[i]["received_ms"] - payloads[i]["timestamp"] for i in range(len(items))]),
            "source_backends": sorted({x.get("source_backend") for x in payloads}),
            "postprocess_backends": sorted({x.get("postprocess_backend") for x in payloads}),
            "fall_events": sum(bool(x["fall_event"]) for x in payloads),
            "visible_person_messages": sum(bool(x["person_detected"]) for x in payloads),
            "tracked_messages": sum(bool(x["tracking"]) for x in payloads),
            "states": sorted({x["state"] for x in payloads}),
        }
    summary = {"collector_command": command, "return_code": return_code,
               "elapsed_sec": elapsed, "captured_messages": len(records),
               "schema_pass": len(records), "schema_fail": len(errors),
               "schema_errors": errors, "streams": streams,
               "latency_scope": "pipeline_ms starts after appsink/cv2 returns a model-input frame; receive-minus-frame timestamp also includes MQTT and clock offset, not source encode time"}
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if return_code or len(records) != args.count or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
