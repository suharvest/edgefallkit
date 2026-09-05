#!/usr/bin/env python3
"""Collect MQTT JSON lines with legacy count and fixed-window modes."""
from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "contracts"))
from validate_payload import validate


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))] if ordered else None


def stats(values):
    return {"mean": statistics.mean(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99),
            "max": max(values)} if values else {}


def summarize(records, errors, return_code, *, window_seconds=None, expected_streams=None,
              min_fps=14.5, elapsed_sec=0.0, mode="count"):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["payload"]["stream_id"]].append(record)
    streams = {}
    for stream_id, items in grouped.items():
        payloads = [x["payload"] for x in items]
        receive_span = max(1e-9, (items[-1]["received_monotonic_ns"] - items[0]["received_monotonic_ns"]) / 1e9)
        source_span = max(1e-9, (payloads[-1]["timestamp"] - payloads[0]["timestamp"]) / 1000)
        ids = [x["frame_id"] for x in payloads]
        streams[stream_id] = {
            "messages": len(items), "schema_pass": len(items), "schema_fail": 0,
            "first_frame_id": ids[0], "last_frame_id": ids[-1],
            "mqtt_observed_fps": (len(items) - 1) / receive_span if len(items) > 1 else 0,
            "source_frame_id_rate": (ids[-1] - ids[0]) / source_span if len(items) > 1 else 0,
            "window_fps": len(items) / window_seconds if window_seconds else None,
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
    expected = sorted(expected_streams or [])
    missing = [stream for stream in expected if stream not in streams]
    reasons = []
    if return_code:
        reasons.append(f"mosquitto_sub exited with code {return_code}")
    if errors:
        reasons.append(f"{len(errors)} invalid payload(s)")
    if expected and missing:
        reasons.append(f"missing expected streams: {', '.join(missing)}")
    if expected and window_seconds:
        low = [stream for stream in expected if streams.get(stream, {}).get("window_fps", 0) < min_fps]
        if low:
            reasons.append(f"streams below {min_fps:g} FPS: {', '.join(low)}")
    if not records:
        reasons.append("zero valid messages")
    passed = not reasons
    summary = {"collector_command": [], "return_code": return_code, "elapsed_sec": elapsed_sec,
               "captured_messages": len(records), "schema_pass": len(records), "schema_fail": len(errors),
               "schema_errors": errors, "streams": streams,
               "latency_scope": "pipeline_ms starts after appsink/cv2 returns a model-input frame; receive-minus-frame timestamp also includes MQTT and clock offset, not source encode time",
               "scope": "fixed monotonic wall-clock window" if window_seconds else "legacy count",
               "window_seconds": window_seconds, "expected_streams": expected,
               "missing_streams": missing, "min_fps": min_fps if expected and window_seconds else None,
               "pass": passed, "pass_reasons": reasons}
    return summary


def _reader(stream, output):
    for line in iter(stream.readline, ""):
        output.put((line, time.monotonic_ns(), time.time_ns()))
    output.put(None)


def _drain_stderr(stream, output):
    for line in iter(stream.readline, ""):
        output.append(line.rstrip())


def _run(args):
    duration = args.duration_seconds is not None
    command = ["mosquitto_sub", "-h", args.host, "-p", str(args.port), "-t", args.topic]
    if not duration:
        command += ["-C", str(args.count), "-W", str(args.timeout)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert process.stdout is not None
    assert process.stderr is not None
    lines = queue.Queue()
    reader = threading.Thread(target=_reader, args=(process.stdout, lines), daemon=True)
    subscriber_stderr = deque(maxlen=200)
    stderr_reader = threading.Thread(target=_drain_stderr,
                                     args=(process.stderr, subscriber_stderr), daemon=True)
    reader.start(); stderr_reader.start()
    errors, records = [], []
    started = time.monotonic()
    warmup_deadline = started + args.warmup_seconds
    while time.monotonic() < warmup_deadline:
        try:
            item = lines.get(timeout=min(.1, max(0, warmup_deadline - time.monotonic())))
        except queue.Empty:
            continue
        if item is None:
            break
        item, _, _ = item
    window_start = time.monotonic()
    window_start_ns = time.monotonic_ns()
    deadline_ns = window_start_ns + int(args.duration_seconds * 1_000_000_000) if duration else None
    deadline = window_start + args.duration_seconds if duration else None
    while True:
        timeout = max(0.0, deadline - time.monotonic()) if deadline else max(0.1, args.timeout)
        try:
            item = lines.get(timeout=timeout)
        except queue.Empty:
            break
        if item is None:
            break
        item, received_ns, received_wall_ns = item
        if duration:
            if received_ns < int(window_start * 1_000_000_000):
                continue
            if received_ns >= deadline_ns:
                break
        try:
            payload = json.loads(item)
            validate(payload)
            records.append({"received_ms": received_wall_ns // 1_000_000,
                            "received_monotonic_ns": received_ns, "payload": payload})
            if not duration and len(records) >= args.count:
                break
        except Exception as exc:
            errors.append(str(exc))
    stopped_by_collector = process.poll() is None
    if stopped_by_collector:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
    else:
        process.wait()
    stderr_reader.join(timeout=1)
    elapsed = time.monotonic() - window_start
    window_seconds = args.duration_seconds if duration else None
    return_code = 0 if stopped_by_collector and duration else process.returncode
    summary = summarize(records, errors, return_code, window_seconds=window_seconds,
                        expected_streams=args.expected_streams, min_fps=args.min_fps,
                        elapsed_sec=elapsed, mode="duration" if duration else "count")
    summary["collector_command"] = command
    summary["subscriber_stderr_tail"] = list(subscriber_stderr)
    Path(args.raw_out).write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in records))
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pass"] or (not duration and len(records) != args.count):
        raise SystemExit(1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", required=True)
    count = ap.add_mutually_exclusive_group(required=True)
    count.add_argument("--count", type=int); count.add_argument("--duration-seconds", type=float)
    ap.add_argument("--warmup-seconds", type=float, default=0); ap.add_argument("--expected-streams")
    ap.add_argument("--min-fps", type=float, default=14.5); ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--raw-out", required=True); ap.add_argument("--summary-out", required=True)
    args = ap.parse_args(argv)
    if args.count is not None and args.count < 1: ap.error("--count must be >= 1")
    if args.count is not None and args.warmup_seconds:
        ap.error("--warmup-seconds requires --duration-seconds")
    if not 1 <= args.port <= 65535: ap.error("--port must be between 1 and 65535")
    if args.duration_seconds is not None and args.duration_seconds <= 0: ap.error("--duration-seconds must be > 0")
    if args.warmup_seconds < 0 or args.min_fps < 0: ap.error("warmup and min-fps must be >= 0")
    args.expected_streams = sorted({x.strip() for x in (args.expected_streams or "").split(",") if x.strip()})
    if args.min_fps != 14.5 and args.duration_seconds is None:
        ap.error("--min-fps is only valid with --duration-seconds")
    if args.expected_streams and args.duration_seconds is None:
        ap.error("--expected-streams requires --duration-seconds")
    return args


def main():
    _run(parse_args())


if __name__ == "__main__":
    main()
