#!/usr/bin/env python3
"""Summarize benchmark_multicontext.sh logs using only the Python stdlib."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


THROUGHPUT_RE = re.compile(r"Throughput: ([0-9.]+) qps")
LATENCY_RE = re.compile(
    r"Latency: .*?mean = ([0-9.]+) ms,.*?percentile\(95%\) = ([0-9.]+) ms"
)
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU \[([^]]+)]")
GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
POWER_RE = re.compile(r"VDD_IN (\d+)mW/")
GPU_TEMP_RE = re.compile(r"gpu@([0-9.]+)C")


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def summarize_run(run_dir: Path) -> dict[str, object]:
    log = (run_dir / "trtexec.log").read_text(errors="replace")
    throughput_match = THROUGHPUT_RE.search(log)
    latency_match = LATENCY_RE.search(log)
    if throughput_match is None or latency_match is None:
        raise ValueError(f"missing TensorRT metrics in {run_dir}")

    ram: list[float] = []
    cpu: list[float] = []
    gpu: list[float] = []
    power: list[float] = []
    temperature: list[float] = []
    for line in (run_dir / "tegrastats.log").read_text(errors="replace").splitlines():
        if match := RAM_RE.search(line):
            ram.append(float(match.group(1)))
        if match := CPU_RE.search(line):
            cores = [float(value) for value in re.findall(r"(\d+)%@", match.group(1))]
            if cores:
                cpu.append(sum(cores) / len(cores))
        if match := GPU_RE.search(line):
            gpu.append(float(match.group(1)))
        if match := POWER_RE.search(line):
            power.append(float(match.group(1)))
        if match := GPU_TEMP_RE.search(line):
            temperature.append(float(match.group(1)))

    rss_kib: list[float] = []
    process_cpu: list[float] = []
    process_log = run_dir / "process.log"
    if process_log.exists():
        for line in process_log.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 4:
                rss_kib.append(float(fields[1]))
                process_cpu.append(float(fields[2]))

    contexts = int(run_dir.name.split("-")[-1])
    throughput = float(throughput_match.group(1))
    return {
        "contexts": contexts,
        "aggregate_fps": throughput,
        "per_context_fps": round(throughput / contexts, 3),
        "latency_mean_ms": float(latency_match.group(1)),
        "latency_p95_ms": float(latency_match.group(2)),
        "process_rss_mib_mean": None if not rss_kib else round(statistics.fmean(rss_kib) / 1024, 3),
        "process_rss_mib_max": None if not rss_kib else round(max(rss_kib) / 1024, 3),
        "process_cpu_percent_mean": mean(process_cpu),
        "system_ram_mib_mean": mean(ram),
        "gpu_util_percent_mean": mean(gpu),
        "board_power_w_mean": None if not power else round(statistics.fmean(power) / 1000, 3),
        "board_power_w_max": None if not power else round(max(power) / 1000, 3),
        "gpu_temperature_c_mean": mean(temperature),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "result_dir": str(args.result_dir),
        "runs": [summarize_run(path) for path in sorted(
            args.result_dir.glob("contexts-*"), key=lambda item: int(item.name.split("-")[-1])
        )],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
