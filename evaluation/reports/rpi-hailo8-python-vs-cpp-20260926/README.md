# Hailo-8 on Raspberry Pi 5: Python vs C++ multi-stream (2026-09-26)

Question: can a Python HailoRT pipeline replace the C++ `fall-hailo` path at 16 streams @ 15 fps?
Answer: no. The Python pipeline reaches 12 streams; 16 streams fail. The C++ path passes 16 streams with no margin.

## Conditions

| Item | Value |
|---|---|
| Device | Raspberry Pi 5 8 GB + Hailo-8, HailoRT 4.21.0 (driver and cp311 wheel) |
| Model | `yolov8s_pose.hef`, Hailo Model Zoo v2.15 hailo8, sha256 `e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d` |
| Input | `fall-07-640x640-15fps.mp4` re-encoded to 640×640 H.264 Constrained Baseline, 15 fps, 1.2 Mbps, GOP 30; N RTSP streams from a MediaMTX host on the LAN |
| Decode | CPU (Pi 5 has no H.264 hardware decode) |
| MQTT | on, local mosquitto |
| Window | 30 s warm-up + 120 s measurement per level; round 1 repeated twice |
| Pass line | every stream ≥ 14.5 fps |
| C++ path | image `fall-detection-rpi-hailo:0.1.0-rc3@sha256:994b363dc1aa68d3ada0ca3590bd810ab26a2240918bcffe426104761a2f772a`; auto policy selected legacy per-stream `hailonet`, batch 1 |

## Round 1 — minimum per-stream fps (process CPU, 100% = one core)

| Streams | C++ fall-hailo | Python single process (batch 8) | Python process shards via hailort service |
|---|---|---|---|
| 8 | 15.00 / 15.00 (102%) | 14.94 / 14.99 (170%) | 14.90 / 14.93 |
| 12 | 15.00 / 14.98 (162–165%) | 14.82 / 14.61 (231–233%) | 9.86 / 9.96 |
| 16 | 14.75 / 14.47 (239–240%) | 9.40 / 9.75 (247%) | 5.95 / 5.95 |

## Round 2 — why Python is slower

- Native ceiling of the HEF on this device: 394 FPS (`hailortcli run` / `benchmark`). Neither path is NPU-bound; the `hailortcli monitor` utilisation column is scheduler occupancy, not NPU load.
- Main cause of the gap: Python used batch 8 on a single-context HEF; C++ used batch 1. With batch 1:
  - 12 streams: full rate, pipeline p95 87 ms (was 193 ms), CPU per FPS 1.24× C++ (was 1.44×).
  - 16 streams: 191–195 aggregate FPS (was 174), min stream 11.7–11.9 fps — still fails.
- Buffer handling (aligned pool, bindings set once, 2/4/8 jobs in flight, decode off the callback) changed results by ±5%.
- HailoRT Python bindings (4.21 and 4.24 on the `hailo8` branch) expose no DMA mapping API, and `infer_model_api.cpp` allocates a DMA output buffer per `run_async` and copies it back for multi-output models.
- Remaining gap at 16 streams: host CPU (~330/400% machine-wide). Python spends ~28% more CPU per frame (numpy decode, tracker, JSON, 16 capture threads, binding overhead).

## Files

- `round1/` — `bench_b.py` (Python pipeline), `monitor.py`, `run_one.sh`, `matrix.sh`, `analyze.py`, `hb-results.tgz` (per-level logs, JSON, CPU, hailortcli monitor, py-spy).
- `round2/` — `synth.py` (synthetic-input variants), `bench_c.py` (bench_b with `--sched/--inflight/--aligned`), `run_synth.sh`, `run_full.sh`, `hb2-results.tgz`.
- `SHA256SUMS`.

LAN addresses in the archived logs and scripts are replaced with 192.0.2.0/24 documentation addresses.
