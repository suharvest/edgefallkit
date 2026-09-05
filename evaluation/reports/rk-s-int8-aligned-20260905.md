# RK3576/RK3588 aligned S INT8 performance — 2026-09-05

## Contract and identities

All production-path rows use one frozen 640×640 H.264 source at 15 FPS, a
30-second warm-up and a fixed 120-second MQTT observation window. A route
passes at 14.5 FPS or higher. The highest passing route count is repeated
three times; the next route count stops after its first failure.

| Artifact | SHA256 |
|---|---|
| Input fixture | `e3d530d3e9e9a1b2d2172a44961f3de747fbfd00059c62722cafa5f66dad1e29` |
| Runtime image | `43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8` |
| Test code bundle | `bf53c012635e0f39f001e33926ff2cba0041b9979a22211dc119714f914b2d34` |
| RK3576 S INT8 model | `1fe0067c615f92509d62b30200f04159b5bb50963a1b798323c50aad6947ca97` |
| RK3588 S INT8 model | `3f73273cf1cd20bd145a7ecae24b4511b668b3683044d6b6973bd80abf994a85` |

`inference_ms` is the RKNNLite call only. `pipeline_ms` starts after source
read/preprocessing and includes RKNN inference, pose decode/NMS, tracking,
temporal/fall state and payload construction; MQTT transmission is outside the
interval. Output FPS is valid MQTT messages divided by the fixed wall-clock
window.

## Production Python path

The measured path is MPP NV12 decode followed by mapped-buffer OpenCV color
conversion/letterbox, a fair four-context pool where applicable, C++ pose
postprocess, per-stream tracking/temporal state and MQTT output.

| Device/routes | Repetition | FPS per route | Inference mean | Inference P95 | Pipeline P95 | Result |
|---|---:|---:|---:|---:|---:|---|
| RK3576 / 1 | 1 | 14.9917 | 43.315 ms | 49.783 ms | 50.137 ms | pass |
| RK3576 / 1 | 2 | 14.8333 | 43.489 ms | 50.564 ms | 50.876 ms | pass |
| RK3576 / 1 | 3 | 15.0083 | 43.809 ms | 50.401 ms | 50.720 ms | pass |
| RK3576 / 2 | 1 | 12.8333 / 12.8083 | 51.448 / 51.450 ms | 59.754 / 58.730 ms | 60.227 / 59.358 ms | fail |
| RK3588 / 5 | 1–3 | 14.9667–15.0083 | 42.63–43.35 ms | not frozen | 50.28–51.56 ms | pass |
| RK3588 / 6 | 1 | 14.4333–14.4917 | 44.39–44.55 ms | not frozen | 52.50–53.12 ms | fail |

The verified 15 FPS capacities for this model/path are one route on RK3576 and
five routes on RK3588. These rows do not replace the separate FP16/M model
microbenchmarks or accuracy results.

## RK3588 Python-control/native-hot-path experiment

The experimental path keeps capture and scheduling in Python but passes the
MPP DMA-BUF fd to a C++ stage. A process-wide serialized RGA conversion writes
RGB directly to RKNN-allocated input memory registered by `rknn_set_io_mem`.
Four RKNN contexts serve five sources. Native outputs are synchronized and
read to form checksums, but are not converted back into pose tensors.

| Repetition | FPS per route | CPU (one core = 100%) | Max RSS | RGA mean / P95 | RKNN mean / P95 | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 15.0000 each | 44.547% | 157168 KiB | 0.485 / 0.791 ms | 39.304 / 44.088 ms | pass |
| 2 | 15.0000 each | 43.747% | 157184 KiB | 0.482 / 0.787 ms | 39.267 / 43.900 ms | pass |
| 3 | 14.9917–15.0000 | 44.547% | 157116 KiB | 0.490 / 0.811 ms | 39.308 / 44.098 ms | pass |

One aligned 10-second warm-up plus 30-second Python/OpenCV hot-path sample used
the same source count, context count and scheduling code but omitted the same
business/MQTT stages as the native experiment. It measured 14.9667–15.0000 FPS
per route, 144.575% CPU, 495488 KiB max RSS and RKNNLite 43.233 ms mean /
51.147 ms P95. Relative to that sample, the native hot path used about 69%
less CPU and 68% less RSS, while RKNN call mean was about 9% lower. Source rate
caps both paths at 15 FPS, so this test does not claim additional route
capacity.

The hybrid result is **experimental performance-only**. It does not execute
pose decoding, tracking, the temporal model or MQTT, and therefore does not
establish production output, end-to-end latency or precision equivalence.

Hybrid probe identities:

- C++ source: `389105baef45982e2913b3c1841e0dd4f74bff6ae4006b7c1ea85a819084cf5c`
- benchmark script: `b92d176b7c1301a26ec3fcfd39d13a312b882c459a00a48454156b2c43a61f18`
- board-built shared object: `6d7b0c12b796ef70bf746ce0af68292f52e26e45d5c545ae3fbe4348cf6b3bc5`

Raw device logs were frozen under
`edgefallkit-work/rk-aligned-20260903/formal-logs/`; this report contains the
reviewed values intended for version control.
