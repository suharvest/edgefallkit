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

`Runtime image` above is the historical image identity used for these
performance rows. The subsequently published RC7 runtime is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc7`,
RepoDigest
`sha256:8c79172138a0f510e26bd0f219f82b6a57ab98ff30f6828d96786e5131dfeae5`.
RC7 adds the native DMA-BUF/RGA/RKNN bridge; it does not change the identity
of the measured rows below.

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

## RK3576 native production-chain qualification

An optional host-native path removes the RK3576 Python preprocessing bottleneck:
MPP retains each decoded NV12 frame as DMA-BUF, RGA writes RGB into the
RKNN-allocated input buffer, and two RKNN contexts are pinned to core masks 1
and 2. Pose decode/NMS, tracking, the temporal state machine and MQTT schema are
the same application stages as the production Python path. MQTT transmission
uses one bounded publisher worker and connection per stream, so JSON encoding
and socket writes do not serialize all inference workers through one shared
background worker.

| Repetition | FPS per route | Inference mean | Inference P95 | Pipeline P95 | Schema | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 15.0000 / 14.9917 | 42.450 / 42.408 ms | 46.460 / 46.304 ms | 46.981 / 46.768 ms | 1800 / 1799 | pass |
| 2 | 15.0000 / 15.0167 | 42.393 / 42.357 ms | 46.137 / 46.427 ms | 46.712 / 46.921 ms | 1800 / 1802 | pass |
| 3 | 14.8500 / 14.8583 | 42.376 / 42.287 ms | 46.351 / 46.212 ms | 46.859 / 46.769 ms | 1782 / 1783 | pass |

Those initial two-route runs established the native inference path. The final
capacity test also removed a shared MQTT publisher bottleneck and tested three
and four sources:

| Routes | Repetition | FPS per route | Inference P95 | Pipeline P95 | Schema | Result |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | 14.9917 / 15.0083 / 15.0000 | 43.774–43.990 ms | 44.275–44.504 ms | 5400 / 5400 | pass |
| 3 | 2 | 14.9833 / 14.9167 / 15.1083 | 44.005–44.060 ms | 44.566–44.637 ms | 5401 / 5401 | pass |
| 3 | 3 | 14.9667 / 15.0000 / 14.9667 | 44.030–44.201 ms | 44.490–44.728 ms | 5392 / 5392 | pass |
| 4 | boundary | 10.1667–10.3333 | 44.959–45.446 ms | 46.074–46.518 ms | 1233 / 1233 | fail |

The verified host-native boundary is therefore **three 15 FPS sources**. The
four-route source-frame rates were only 11.787–11.979 FPS, so its failure is an
inference-capacity boundary rather than an MQTT observation artifact.

The first three-route diagnostics used one asynchronous publisher worker for
all streams. One window appeared to pass, while the next delivered only
11.62–11.71 FPS per route even though source-frame IDs advanced at
14.96–15.00 FPS. Moving only `TCP_NODELAY` did not correct the failure. The
old RK3588 five-route run did not exhibit the bottleneck: it delivered
14.9667–15.0083 FPS per route in three formal windows. Its implementation did
JSON serialization concurrently in each inference thread and locked only the
socket send, a code-path difference consistent with the observed behavior.
The evidence does not isolate that difference from CPU, synchronous backpressure
or other runtime differences as the sole cause. Per-stream publisher workers
restored the RK3576 output rate in all three formal repetitions. The failed
shared-publisher diagnostics are not capacity evidence.

The qualification has four limits:

- The frozen source and model input are both 640×640. The native adapter
  accepts only an exact 640×640 source until source-aware RGA resize and
  letterbox padding are implemented.
- All 10,766 messages from the initial two-route repetitions and all 16,193
  messages from the final three-route repetitions passed the output schema,
  but this fixture produced no visible-person messages at the configured
  threshold. The runs establish throughput and interface continuity, not
  native/legacy pose or fall-state equivalence.
- `pipeline_ms` ends before MQTT transmission. The separately observed
  receive-minus-frame timestamp includes broker/network delay and clock offset
  and is not used as an inference-latency value.
- The RC2 container available when these results were measured did not
  negotiate `memory:DMABuf`, so the formal runs used the board host runtime.
  RC7 was subsequently published with this path and passed a shorter container
  smoke; the README production-capacity table remains tied to the formal runs.

Frozen native qualification identities:

| Artifact | SHA256 |
|---|---|
| Deployed `app.py` (three-route test) | `d24a3223e41c459df1256931d7e8c29b681328d4ae595085b095fbff4e50aa3d` |
| Deployed `rknn_pose.py` (three-route test) | `ddcaa4eaa30f240532f9dc7e6bae315937c8fc0d35caefda861f8c7960293f05` |
| Deployed `video_source.py` (three-route test) | `8b389ae47b3f2a72497a49e363bdbf54187bd9c6df585efc8e3683e3ba51041c` |
| Deployed `native_rknn.py` (three-route test) | `db7ef956d6c115a1f1ef9494a7d9b6dbfe4f1d842b96384446f09e82de40d99e` |

Raw MQTT evidence is frozen under
`edgefallkit-work/rk-aligned-20260903/rtsp-results/cat-native-s-int8-final-formal/`
for the initial two-route run and
`edgefallkit-work/rk-aligned-20260903/rtsp-results/cat-native-s-int8-n3-n4-20260905T1955/`
for the three/four-route boundary.

## RK3588 shared-native production-chain qualification

The RK3576 host-native implementation was reused without a platform-specific
application fork. Four native RKNN contexts served five sources; only the
RK3588 model identity and runtime platform differed. The path executed MPP
DMA-BUF capture, serialized RGA conversion into RKNN-owned input memory, native
RKNN inference, C++ pose decode/NMS, tracking, the temporal state machine and
one bounded MQTT publisher per stream.

| Repetition | FPS per route | Inference P95 | Pipeline P95 | Schema | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 14.9917–15.0000 | 44.905–45.421 ms | 45.727–46.181 ms | 8997 / 8997 | pass |
| 2 | 15.0000 each | 44.820–45.143 ms | 45.597–45.969 ms | 9000 / 9000 | pass |
| 3 | 14.9917–15.0000 | 44.785–45.302 ms | 45.574–46.094 ms | 8998 / 8998 | pass |

All 26,995 messages contained visible-person and tracked-person output; source
frame IDs were continuous within every route/window. The evidence therefore
covers the pose decoder and business chain, unlike the earlier checksum-only
hybrid probe below. A snapshot after 784 seconds measured 68.3% process CPU,
168556 KiB RSS, 53 threads and 176 open descriptors. Kernel-side evidence
showed five `/dev/mpp_service` descriptors, one `/dev/rga` descriptor and five
pairs of MPP parser/HAL threads. The process exited after SIGTERM and the four
pre-existing RTSP containers were restored. The snapshot and artifact hashes
are frozen in
[`evidence/rk3588-native-n5-resource-20260905.txt`](evidence/rk3588-native-n5-resource-20260905.txt).

This qualification used the board host runtime, so it remains host-runtime
capacity evidence. RC7 subsequently packaged the native bridge and passed a
separate five-route container smoke; that packaging smoke does not replace the
three formal 120-second repetitions. Raw MQTT evidence is frozen under
`edgefallkit-work/rk-aligned-20260903/rtsp-results/radxa-shared-native-s-int8-n5-20260905/`.

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
