# Raspberry Pi 5 + Hailo-8 fall detection

Native C++ runtime for the shared fall-detection algorithm. Frames are decoded
by GStreamer and resized to RGB 640x640. Single-context models use the native
`hailonet` plugin; multi-context models can use the shared direct-HailoRT batch
backend. The process decodes the nine quantized YOLOv8-Pose output tensors,
tracks every person independently, runs the same 48-frame temporal MLP and
state machine as the other platforms, then publishes reCamera-compatible MQTT.
There is no Torch, Ultralytics, ONNX Runtime, or Python in the deployed hot path.

## Tested target and ABI lock

The verified target is Fleet `harvest-pi`: Raspberry Pi 5, Debian 13, kernel
6.12.75, 16 KB pages, Hailo-8 (26 TOPS), and HailoRT driver/firmware/user ABI
4.21.0. `force_desc_page_size=4096` is active. The compose file bind-mounts
host `libhailort.so.4.21.0` and `libgsthailo.so`; do not use it with another
driver minor version without changing all three together.

The pose model is official Hailo Model Zoo v2.15 `yolov8s_pose.hef` for
Hailo-8, UINT8/UINT16 outputs, COCO-17, 640x640. Upstream reports COCO keypoint
mAP 59.2 full precision and 56.36 after hardware quantization. These numbers
are pose accuracy, **not fall-detection accuracy**. The platform-specific
frozen fall test is reported below.

```bash
STREAMS='lobby|rtsp://host/a;ward|rtsp://host/b' \
  ./deploy.sh --accept-upstream-license
```

`deploy.sh` is the one-command model preparation and deployment path. The
license acknowledgement is mandatory. It accepts `--hef PATH` for a local
model, `--offline` to prohibit model/image network access, and `--dry-run` to
show the plan without writing or starting containers. A verified cache is
reused. New files are downloaded/copied to a same-directory temporary file,
checked against the fixed SHA256, and atomically renamed; any model, pull, or
Compose validation failure occurs before `docker compose up`.

The default published runtime is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc3`.
Override it with `FALL_HAILO_IMAGE=...` for a pinned mirror or local build. To
rebuild instead of pull:

```bash
docker compose --profile build build builder
docker compose --profile build run --rm builder
docker compose build fall-detection
docker compose config
```

For a self-hosted broker, add `--profile local-broker`. Otherwise set
`MQTT_HOST`, port, credentials and topic. `person_count` counts currently
visible tracks; `fallen_count` includes briefly retained missed tracks.
`event_id/global_event_id` is stream-global and `persons[].event_id` is
per-track. Coordinates in `bbox` and `pose17` are normalized to the 640x640
letterboxed inference space.

`pipeline_ms`/`pipeline_time_ms` depend on the selected backend. Legacy
`hailonet` reports `latency_metric=pre_hailonet_to_hailonet_src`; its
`pipeline_full_metric` is `pre_hailonet_to_post_tracker`. The shared backend
reports `latency_metric=appsink_enqueue_to_hailort_completion`; its full metric
is `appsink_enqueue_to_post_tracker`. Neither includes RTSP receive/decode or
resize before the named origin, and neither `pipeline_ms` value includes C++
postprocess, tracking, or MQTT. The runtime does not expose a separate
accelerator-call duration, so `inference_time_ms` remains `0` with
`inference_time_metric=unavailable`; it must not be mislabeled with either
larger pipeline interval.
`hailortcli benchmark` for pure NPU throughput.

### Shared batch selection

At startup `HAILO_BATCH_MODE=auto` selects the shared batch backend only when
the HEF exposes exactly one network group marked multi-context. It selects
batch 1 for 1--3 streams, batch 4 for 4 streams, and batch 8 for 5 or more.
`off` keeps the legacy per-stream backend; `1`, `4`, and `8` are explicit
shared-backend diagnostic overrides. `HAILO_BATCH_WAIT_MS` (0--1000, default
20) bounds the time used to form a partial batch. The selected mode is printed
as a machine-readable `HAILO_BATCH` line. Single-context YOLOv8s remains on
the legacy path by default; multi-context YOLOv8m is the intended consumer.

The shared backend collects the latest RGB frame from each stream and performs
one direct HailoRT inference for the group. It configures the VDevice without
the round-robin scheduler, explicitly activates the model, and pads partial
submissions to the configured batch size. Page-aligned input/output slots,
DMA mappings, and bindings are allocated once and reused. Shutdown stops frame
collection, joins the inference worker after its current checked job, and then
shuts down the configured model; this ordering avoids a HailoRT 4.21 lock
conflict between vector `run_async` and `shutdown`.

## Benchmark

Set `BENCHMARK_SECONDS=30` and optionally `BENCHMARK_WARMUP_SECONDS=5`; use
`test://ball` to isolate inference from RTSP. Warmup frames are processed and
published, but are excluded from the reported `frames`, `fps`, and mean latency:

```bash
STREAMS='one|test://ball' BENCHMARK_WARMUP_SECONDS=5 BENCHMARK_SECONDS=30 docker compose run --rm fall-detection
STREAMS='one|test://ball;two|test://ball' BENCHMARK_WARMUP_SECONDS=5 BENCHMARK_SECONDS=30 docker compose run --rm fall-detection
```

The inference queue defaults to `INFERENCE_QUEUE_DEPTH=1` with downstream
leaking and all other queue limits disabled. `queue=1` reduces stale-frame
latency under load; it does not claim to increase throughput. For RTSP,
`RTSP_DROP_ON_LATENCY` defaults to `false` (accepted values: `true`/`false` or
`1`/`0`). Set it to `true` when prioritizing lower latency over retaining every
frame; frames exceeding the configured source latency can then be discarded.

Before benchmarking, ensure no other process owns `/dev/hailo0`. HailoRT 4.21
direct-mode contexts are exclusive. In the authorized 2026-08-13 maintenance
window, HailoRT measured 393.3 FPS / 6.87 ms HW latency; the application ran
29.77 FPS single-stream and 29.86 + 29.66 FPS dual-stream using `test://ball`.
The pre-hailonet-to-source probe averaged about 7.81 ms. Dual-stream CPU/RSS
snapshots were 38.2% / 118,480 KiB at 60.05 C. `mcp_face_rec` was restarted
afterward and verified healthy. These synthetic-input numbers are throughput
evidence, not RTSP end-to-end latency or fall accuracy.

The runtime image is a multi-stage build and does not contain compiler,
headers, source, HEF, or HailoRT. Model and ABI-locked host libraries are
mounted read-only; the `builder` is isolated behind the `build` profile. The
Pi-native image build was verified as `fall-detection-rpi-hailo:4.21`, digest
`sha256:7e7d81503ed94160d8a9f64d5caa388a806a51a0e801889c40aeb85f591c86e2`,
with 143,442,009 bytes content size (`docker images` reports about 533 MB disk usage
including unpacked/shared layers).

The MQTT-enabled RC image was pushed from Fleet `spark` and pulled back on
`harvest-pi` as
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc3`.
The registry RepoDigest is
`sha256:994b363dc1aa68d3ada0ca3590bd810ab26a2240918bcffe426104761a2f772a`;
registry inspection and device-side pull report Linux/ARM64, revision
`8fbe716ba26261f2e6973185fa42b00c1a45aabe`, and entrypoint
`/usr/local/bin/fall-hailo`. The production binary SHA256 is
`d924c38ea07a9011b249f1630d91c325bd420f9d0c2175ead355a9081da676d6`.
Single-stream synthetic-input smoke tests exercised the NPU and MQTT with both
the default YOLOv8s HEF and official YOLOv8m HEF; the existing `mcp_face_rec`
container was restored afterward and verified healthy.

### Mac RTSP end-to-end control test

On 2026-08-13 a Mac-hosted MediaMTX stream was consumed from Fleet
`harvest-pi` over Tailscale. The controlled source was H.264 Constrained
Baseline, 640x640 at 15 FPS, approximately 1.2 Mbps CBR, GOP 30, no B-frames.
The test exercised RTSP TCP decode, resize, `hailonet`, the nine-output pose
decoder, tracker/temporal update, MQTT publishing, and the shared payload
validator.

| Source / streams | App FPS including startup | Steady MQTT FPS | Probe mean / P95 | CPU / RSS max | Temperature |
|---|---:|---:|---:|---:|---:|
| 640x640 control, one | 14.32 | 15.03 | 7.77 / 8.36 ms | 11.5% final / 127,760 KiB | 58.75 C mean, 60.4 C max |
| 640x640 control, two | 14.33 + 14.30 | 15.07 + 15.04 | 9.92 / 15.37 and 9.76 / 15.43 ms | 22.9% final / 182,080 KiB | 59.51 C mean, 60.9 C max |
| 1280x720 comparison, one | 13.41 | 14.17 | 7.88 / 8.50 ms | 24.8% final / 133,184 KiB | 58.52 C mean, 60.4 C max |
| Spark LAN GMDCSA S4 Fall/01 loop, one | 14.72 | 15.02 | 7.48 / 8.53 ms | 12.5% final / 130,784 KiB | 58.87 C mean, 62.0 C max |

All 1,718 captured MQTT messages passed the common contract validator and the
two control streams retained independent `stream_id` and frame counters.
MediaMTX reported slow-reader/discard activity for the 720p comparison, so its
lower frame rate is network-path evidence, not lower Hailo inference capacity.
The control clip produced no person detections and validates the empty-detection
branch. A separate authorized run consumed Spark's `fall-person` loop for 60.05
seconds: all 884 messages passed the contract, 615 carried a visible valid
person with 17 keypoints, 813 retained a track, 462 person messages were
temporal-positive, and the payload covered `normal`, `suspected`, `fallen`, and
`recovering`. Five fall-event edges advanced the stream-global event ID from 0
to 5; every event had a valid current observation, 17-point pose, and temporal
probability 1.0. Multiple events are expected from the looping positive clip and
prove the functional path only; they are not Accuracy/Recall measurements.
For these legacy runs, `pipeline_ms` remains the hailonet probe described above, not camera-to-MQTT
latency. Power is N/A because this Pi exposes no reliable board-power telemetry.

The raw compressed MQTT, resource samples, application logs, summary, and
SHA256 entries are under
[`../../evaluation/reports`](../../evaluation/reports). The pre-existing
`mcp_face_rec` container was stopped only for the authorized Hailo-exclusive
window, then restarted and verified `healthy`.

Spark provides a LAN control stream
at `rtsp://192.168.3.42:8554/fall-e2e-low` and a person-positive GMDCSA S4
Fall/01 loop at `rtsp://192.168.3.42:8554/fall-person`. Both were ffprobe-verified
from the Pi as H.264 Constrained Baseline 640x640@15; LAN RTT averaged 3.96 ms.
The person-positive E2E and native temporal-profile evaluation are complete.
The subject-disjoint extraction, training, freeze, and integration procedure is
documented in [`TEMPORAL_TRAINING.md`](TEMPORAL_TRAINING.md).

## Hailo-native temporal profile

The Hailo-8 pose frontend produced all 160 GMDCSA traces in 307 seconds with
zero failed clips. The service that normally owns the accelerator was restored
immediately afterward and verified healthy. The trace tree is backed up on
Spark; its verified digest is
`0cbd4f6f50c3a9907ccc67010d895f96c5ed71fb33c916f5d534e17bf79816ff`.

Subjects 1-2 fit the model, Subject 3 selected the configuration, Subjects 1-3
refit it, and only after the freeze manifest was written was Subject 4 read.
The selected 48-frame profile uses all features, 16 hidden units, alpha 0.01,
threshold 0.75 and three consecutive evaluations. The generated 175,450-byte
header SHA256 is
`dec7237a1204cd2d9d54aa6810ca941ef82b83a18e65bb06a4eb7893bb55faf9`.

| Split / output | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | Mean latency | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Subject 3 development temporal gate | 20 | 1 | 22 | 0 | 97.67% | 95.24% | 100% | 100% | 97.56% | 1.105 s | — |
| Subject 4 clean frozen temporal gate | 12 | 0 | 12 | 3 | 88.89% | 100% | 80% | 80% | 88.89% | 1.608 s | 92.02% |
| Subject 4 old geometry heuristic | 4 | 8 | 13 | 2 | 62.96% | 33.33% | 86.67% | 66.67% | 44.44% | 0.005 s | 92.02% |

The clean test has 27 clips (12 Fall, 15 ADL), 2,981 emitted frames and 2,743
valid pose17 frames. These are temporal-gate metrics; they must not be relabeled
as the complete deployed state-machine result. Reports, coverage details,
freeze manifest and checksums are under `evaluation/reports`. The runtime now
constructs every Hailo track with `Hailo8YoloV8sPose`; Pi-native compile, logic
tests and contract tests passed after the switch.

Python control-plane feasibility and the exact HailoRT 4.21 ABI boundary are
documented in [`PYTHON_CONTROL_PLANE.md`](PYTHON_CONTROL_PLANE.md).

## 2026-08-30 多路 RTSP 路数边界

Spark LAN MediaMTX 受控源为 H.264 Constrained Baseline 640x640@15 FPS、约
1.2 Mbps、GOP30。warmup 10 秒、测量 60 秒，目标每路至少 14.5 FPS。`ENABLE_MQTT=OFF`
（Pi 缺少 mosquitto development headers），所以以下是 RTSP→软解→Hailo→pose
decode/tracker→payload construction 吞吐，不包含 broker publish，也不替换既有 MQTT
contract 证据。

| 配置 | 15 路 | 16 路 | 17 路 |
|---|---:|---:|---:|
| queue=2, drop=false | 15.0207–15.104 FPS；53.27–57.42 ms | 14.6152–14.6652 FPS；44.30–51.75 ms | 13.0236–13.057 FPS；fail |
| queue=1, drop=false | 14.9320–14.9987 FPS；40.53–43.72 ms | 14.5215–14.5715 FPS；36.16–40.93 ms | 13.2264–13.2597 FPS；fail |
| queue=1, drop=true | 14.6652–14.8485 FPS；37.03–40.78 ms | 14.3982–14.5815 FPS；fail | — |

当前真实 RTSP 最大通过路数为 16（queue=1、drop=false）。queue=1 主要减少在途旧帧
造成的陈旧帧延迟，不宣称提升吞吐。`RTSP_DROP_ON_LATENCY=true` 可用于低延迟丢帧策略，
但本轮 16 路最低 FPS 为 14.3982，生产默认保持关闭。完整记录见
[`../../evaluation/reports/rpi-hailo8-multistream-20260830.json`](../../evaluation/reports/rpi-hailo8-multistream-20260830.json)。

最终源码在 Pi 上以 `BUILD_APP=ON`、`ENABLE_MQTT=OFF` 做 Release 构建并通过 6/6
CTest。未显式设置 queue/drop 的默认配置复验 16 路为 14.5828–14.6328 FPS/路，
CPU 239%、RSS 1,258,256 KiB、70.8°C。

### Official YOLOv8m-Pose benchmark (2026-08-30)

Official Hailo Model Zoo v2.19.0 Hailo-8 `yolov8m_pose.hef` is 31,608,992 bytes,
SHA256 `fa0bfbf83dba494f4d75ec2fd0ef497ca9d402a65c324afc9865ffc327a53514`, with
3 contexts and 9 raw outputs. On `harvest-pi` with HailoRT 4.21, bare HailoRT
measured 30.87–30.98 FPS and 26.92–26.97 ms hardware latency. Synthetic app
Before the shared backend was added, throughput on the per-stream batch-1 path
was 30.0 FPS (one stream),
15.4862 / 15.4695 (two), and 10.3143 / 10.3143 / 10.3309 (three). Controlled
RTSP 640x640@15 on that path measured 15.0098 /
14.9932 FPS for two streams and 10.3278 / 10.3278 / 10.3111 for three; at the
14.5 FPS target this historical path reached two RTSP streams. This is the
pre-optimization baseline, not the current maximum; the shared auto-batch
result below supersedes it. `ENABLE_MQTT=OFF` excluded broker publishing
because mosquitto development headers were unavailable.

Bare-HEF batching increased total throughput to 69.38 FPS at batch 4
(45.67 ms hardware latency per batch) and 86.91 FPS at batch 8 (71.08 ms per
batch), or 2.25x and 2.81x the batch-1 throughput. The shared auto-batch backend
now exploits this capacity. With synthetic `test://ball` input it measured
17.4122 FPS on each of four streams (69.6488 FPS total, 12 seconds, auto batch
4). Five streams with auto batch 8 measured 17.4061, 17.4894, 17.4894, 17.4894,
and 17.4061 FPS (87.2804 FPS total); max/min fairness was 1.0048.

The controlled RTSP boundary used 10 seconds warmup and 60 seconds measurement.
Five streams measured 14.9826, 14.9993, 14.9993, 15.0159, and 14.9993 FPS, so
all passed the 14.5 FPS target. Six streams measured 14.2043, 14.2209,
14.0544, 14.0044, 13.9545, and 14.0211 FPS, so all fell below the target. The verified
YOLOv8m-Pose maximum is therefore **five 15 FPS RTSP streams**. A five-stream
resource sample was 72.5% CPU, 234,368 KiB RSS, and 63.9°C. MQTT publishing was
disabled for these runs, so they do not replace the existing payload-contract
evidence.

Raw application logs and their SHA256 values are recorded in
[`../../evaluation/reports/rpi-hailo8-yolov8m-pose-20260830.json`](../../evaluation/reports/rpi-hailo8-yolov8m-pose-20260830.json),
along with the Pi-native build and 6/6 CTest log.

The scheduler choice accounted for the remaining gap after buffer reuse:
round-robin scheduling measured 12.5743 FPS per stream at synthetic batch 4,
while scheduler-disabled direct HailoRT measured 17.4002 FPS per stream. The
official HEF, its digest, its three contexts, and its nine raw outputs were
unchanged; the improvement came from shared batching, full-batch padding, and
removing round-robin wait overhead rather than recompiling the model.

The local m-model compile is not a result artifact: 64 calibration images reduced
the optimization level to 1, and GPU noise analysis failed with a malformed device
name. A diagnostic-skip retry completed QAT; its active multi-context allocator was
stopped after about 3.5 hours when the official HEF was found. It did not report a
timeout or allocation failure. Structured evidence is in
[`../../evaluation/reports/rpi-hailo8-yolov8m-pose-20260830.json`](../../evaluation/reports/rpi-hailo8-yolov8m-pose-20260830.json).
