# Raspberry Pi 5 + Hailo-8 fall detection

Native C++ runtime for the shared fall-detection algorithm. Frames are decoded
by GStreamer, resized to RGB 640x640, and inferred by the native `hailonet`
plugin. The process decodes the nine quantized YOLOv8s-Pose output tensors,
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
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc1`.
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

`pipeline_ms`/`pipeline_time_ms` are measured from the buffer immediately before `hailonet`
to its source pad, so it includes Hailo scheduling/output transfer but not RTSP
decode, resize, C++ postprocess, tracking, or MQTT. Hailo's GStreamer element
does not expose the accelerator-call duration at this probe, so
`inference_time_ms` is `0` with `inference_time_metric=unavailable`; it must not
be mislabeled with the larger probe latency.
`hailortcli benchmark` for pure NPU throughput.

## Benchmark

Set `BENCHMARK_SECONDS=30` and use `test://ball` to isolate inference from RTSP:

```bash
STREAMS='one|test://ball' BENCHMARK_SECONDS=30 docker compose run --rm fall-detection
STREAMS='one|test://ball;two|test://ball' BENCHMARK_SECONDS=30 docker compose run --rm fall-detection
```

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

The RC image was pushed and pulled back on `harvest-pi` as
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc1`.
The registry RepoDigest is
`sha256:1243fb26141a43f67434e1954e6f7ff227e27b8de8aabe2a50a0e3cb48f823a0`;
device-side inspect after the pull reports 144,679,212 bytes, Linux/ARM64, and
entrypoint `/usr/local/bin/fall-hailo`. A metadata-only container smoke verified
the executable while deliberately mounting neither `/dev/hailo0` nor the HEF,
so it did not exercise or compete for the NPU. The existing `mcp_face_rec`
container remained healthy.

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
`pipeline_ms` remains the hailonet probe described above, not camera-to-MQTT
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
