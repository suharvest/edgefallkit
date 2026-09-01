# Jetson multi-stream fall detection

## One-command model preparation and deployment

The published model-free ARM64 runtime is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-jetson:0.1.0-rc2`
with RepoDigest
`sha256:87cd652844d05eb17c7f16c9f8c95d23e5a9abda10692be10fa0ceb447750d9b`.
It carries the non-blocking CUDA stream path and process sharding described
under *Process sharding for stream density*, and was pulled back and verified
on AGX Orin, Orin NX Super and Orin Nano Super. The previous
`0.1.0-rc1` digest is
`sha256:162824bedda86eeadb1bc265b21ae14bb264ad907f68f3ea001db745e38f32ff`.
The slim Compose uses this image by default and supports
`FALL_DETECTION_IMAGE` for a pinned mirror or local build.

From the project root, prepare an external model and build TensorRT on the
destination Orin:

```bash
./deploy.sh jetson --model yolo11s-pose --device orin-nano \
  --accept-upstream-license --deploy
```

Use `yolo11m-pose` with `orin-nx`. The helper requires an explicit upstream
license acknowledgement, exports ONNX in an isolated `uv` environment, builds
the FP16 engine with the target's TensorRT 10.3/SM87, records a provenance
manifest, and sends only the engine and manifest into the deployment checkout.
A matching manifest is a cache hit. Use `--offline --onnx PATH` for an existing
ONNX, `--dry-run` to inspect the plan, and `--remote-project-dir` when the target
checkout is not `/home/harvest/fall-detection`.

The runtime image contains neither the ONNX nor the engine. The Apache-2.0
project license does not relicense models obtained by this helper; converted
artifacts retain their upstream provenance.

This solution is the Jetson Orin Nano/NX runtime counterpart to the reCamera
`fall-detection` application. It keeps compatibility fields used by the
reCamera MQTT payload and adds `stream_id` plus independent `persons[]` entries
for every tracked person.

The deployment process is split in two stages:

1. Export YOLO11-Pose to ONNX on a development machine (Python is allowed in
   this stage).
2. Build the final `.engine` on the target Jetson with its **host TensorRT
   10.3 + CUDA + SM87** using `tools/build_engine.sh`.

The deployed control plane is a small, readable `app.py`. It does not install
or import PyTorch, Ultralytics, ONNX Runtime, MMDeploy, or another heavy Python
inference stack. Python owns RTSP reconnects, per-stream orchestration, MQTT,
and per-track state. The hot data plane is the `libjetson_fall_trt.so` C++17
shared library called through ctypes: it wraps CUDA preprocessing,
TensorRT `enqueueV3`, YOLO parsing/NMS, and the compact learned temporal gate.
No image tensor or per-pixel loop crosses into Python. OpenCV captures through
an explicit Jetson `nvv4l2decoder ! nvvidconv` GStreamer pipeline; MQTT uses
the lightweight system `python3-paho-mqtt` client.

The frame path is intentionally explicit:

```text
RTSP -> nvv4l2decoder (NVDEC) -> nvvidconv (Jetson VIC) -> appsink BGR host
     -> reusable pinned host staging + cudaMemcpy2DAsync (reused device BGR staging)
     -> CUDA preprocess kernel (resize/letterbox/RGB/normalize/NCHW)
     -> TensorRT enqueueV3 on the same CUDA stream
```

The capture buffer is copied row-wise into reusable pinned host staging, then
to device staging with an asynchronous H2D transfer; this is **not zero-copy**.
The two capture-side allocations grow geometrically and are reused, so normal
frames do not call `cudaMalloc`/`cudaHostAlloc`. The pinned staging the result
copy lands in is sized to the engine's output on first use and reused unchanged
after that, since a fixed-shape engine never asks for more.
Preprocessing and inference share one stream, and the kernel writes directly
to TensorRT's FP32 or FP16 input allocation. Python still only receives the
small detection/keypoint arrays through ctypes.

## Model choice and measured Orin performance

The default recommendation is **YOLO11s-Pose FP16 for Orin Nano** and
**YOLO11m-Pose FP16 for Orin NX**. The s model leaves more decoder and
multi-stream scheduling headroom; the m model raises the published COCO pose
mAP50-95 from 58.9 to 64.9 (the n model is 50.0) when accuracy matters more
than stream density. These task metrics come from the official Ultralytics
pose documentation: <https://docs.ultralytics.com/tasks/pose/>.

The following engines were built on the target itself with TensorRT 10.3,
CUDA 12.6, SM87, FP16, fixed 1x3x640x640 input and then measured with
`trtexec --useCudaGraph --noDataTransfers`. They are **inference-core**
figures, not end-to-end RTSP rates:

| Engine | Orin Nano | Orin NX |
|---|---:|---:|
| YOLO11s-Pose | 12.20 ms / 81.97 FPS | 10.73 ms / 93.18 FPS |
| YOLO11m-Pose | 21.30 ms / 46.95 FPS | 18.58 ms / 53.83 FPS |

For an initial 15 FPS deployment budget, start conservatively with 4 s-model
streams on Nano or 3 m-model streams on NX, then measure the real cameras,
codec, scene density, MQTT load and power mode. The raw numbers suggest higher
ceilings, but do not include NVDEC/VIC, H2D, CUDA preprocessing, output copy,
NMS or tracking and therefore are not a route guarantee.

### Multi-stream compilation and capacity

No special "concurrent engine" compilation is required for independent RTSP
streams. The production default remains a fixed **batch=1** engine. Each active
stream must own a distinct TensorRT execution context, CUDA stream and input /
output / staging buffers; an execution context must never be entered from two
threads at the same time. TensorRT can share one deserialized `ICudaEngine`
between those contexts. The current Python/C bridge is safe but more
conservative: every `TrtBridge` owns a complete `TrtRunner`, so it also
deserializes another engine per stream. That does not prevent concurrency, but
it uses more RAM than a future process-level shared-engine cache.

The following MAXN_SUPER measurements use TensorRT 10.3
`trtexec --useCudaGraph --noDataTransfers --infStreams=N`; `infStreams` creates
independent contexts and inference streams over one engine. Existing services
were deliberately left running: Nano also had a one-stream fall-detection RTSP
container, while NX retained its edge-LLM and voice services. Therefore these
are coexistence numbers, not an idle-device peak claim.

| Device / model | Contexts | Aggregate FPS | FPS/context | Mean / P95 | trtexec CPU | trtexec RSS mean/max | GPU | Board power mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nano / YOLO11s | 1 | 69.66 | 69.66 | 14.35 / 22.02 ms | 8.9% | 249.5 / 269.7 MiB | 94.8% | 16.81 / 17.85 W |
| Nano / YOLO11s | 2 | 71.12 | 35.56 | 28.11 / 34.14 ms | 8.6% | 261.9 / 283.0 MiB | 93.1% | 17.57 / 18.77 W |
| Nano / YOLO11s | 3 | 70.07 | 23.36 | 42.75 / 48.49 ms | 11.3% | 273.2 / 295.3 MiB | 91.3% | 17.47 / 18.53 W |
| Nano / YOLO11s | 4 | 70.15 | 17.54 | 56.78 / 64.11 ms | 12.7% | 284.0 / 307.0 MiB | 91.2% | 17.70 / 18.73 W |
| Nano / YOLO11s | 6 | 70.85 | 11.81 | 84.53 / 92.77 ms | 15.7% | 306.0 / 330.7 MiB | 91.2% | 17.63 / 18.53 W |
| NX / YOLO11m | 1 | 53.83 | 53.83 | 18.57 / 18.60 ms | 6.9% | 203.3 / 219.7 MiB | 91.4% | 23.87 / 25.16 W |
| NX / YOLO11m | 2 | 54.27 | 27.14 | 36.83 / 37.04 ms | 8.2% | 214.6 / 231.9 MiB | 91.4% | 24.97 / 26.39 W |
| NX / YOLO11m | 3 | 53.60 | 17.87 | 55.80 / 60.34 ms | 9.6% | 226.4 / 244.6 MiB | 91.4% | 24.85 / 26.23 W |
| NX / YOLO11m | 4 | 53.60 | 13.40 | 74.31 / 89.40 ms | 10.7% | 237.4 / 256.6 MiB | 91.4% | 25.04 / 26.38 W |
| NX / YOLO11m | 6 | 53.54 | 8.92 | 110.85 / 127.91 ms | 13.7% | 260.3 / 281.4 MiB | 91.4% | 25.33 / 26.68 W |

Total throughput is almost flat once the GPU is saturated; contexts divide
that budget rather than multiplying it. At a 15 FPS target the measured
inference-core boundary is therefore **four YOLO11s streams on Nano** and
**three YOLO11m streams on NX**. Six contexts are stable, but not fast enough
per route. Keep deployment below this boundary until real RTSP decode,
preprocessing, NMS, tracking and MQTT headroom has also been measured.

### Production RTSP end-to-end validation

The latest slim image was rebuilt after aligning the Jetson payload with the
cross-platform MQTT contract. A Spark LAN MediaMTX source supplied H.264
Constrained Baseline 640x640@15 FPS at about 1.2 Mbps. The production Python
application used Jetson NVDEC/VIC, the native TensorRT bridge, tracker,
temporal model and state machine. The publisher boundary was intercepted to
capture the exact JSON passed to MQTT; broker transport latency was not part
of this run.

| Device / model | Output FPS | Output interval P95 | Infer mean/P95 | CPU mean | RSS mean | GPU mean/P95 | Board power mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nano / YOLO11s | 14.94 | 69.63 ms | 14.76/14.79 ms | 11.08% | 211.3 MiB | 24.82/77% | 9.07/9.38 W |
| NX / YOLO11m | 15.02 | 69.74 ms | 20.98/21.03 ms | 10.43% | 126.7 MiB | 28.48/91% | 13.85/14.11 W |

All 918 Nano and 932 NX low-stream payloads passed
`contracts/validate_payload.py`, including stream-global `event_id`,
`global_event_id`, `event_id_scope`, empty-frame feature defaults and
top-level/per-person `keypoints`/`pose17`. A second looped GMDCSA S4 Fall/01
RTSP source exercised the complete positive path: Nano produced two fall-event
edges and NX one, both with person tracking, 17-point pose, temporal-positive,
fallen and recovering outputs. This is a runtime path check, not an accuracy
measurement. The output-interval P95 is measured between published messages;
it is not camera-capture-to-MQTT latency.

Reproduce and summarize the test with:

```bash
tools/benchmark_multicontext.sh models/model.engine /tmp/result 1 2 3 4 6
tools/summarize_multicontext.py /tmp/result
```

### Calibrated INT8 engines

`trtexec --int8` without calibration does not produce a detection-valid
comparison. Build INT8 engines on each destination Orin with an explicit image
manifest and a model-specific cache:

```bash
tools/make_gmdcsa_calib.sh /data/gmdcsa24 /data/calib-yolov8 64
python3 tools/build_calibrated_int8.py \
  --onnx models/yolov8s-pose.onnx \
  --manifest /data/calib-yolov8/calibration.txt \
  --cache models/yolov8s-pose.int8.cache \
  --engine models/yolov8s-pose.int8.engine
```

The builder uses centered 640-square letterbox preprocessing, RGB order and
`float32 / 255`, then enables INT8 with FP16 fallback for layers without an
INT8 implementation. Keep separate cache files for different ONNX graphs.
Performance qualification still requires a real-image application smoke test;
successful random-input `trtexec` timing alone does not prove usable pose
outputs.

The 2026-09-01 idle-device run used the same YOLOv8 ONNX files and the same 64
calibration image contents on both Orins. Each core row is the median of three
120-second runs after 30 seconds warm-up:

| Device / model | FP16 FPS / mean | Calibrated INT8 FPS / mean | INT8 speedup |
|---|---:|---:|---:|
| Orin Nano Super / YOLOv8s-Pose | 169.66 / 5.89 ms | 266.34 / 3.75 ms | 1.57x |
| Orin Nano Super / YOLOv8m-Pose | 76.64 / 13.05 ms | 123.22 / 8.11 ms | 1.61x |
| Orin NX Super / YOLOv8s-Pose | 192.37 / 5.20 ms | 299.61 / 3.34 ms | 1.56x |
| Orin NX Super / YOLOv8m-Pose | 88.26 / 11.33 ms | 139.16 / 7.18 ms | 1.58x |

Two and four `--infStreams` kept aggregate INT8 throughput near the one-stream
result while dividing per-context FPS and increasing per-context latency. More
contexts therefore provide scheduling isolation, not proportional throughput.
The real RTSP application also produced person detections with all eight
device/model/precision combinations. INT8 and FP16 fall-state counts differed,
including zero M INT8 fall-state frames in the measured windows; this smoke
test is not an accuracy equivalence result.

The embedded `yolov8-int8-pose` temporal profile was subsequently fit with S/M
INT8 Subjects 1–2 traces, selected on Subject 3, refit on Subjects 1–3, and
evaluated without using Subject 4 for selection. The 27-clip deployed evaluator
produced the following idle-device results:

| Device / frontend | Temporal gate TP/FN/TN/FP, F1 | Deployed TP/FN/TN/FP, F1 | Pose coverage | Mean inference |
|---|---|---|---:|---:|
| Orin Nano Super / YOLOv8s INT8 | 9/3/13/2, 78.3% | 7/5/13/2, 66.7% | 78.3% | 5.28 ms |
| Orin NX Super / YOLOv8s INT8 | 9/3/13/2, 78.3% | 7/5/13/2, 66.7% | 78.3% | 4.72 ms |
| Orin Nano Super / YOLOv8m INT8 | 4/8/12/3, 42.1% | 0/12/12/3, 0.0% | 64.7% | 9.87 ms |
| Orin NX Super / YOLOv8m INT8 | 5/7/12/3, 50.0% | 0/12/12/3, 0.0% | 64.5% | 8.80 ms |

The M result is bounded by frontend coverage rather than Orin throughput.
Subject 4 Fall/03, Fall/04 and Fall/08 pose coverage changed from
76.0%/54.9%/57.4% with S to 23.0%/7.0%/42.6% with M. The existing calibration
candidates include Subject 4, so this table is engineering validation and does
not replace the publication-grade frozen accuracy table. See the
[development report](evaluation/temporal-yolov8-int8-development.json) and the
[Nano S](evaluation/eval-nano-s-int8.json),
[Nano M](evaluation/eval-nano-m-int8.json),
[NX S](evaluation/eval-nx-s-int8.json), and
[NX M](evaluation/eval-nx-m-int8.json) deployed reports. See also the
[structured cross-precision report](../../evaluation/reports/jetson-yolov8-crossprecision-20260901.json).

`tools/build_engine.sh` assumes dynamic input shapes. For an ONNX graph already
fixed at `1x3x640x640`, set `TRT_STATIC_SHAPE=true` so the script does not pass
an unnecessary optimization profile.

### Static batch throughput experiment

The source ONNX inputs are genuinely dynamic (`images=[batch,3,height,width]`),
so no graph rewriting was used. Static batch=4 engines were built locally on
each target with the same TensorRT 10.3 / FP16 / SM87 toolchain:

```bash
tools/build_engine.sh model.onnx model.b4.fp16.engine \
  --minShapes=images:4x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --memPoolSize=workspace:1024
```

`trtexec` reports queries (batches) per second, so image throughput below is
`batches/s * static batch`.

| Device / model | Static batch | Engine size | Batches/s | Images/s | Batch mean / P95 | Board power mean |
|---|---:|---:|---:|---:|---:|---:|
| Nano / YOLO11s | 1 | 28,217,060 B | 69.66 (coexistence run) | 69.66 | 14.35 / 22.02 ms | 16.81 W |
| Nano / YOLO11s | 4 | 23,033,340 B | 44.99 | **179.94** | 22.23 / 22.79 ms | 20.81 W |
| NX / YOLO11m | 1 | 51,571,548 B | 53.83 | 53.83 | 18.57 / 18.60 ms | 23.87 W |
| NX / YOLO11m | 4 | 44,795,132 B | 26.49 | **105.96** | 37.75 / 38.02 ms | 30.38 W |

Static batching is a throughput option, not the current production ABI. The
deployed bridge forces batch=1 and returns one frame's detections. Enabling
batch=4 requires a cross-stream micro-batcher, batched output demultiplexing,
and a maximum wait policy before results are routed back to each stream's own
tracker/state machine. At 15 FPS, accumulating four frames from one camera adds
up to about 200 ms before GPU work. Batching one frame
from several cameras can reduce that wait, but becomes sensitive to RTSP jitter
and the slowest input. For live fall alerts, batch=1 independent contexts remain
the default; batch=4 should be enabled only when throughput matters more than
alert latency and after a bounded-wait scheduler is implemented and measured.

The reference YOLO weights are distributed by Ultralytics under AGPL-3.0.
Confirm that license is suitable for the product, obtain the appropriate
commercial license, or replace the pose model with a compatible-licensed raw
TensorRT pose engine before shipping a closed commercial deployment. The
runtime/parser itself is model-file agnostic within the documented output
contract.

### Process sharding for stream density

One Python process stops scaling well before the accelerator does. Per-frame
control-plane work — payload construction, `json.dumps`, and the paho publish
path — all runs under one GIL. A `py-spy --gil` sample of a saturated
AGX Orin process (520 samples, 16 streams) attributes that time as
`json.dumps` 22.5%, paho publish 21.0%, `_payload`/`as_json` 14.4%,
`TrtBridge.infer` (ctypes call plus keypoint marshalling) 12.7%,
`temporal_update` 5.8%, remainder in tracking and the read loop.

`app.py` therefore shards itself across OS processes:

```json
"runtime": {
  "workers": "auto",
  "max_streams_per_worker": 7
}
```

`auto` starts `ceil(enabled_streams / max_streams_per_worker)` processes and
distributes streams round-robin, so shard sizes differ by at most one. A
supervisor process owns the children and restarts a shard that exits, with a
5 s backoff. `workers` also accepts an explicit integer. Omitting the section,
or setting `max_streams_per_worker` to 0, keeps the historical single-process
behaviour, and a stream count at or below the calibration also resolves to one
process with no extra supervisor overhead.

Each shard opens its own broker session: the configured `mqtt.client_id` gains
a `-<shard>` suffix. Sharing one client id makes the broker evict the previous
session and every publish then fails with `rc=4`.

`max_streams_per_worker` is a per-device calibration, not a constant. It is the
number of streams one process sustains at the target frame rate. Measured at
15 FPS with YOLO11s-Pose, one AGX Orin process holds 7 streams and caps near
110 published FPS. The default of 7 has not been retuned for Orin NX Super or
Orin Nano Super, and both exceed it before they saturate: their measured maxima
of 9 and 8 streams run as two shards, which is also where each gains — Orin
Nano Super goes from 104.0 aggregate on one process at 7 streams to 119.6 on
two at 8, GR3D 77% to 91%.

Sharding costs memory: every process deserializes its own engine. Measured
container RSS on AGX Orin, YOLO11s-Pose: 8 streams in 2 shards 703 MiB,
16 streams in 3 shards 1161 MiB, 20 streams in 3 shards 1292 MiB.

Measured on three Orin modules. Sources are 640x640 H264 Constrained Baseline
15 FPS 1.2 Mbps loops of GMDCSA-24 `subject-4/Fall/01`, 60 s windows, MAXN.
"Max at 15 FPS" is the largest stream count where every stream sustained at
least 14.5 published FPS.

Orin NX and Orin Nano pulled from an external RTSP host on the same LAN, so the
device under test only decoded and inferred. The AGX Orin rows published
locally with `-c copy`; its host CPU stayed at 4.8 of 12 cores while GR3D was
already 89–99%, so that row is accelerator-bound either way.

| Module | trtexec core | Max at 15 FPS, before | after | Aggregate ceiling before → after | GR3D before → after |
|---|---:|---:|---:|---:|---:|
| AGX Orin 32G (J501) | 322.9 qps / 3.09 ms | 6 | 7 single, **16 sharded** | 98 → 105 (240 sharded) | 59% → 89% |
| Orin NX Super 16G | 173.8 qps / 5.75 ms | 8 | **9** | 116.5 → 134.4 | 77% → 95% |
| Orin Nano Super 8G | 155.3 qps / 6.44 ms | 6 | **8** | 98.2 → 119.6 | 74% → 90% |

The pattern is the same on all three: before the fix GR3D sat at 59–77% while
aggregate throughput refused to rise, because the synchronous output copy on
the legacy default stream serialised every runner. After it, all three reach
89–95% and the accelerator becomes the limit.

`trtexec core` is `--useCudaGraph --noDataTransfers` on an idle device. The
Super modules measure far above the non-Super figures recorded earlier in this
document. Orin NX Super leads Orin Nano Super by 12% in the core measurement
and by 12% end to end: both carry 1024 CUDA cores, and this FP16 GPU path does
not reproduce the ratio their TOPS ratings suggest.

Per-device ramps:

| Module | Streams | Shards | Per-stream FPS | Aggregate | GR3D | Container RSS |
|---|---:|---:|---:|---:|---:|---:|
| AGX Orin | 8 | 1 | 13.21 | 105.7 | 59% | 502 MiB |
| AGX Orin | 8 | 2 | 15.00 | 120.0 | 62% | 703 MiB |
| AGX Orin | 16 | 3 | 14.99 | 239.8 | 89% | 1161 MiB |
| AGX Orin | 20 | 3 | 12.32 | 246.4 | 99% | 1292 MiB |
| Orin NX Super | 6 | 1 | 14.68 | 88.1 | 59% | — |
| Orin NX Super | 8 | 2 | 14.70 | 117.6 | 84% | 664 MiB |
| Orin NX Super | 9 | 2 | 14.93 | 134.4 | 95% | — |
| Orin NX Super | 10 | 2 | 13.05 | 130.5 | 99% | — |
| Orin Nano Super | 6 | 1 | 14.79 | 88.8 | 65% | 390 MiB |
| Orin Nano Super | 7 | 1 | 14.85 | 104.0 | 77% | — |
| Orin Nano Super | 8 | 2 | 14.95 | 119.6 | 91% | — |
| Orin Nano Super | 9 | 2 | 13.36 | 120.3 | 98% | 711 MiB |

Sharding earns its memory on every module, not only on AGX Orin: Orin Nano
Super goes from 104.0 aggregate on one process to 119.6 on two, and its GR3D
rises from 77% to 91% in the same step. The default calibration of 7 shards at
8 streams and above, which is where each board needs it.

## Build the engine on Orin

Obtain the evaluated ONNX from Spark or export the named official model by
following [`models/README.md`](models/README.md). ONNX is deliberately mounted
from `./models` rather than baked into the runtime image. Verify the host tools,
then build the engine on that same device:

```bash
dpkg -l | grep tensorrt-libs       # must report the host's 10.3 package
/usr/src/tensorrt/bin/trtexec --version
chmod +x tools/build_engine.sh
tools/build_engine.sh \
  /models/yolo11s-pose.onnx \
  models/yolo11s-pose.sm87.trt10.3.fp16.engine
```

The helper passes a fixed `images:1x3x640x640` min/opt/max profile by default;
this avoids TensorRT choosing a dynamic ONNX placeholder such as 1x1. If the
export uses another input name, set `TRT_INPUT_NAME` (for example
`TRT_INPUT_NAME=input`) or pass explicit `--minShapes/--optShapes/--maxShapes`
arguments, which suppress the defaults.

Do not copy an engine built on an x86 GPU or another TensorRT release. Engines
are tied to GPU architecture and TensorRT ABI. For a non-standard ONNX input
name/profile, append extra `trtexec` arguments after the two paths, for example
`--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640
--maxShapes=input:1x3x640x640`.

## Compile and run

On a Jetson with native development packages:

```bash
cmake -S . -B build -DBUILD_APP=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target jetson_fall_trt
ctest --test-dir build --output-on-failure
```

The provided Dockerfile is the reproducible route. Its CUDA *devel* build stage
compiles the `.cu` bridge; compose mounts the target's TensorRT/CUDA libraries
at runtime. Pin `L4T_CUDA_DEVEL_IMAGE` to the device's JetPack/CUDA devel tag;
the host's TensorRT 10.3 libraries remain authoritative at runtime.
The mounts are intentional: a base image with a different TRT minor version
must not override the host 10.3 ABI.

```bash
docker compose build
docker compose run --rm fall-detection --validate
docker compose up -d
docker compose logs -f fall-detection
```

Before starting, edit `config/config.json`:

```json
{
  "engine_path": "/models/yolo11s-pose.sm87.trt10.3.fp16.engine",
  "temporal_profile": "auto",
  "mqtt": {
    "host": "192.168.1.20",
    "port": 1883,
    "username": "fall",
    "password": "secret",
    "topic": "recamera/fall-detection/results/{stream_id}"
  },
  "streams": [
    {"id": "cam-a", "rtsp_url": "rtsp://10.0.0.11:554/live"},
    {"id": "cam-b", "rtsp_url": "rtsp://10.0.0.12:554/live"}
  ]
}
```

`temporal_profile=auto` selects `yolov8-int8-pose` when the engine filename
contains both `yolov8` and `int8`; otherwise it selects the independently
trained `yolo11s-pose` or `yolo11m-pose` weights. Set the profile explicitly
when an engine is renamed. The profiles are embedded in the
native library, so switching between the recommended Nano and NX engines does
not require Torch, Ultralytics, scikit-learn, or another runtime model file.

`{stream_id}` is replaced in the MQTT topic. Set `mqtt.tls=true` and provide
`ca_file`, `cert_file`, and `key_file` when the broker requires TLS. Every
stream has its own OpenCV capture, tracker, 48-frame temporal window, and
Python `Track` state; one person's pose cannot contaminate another person's
state. The native bridge handle is also per stream because TensorRT execution
contexts are not shared across worker threads.

The production state machine has one confirmation authority: the learned
temporal gate. Geometry (hip drop, torso angle and box aspect) can move a track
from `normal` to `suspected`, keep that suspicion alive, or drive recovery, but
cannot enter `fallen` when `fall.temporal_confirmation_required` is `true`
(the default). A first frame that is already lying therefore does not emit an
event. The temporal window is still updated during a short detector occlusion,
but a missed/stale track cannot create a new event; a reacquired visible pose
can confirm from the retained history. `fall_event` is an edge (one frame at
the transition); `fall_detected`
remains true through `fallen`/`recovering`, and the configured cooldown blocks
duplicate confirmations. Set `temporal_confirmation_required` to `false` only
for an explicit legacy geometry-only bring-up; this compatibility mode is not
the production/evaluation default.

The default GStreamer pipeline is equivalent to:

```text
rtspsrc location=<url> protocols=tcp latency=100 ! rtph264depay ! h264parse !
nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert !
video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false
```

Set a stream's `codec` to `h265`/`hevc` for the H.265 depay/parser pair.

## Output contract

The top-level object retains reCamera-compatible fields such as
`fall_detected`, `fall_event`, `event_id`, `state`, `person_count`,
`fallen_count`, `tracking`, and `features`. Jetson additions are:

```json
{
  "stream_id": "cam-a",
  "persons": [
    {
      "track_id": 3,
      "fall_detected": true,
      "fall_event": true,
      "event_id": 1,
      "state": "fallen",
      "features": {"temporal_probability": 0.93},
      "bbox": [0.5, 0.52, 0.23, 0.54],
      "pose17": [[0.5, 0.2, 0.9]]
    }
  ]
}
```

Each `persons[].event_id` is monotonic within its `track_id`. The top-level
`event_id` and `global_event_id` are the same stream-global monotonic event
sequence (`event_id_scope=stream_global_event_id`), so a new fall edge always
advances even when tracker IDs are retired or reused. A primary
highest-confidence person's `features` object is also exposed at the top
level for old consumers that do not understand `persons[]`.

## Tensor/output assumptions

The parser accepts common raw YOLO11-Pose output shapes `[1,56,8400]`,
`[1,8400,56]`, `[56,8400]`, and `[8400,56]` (plus arbitrary `3*K` pose tails).
It applies confidence filtering and class-agnostic NMS, reverses letterboxing,
and converts all keypoints to source-frame coordinates. Both FP32 and FP16
TensorRT output tensors are supported.

The lightweight tracker uses greedy IoU/centre matching and retires a track
after `max_missed_frames`. Tune that, score thresholds, MQTT credentials, and
fall state-machine thresholds in `config/config.json`; the structure is
documented by `config/config.schema.json`.

## Host-only tests

The core tests need no CUDA, TensorRT, OpenCV, MQTT, or camera:

```bash
cmake -S . -B /tmp/jetson-fall-build -DBUILD_APP=OFF -DBUILD_TESTING=ON
cmake --build /tmp/jetson-fall-build -j2
ctest --test-dir /tmp/jetson-fall-build --output-on-failure
```

They cover the native temporal fall state machine, both YOLO raw tensor
layouts, multiple-person identity persistence, JSON configuration validation,
and the Python per-track controller (`tests/python_app_test.py`), including
first-frame lying, geometry-only suppression, temporal confirmation during
occlusion, event-edge and recovery behavior. On a host
without OpenCV/CUDA/TensorRT, Python config validation still works with:

```bash
python3 app.py --config config/config.json --validate
```

## Reproducible fall-accuracy evaluation

The stable cross-version comparison ledger is [`RESULTS.md`](RESULTS.md).
Detailed protocol notes, historical baselines and clip-level artifact links are
in [`EVALUATION.md`](EVALUATION.md).

`tools/evaluate_videos.py` decodes public videos, samples them at the frozen
15 FPS timebase, creates fresh tracking/temporal state for every clip, and
reports two deliberately separate metrics. `temporal_gate` is directly
comparable with the reCamera v0.2 report for the baseline; optimized runs use
the selected model profile (both current profiles select probability >= 0.8
for three evaluations). `deployed_alert` is the actual Python state-machine
result.
Alerts over 0.5 seconds before a labelled fall onset count as early false
alerts and as false negatives, matching the original protocol.

```bash
python3 tools/evaluate_videos.py \
  --config config/config.json \
  --engine /models/yolo11s-pose.fp16.engine \
  --library ./build/libjetson_fall_trt.so \
  --gmdcsa /datasets/gmdcsa24 \
  --output evaluation/gmdcsa24-yolo11s-trt10.3.json

python3 tools/evaluate_videos.py \
  --config config/config.json \
  --engine /models/yolo11m-pose.fp16.engine \
  --library ./build/libjetson_fall_trt.so \
  --realbiomfall-manifest /datasets/realbiomfall-manifest.json \
  --output evaluation/realbiomfall-yolo11m-trt10.3.json
```

The JSON includes clip-level trigger time, pose coverage, mean people/frame,
mean and p95 inference time, the confusion matrix, F1 and detection latency.

For model-specific temporal retraining, extract development traces with the
same TensorRT pose frontend. Subjects 1–2 are training data and Subject 3 is
validation data; do not include Subject 4 while selecting a model:

```bash
python3 tools/extract_gmdcsa_traces.py \
  --config config/config.json --engine /models/yolo11m-pose.fp16.engine \
  --library ./build/libjetson_fall_trt.so --dataset /datasets/gmdcsa24 \
  --subjects 1,2,3 --output /traces/yolo11m --resume

python3 tools/train_temporal_for_pose.py \
  --traces /traces/yolo11m --dataset /datasets/gmdcsa24 \
  --header /tmp/temporal_model_weights_yolo11m.h \
  --report /tmp/temporal-yolo11m-development.json \
  --namespace jetson_fall::temporal_yolo11m_weights
```

The search fits only Subjects 1–2, selects feature mask, hidden width,
regularization, threshold and consecutive count only on Subject 3, then refits
the frozen configuration on Subjects 1–3. Subject 4 is read only by the final
test command and must never be used to choose a candidate.
