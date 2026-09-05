# RK3576 / RK3588 aligned test plan

This document is a runbook for the next Rockchip measurement. It records the
test boundary and the artifacts already prepared; it does not report a device
result. The owner released `cat-remote` and `radxa` for testing on 2026-09-05.
Each run still starts with a fresh workload snapshot so a newly started task is
not stopped merely because of that earlier release.

## 1. Release gate and preflight

Before owner release, do not stop or restart an existing `cat-remote` or
`radxa` workload. After release, record the following before deployment:

```text
fleet status cat-remote
fleet status radxa
fleet exec --sudo cat-remote -- df -h
fleet exec --sudo radxa -- df -h
fleet exec --sudo cat-remote -- docker ps --format '{{.Names}} {{.Status}}'
fleet exec --sudo radxa -- docker ps --format '{{.Names}} {{.Status}}'
```

The actual run must record the occupied-process/container owner, board model,
kernel, RKNN runtime version, NPU core mask, free disk, and the exact image
digest. Do not infer a release result from a low-load smoke test. A production
service may be stopped only after its owner has released the device; stop only
the named test container or process and restore the prior service state after
the run.

The aligned workspace is `/mnt/d/edgefallkit-rk-aligned-20260903`. The WSL
toolchain is already prepared with RKNN Toolkit 2.3.2 and ONNX 1.16.1. The
WSL root has about 2.9 GB free and the aligned D: workspace about 1.7 TB free
at preparation time; do not download a root image or rebuild the base
environment during the device run.

## 2. Frozen inputs and conversion matrix

Use the same aligned source ONNX for both boards and both precisions:

| Model | Source SHA256 |
|---|---|
| YOLOv8s-Pose source, before raw-head extraction | `95ab38a6bcad2ec4e77c44b2f6814ed8cc2aa213574da83b74a01e7502b50919` |
| YOLOv8m-Pose source, before raw-head extraction | `b44ece4006060bd708b7a542f25ae76c758ba2827a6ebe2e22b18be548e23a0b` |

Build four artifacts per board: S FP16, S INT8, M FP16, and M INT8. Use
`prepare_aligned_onnx.py` before conversion where required, then use the real
converter CLI. The resulting device artifacts use these exact names:

```text
yolov8s-pose-rk3576-fp16.rknn  yolov8s-pose-rk3576-int8.rknn
yolov8m-pose-rk3576-fp16.rknn  yolov8m-pose-rk3576-int8.rknn
yolov8s-pose-rk3588-fp16.rknn  yolov8s-pose-rk3588-int8.rknn
yolov8m-pose-rk3588-fp16.rknn  yolov8m-pose-rk3588-int8.rknn
```

```bash
python platforms/rknn/tools/prepare_aligned_onnx.py --onnx SOURCE.onnx --out ALIGNED.onnx
python platforms/rknn/tools/convert_pose_rknn.py \
  --onnx ALIGNED.onnx --platform rk3576 --precision fp16 --out S.fp16.rk3576.rknn
python platforms/rknn/tools/convert_pose_rknn.py \
  --onnx ALIGNED.onnx --platform rk3576 --precision int8 \
  --dataset /path/to/calibration.txt --out S.int8.rk3576.rknn \
  --strict-calibration --strict-9-head
```

Repeat with `--platform rk3588` and for M. The 494 generated S1–S3 images
are fresh calibration inputs, not byte-identical copies of the earlier image
set. Their source-image manifest SHA256 is
`c3ccb18c089bbac8375411c0edcb4dc7d88ff705cd054c72ad018d6d5ef74ed8`.
The calibration manifest must contain Subjects 1–3 only. Subject 3 is seen by
the quantizer and therefore is not an independent pipeline-validation set.

Record each artifact SHA256, conversion log, toolkit version, calibration
manifest hash, output count/shapes, and target platform. A conversion success
is not a device success; no conversion or device result is claimed here.

The aligned graph keeps the source weights and exposes nine heads with
channels `(64, 1, 51)` at each of `(80, 40, 20)` spatial resolutions. Score
heads have an explicit Sigmoid; box/keypoint heads remain raw. FP16/INT8 are
converter precision requests, not claims about every internal operation.
Preserve compiler logs when reporting the actual precision configuration.

## 3. Temporal model and accuracy boundary

Use the explicit training implementation below rather than an implicitly
resolved copy:

```text
/home/harvest/fall-int8-eval/edgefallkit/platforms/fall-detection/tools/train_temporal_model.py
SHA256: b2bac0f5247d71058fa7aba9d14304f49658144795804ac8965216115110fa67
```

The split is fixed as follows:

1. Subjects 1–2 fit the temporal model.
2. Subject 3 selects parameters.
3. Subjects 1–3 are refit before the frozen run.
4. Subject 4 is the clean 27-clip holdout.
5. The 10 smoke clips are excluded from the 27-clip accuracy table.

Run the same frozen FP32 temporal profile and threshold against both frontends
for the primary FP16-versus-INT8 comparison, isolating frontend precision. If
an adapted temporal profile is trained for INT8, run it as a separate
secondary comparison and record its training header/report hash. Subject 4
results previously used during M debugging are regression evidence, not
pristine blind validation.

## 4. Throughput benchmark

The benchmark uses an already prepared RGB frame, not RTSP decode. Use the
same model/input on each board and run it for every S/M and FP16/INT8 variant.

The prepared real-person input is
`/home/harvest/project/edgefallkit-work/rk-aligned-20260903/benchmark-input/input.npy`
(640x640x3 uint8),
with `.npy` file SHA256
`5673aa64140b23ddf82f5910bd341af89f0d1990ecf40c8471efd77937d19085`.
Its decoded contiguous array-byte SHA256 is
`8d4ca12b3c594407d7e124c8395164576d26693c01edc796a8769117d3671812`.
The benchmark JSON field `input_sha256` records the latter, whereas transfer
verification records the former; these hashes cover different byte streams
and must not be compared as though one were wrong.
It is derived from one Subject 1 calibration-source frame; no repository
evidence matched the original Jetson input, so identity with that input is not
claimed. The adjacent `input-manifest.json` records source and array hashes.

```bash
PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
python /code/platforms/rknn/benchmark.py \
  --model /models/yolov8m-pose-rk3576-fp16.rknn \
  --contexts 1 --warmup-seconds 30 --duration-seconds 120 --repetitions 3 \
  --core-mask auto --input-npy /input/input.npy \
  --postprocess-backend cpp --json-out /results/rk3576-M-fp16-1ctx.json
```

Use `core_mask=auto` for the aligned baseline. A numeric mask is allowed only
after the installed RKNN Lite package has been queried for the corresponding
runtime constant and the constant name and integer value are saved in the raw
evidence. Do not copy a mask value from another board.

Context selection is adaptive for each of the eight board/model/precision
combinations. Smoke-test 1, 2, 4, ... contexts with 5 seconds of warmup and a
20-second measured window until the first invalid output, process failure, or
throughput regression. If the first failed value is not adjacent to the last
pass, scan the intervening integer counts. Then run the selected passing
boundary with 30 seconds warmup, 120 seconds duration, and three repetitions.
Also retain the adjacent failing or slower-boundary smoke result. Context
counts selected for one precision or model size are not reused without a run.

The benchmark's `rknn_call_ms` is the `RKNNPose.infer()` internal
`RKNNLite.inference()` call, including runtime input/output handling. It is
not a hardware-only number. `pipeline_ms` starts from the prebuilt RGB frame
and includes infer plus pose decode/NMS; it excludes video decode, tracking,
and MQTT. Throughput is completed samples divided by the measured monotonic
repetition wall time. Each context owns its runner and decoder, and all
contexts finish warmup before steady-state timing.

Store the raw JSON with model/input hashes, requested core mask, per-context
and per-repetition statistics, and the aggregate. A failed output (None,
empty, NaN, or Inf) invalidates that run.

## 5. Executable isolated container layout

Extract the verified source bundle below the per-device test directory and use
the platform Compose file from that bundle. Do not mount the source tree over
`/opt/fall-detection`, because that hides the packaged native extension. Add
the following mounts to each isolated `docker compose run --rm` invocation:

```text
HOST/code                         -> /code:ro
HOST/models                       -> /models:ro
HOST/results                      -> /results
HOST/input                        -> /input:ro
HOST/config                       -> /config:ro
HOST/gmdcsa24                     -> /data:ro             # accuracy only
HOST/training                     -> /training:ro         # accuracy only
```

For example, on RK3576, where `HOST` is
`/home/cat/edgefallkit-rk3576-test`:

```bash
cd /home/cat/edgefallkit-rk3576-test/code/platforms/rk3576
FALL_RK_IMAGE='sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc2' \
docker compose --profile benchmark run --rm --no-deps \
  -v /home/cat/edgefallkit-rk3576-test/code:/code:ro \
  -v /home/cat/edgefallkit-rk3576-test/models:/models:ro \
  -v /home/cat/edgefallkit-rk3576-test/results:/results \
  -v /home/cat/edgefallkit-rk3576-test/input:/input:ro \
  -v /home/cat/edgefallkit-rk3576-test/config:/config:ro \
  -e PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
  benchmark python /code/platforms/rknn/benchmark.py \
    --model /models/yolov8s-pose-rk3576-fp16.rknn \
    --contexts 1 --warmup-seconds 5 --duration-seconds 20 \
    --core-mask auto --input-npy /input/input.npy \
    --postprocess-backend cpp --json-out /results/smoke.json
```

The RK3588 command uses the corresponding platform directory, host workspace,
and `rk3588` artifact. Before the run, resolve the local image to its immutable
ID/RepoDigest and record it. An absent image is a blocker unless pulling that
single image has been separately authorized; do not silently use an unrelated
image or build a replacement during measurement.

## 6. Fixed-window RTSP/MQTT validation

After the process is running, use one MQTT subscription for the complete
window; do not reconnect after warmup:

```bash
# Run this on Spark, where mosquitto_sub is installed. The broker and RTSP
# fixture are both hosted at 192.168.3.42 for this campaign.
python platforms/rknn/tools/collect_mqtt.py --host 192.168.3.42 --port 1884 \
  --topic 'edgefallkit/rk3576/RUN_ID/yolov8s-fp16/results/+' \
  --warmup-seconds 30 --duration-seconds 120 \
  --expected-streams rk3576-yolov8s-fp16-cam-00,rk3576-yolov8s-fp16-cam-01 \
  --min-fps 14.5 \
  --raw-out /home/harvest/project/edgefallkit-work/rk-aligned-20260903/results/rk3576/raw.jsonl \
  --summary-out /home/harvest/project/edgefallkit-work/rk-aligned-20260903/results/rk3576/summary.json
```

Generate a new config for every variant and stream count. It must set the
exact model path, `video.backend=gstreamer_mpp`, strict video and C++
postprocess with no fallback, broker `192.168.3.42`, a topic prefix containing
the platform, run ID, model and precision, and stream IDs prefixed the same
way. Each enabled stream uses the campaign RTSP fixture URL (normally
`rtsp://192.168.3.42:8554/fall-e2e-low`) and a distinct fixture instance or
verified independent source when the server exposes them. Save the generated
config and its SHA256 with the result. Never reuse production stream IDs or
the production MQTT topic.

The collector timestamps arrival with `monotonic_ns`. Warmup messages are
discarded by their arrival timestamp, including messages already buffered at
the window boundary. A duration run fails on zero valid messages, any invalid
payload, a nonzero subprocess exit, a missing expected stream, or any expected
stream below 14.5 FPS. FPS is `N / 120`, not the first-to-last message span;
this rejects sparse streams that would otherwise look fast. Re-run the full
window for each repetition; the exact-boundary case is `N=1740` valid messages
per stream, and `N=1739` must fail.

The subscription topic is the configured topic prefix plus `+`; `{stream_id}`
is publisher-side substitution only. The `/opt/fall-detection` entry in
`PYTHONPATH` preserves the packaged C++ postprocess extension when the source
tree is mounted at `/code`.

The report's inference and pipeline latency fields are application metrics.
They do not represent camera-to-alert end-to-end latency and must not be
renamed as such. `receive_minus_frame_timestamp_ms` includes MQTT transport and
clock offset and is not source encode time.

First freeze the exact source/config/profile identity, then evaluate the clean
Subject 4 set for each frontend/profile:

```bash
PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
python /code/platforms/rknn/tools/extract_gmdcsa_traces.py \
  --model /models/yolov8s-pose-rk3576-fp16.rknn --platform rk3576 \
  --config /config/config.json --dataset /data/gmdcsa \
  --output /results/traces --subjects 1,2,3
PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
python /code/platforms/rknn/tools/train_temporal_profile.py \
  --platform rk3576 --traces /results/traces --dataset /data/gmdcsa \
  --output /results/temporal-rk3576.npz \
  --development-report /results/development-rk3576.json \
  --model /models/yolov8s-pose-rk3576-fp16.rknn --require-trace-identity \
  --training-module /training/train_temporal_model.py
PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
python /code/platforms/rknn/tools/evaluate_videos.py --platform rk3576 \
  --model /models/yolov8s-pose-rk3576-fp16.rknn \
  --temporal-model /results/temporal-rk3576.npz \
  --config /config/config.json --dataset /data/gmdcsa \
  --freeze-only --report /results/freeze-rk3576.json
PYTHONPATH=/opt/fall-detection:/code/platforms/rknn:/usr/lib/python3/dist-packages \
python /code/platforms/rknn/tools/evaluate_videos.py --platform rk3576 \
  --model /models/yolov8s-pose-rk3576-fp16.rknn \
  --temporal-model /results/temporal-rk3576.npz \
  --config /config/config.json --dataset /data/gmdcsa \
  --subject 4 --allow-holdout --freeze-manifest /results/freeze-rk3576.json \
  --report /results/frozen-s4-rk3576.json
```

For capacity, run an adaptive scan independently for every S/M and FP16/INT8
variant on both boards. Start at one stream, increase 1, 2, 4, ... until the
first failed 14.5 FPS gate, then test intervening integer counts. The canonical
evidence is three full repetitions at the largest passing N and three full
repetitions at N+1; N+1 must actually fail before N is labelled a measured
maximum. If N+1 passes, continue scanning. A crash, missing stream, invalid
payload, software decode/postprocess fallback, or resource/thermal violation
is a failed capacity run, not a reason to omit that repetition.

At every run boundary also record CPU/RSS, board temperature and frequency,
the selected `core_mask`, 640x640 input geometry, 15 FPS H.264 source, MPP
hardware decoder, and C++ postprocess backend. The 14.5 FPS gate is per
expected stream and per repetition. Capacity is measured by increasing the
number of configured streams: run N streams, then N+1 streams under the same
15 FPS source conditions. The 14.5 FPS gate applies independently to every
expected stream; it is not a substitute for the N+1 capacity run.

## 7. Required evidence bundle

For each board, preserve: preflight output, device/runtime versions, container
image digest, model and calibration hashes, conversion logs, benchmark JSON,
raw MQTT JSONL, collector summary, process/container restoration output, and a
matrix row identifying model size, precision, context count, core mask, and
metric scope. Missing devices or incomplete runs remain `pending`; do not fill
the matrix with projected or copied results.

## 8. Preparation checkpoint (2026-09-03)

Local RK tests: 53 passed, one skipped because the compiled C++ extension is
unavailable on this host, and one existing x86-only deployment dry-run test
excluded on the Spark ARM64 host. This is script validation, not board
validation. Independent source review found no remaining blocking defects.

WSL conversion completed all eight artifacts. The generated manifest and ready
bundle are under `/mnt/d/edgefallkit-rk-aligned-20260903`:

```text
preparation.json SHA256: ab57a6925dd03c84d2dae9b14de33469057874a1a90f48a6d86824466ff80ac7
rk-aligned-ready.tar.gz SHA256: 9a349dd16f7d15d23942cb083f223c79ea1e25b122f4fd105c6910dcc6821c71
```

The manifest records target and precision-request verification for all eight
files. It remains conversion evidence, not a true-device pass.

The Spark task workspace is
`/home/harvest/project/edgefallkit-work/rk-aligned-20260903`. It retains source
calibration images, the fixed benchmark input and the task-local build and
manifest scripts. Transfer the verified board-specific models directly through
Fleet; no MacBook relay is needed. Keep production defaults unchanged and
restore every workload that was running before the isolated test.
