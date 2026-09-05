# RK3576 measured results

Measured 2026-08-13 on Fleet `cat-remote`, LubanCat RK3576, kernel
`6.1.99-rk3576`, NPU driver 0.9.8, `rknn-toolkit-lite2` Python package 2.3.0,
`librknnrt` 2.3.2. Governor was `rknpu_ondemand`, available frequencies
300-950 MHz, sampled at 950 MHz before the run. A pre-existing idle voice
service remained running; it was not stopped.

Model: YOLO11n-Pose raw nine-head, fixed 1x640x640 RGB, FP16, RKNN Toolkit
2.3.2, target `rk3576`, size 10,532,939 bytes, SHA256
`659519ae8179749925c3f15d978b760f6040a00ae01666c9050036077bac8bbd`.

| Input / contexts | Iterations | Throughput | Inference mean / p95 | Pipeline mean / p95 | RSS max | Detections |
|---|---:|---:|---:|---:|---:|---:|
| blank 640 / 1 | 200 | 17.50 FPS | 56.13 / 63.82 ms | 56.89 / 64.67 ms | 179.2 MiB | 0 |
| blank 640 / 2 | 200 | 29.15 FPS | 67.10 / 73.65 ms | 68.35 / 75.35 ms | 246.1 MiB | 0 |
| model-zoo bus / 1 | 500 | 15.15 FPS | 63.03 / 74.44 ms | 65.73 / 77.98 ms | 174.4 MiB | 4 |

`inference` is only `rknnlite.inference()`. `pipeline` adds raw-head
decode/DFL/keypoints/NMS, but excludes RTSP decode, tracker, temporal MLP and
MQTT. During the 500-frame bus run, one snapshot measured Python at 63.1% CPU,
162,616 KiB RSS and NPU load `Core0 36%, Core1 0%`; this is a snapshot, not an
integrated utilization average. The full one-frame RKNN -> four poses ->
tracker -> temporal MLP -> state-machine smoke passed (two poses met the
four-joint validity gate and all tracks stayed normal).

## Frozen temporal-profile accuracy

The RK3576 pose frontend extracted all 160 GMDCSA clips with zero failures.
The temporal configuration was fit on Subjects 1-2, selected on Subject 3,
refit on Subjects 1-3, and frozen before Subject 4 was read. The selected
profile is pose features, 32 hidden units, alpha 0.01, threshold 0.80 and three
consecutive positives. Subject 3 development accuracy/F1 was 95.35%/95.45%.

The clean, frozen Subject 4 test excludes the 10 clips previously used for
pipeline smoke, leaving 27 clips: TP=12, FN=0, TN=12, FP=3; accuracy 88.89%,
recall 100%, specificity/precision 80%, F1 88.89%, no early fall alerts, mean
detection latency 1.492 s and median 1.15 s. False positives were ADL 06, 07
and 16. This is temporal-gate clip accuracy for the independently extracted
RK3576 traces, not a reused Jetson metric or end-to-end state-machine accuracy.
The deployed profile SHA256 is
`dadcb916135e7dec284fe51a49432573e5451ba9c921325df119fbd2e33ebd82`.

Raw JSON: `results/rk3576-host-1ctx.json`, `rk3576-host-2ctx.json`,
`rk3576-bus-1ctx.json`, and `rk3576-bus-utilization.json`.

The shared multi-stage ARM64 runtime image was built and audited on the Mac:
144,142,723 bytes unpacked (`docker image inspect`) and 143,348,208 bytes as a
streamed gzip level-1 `docker save`. It contains NumPy/OpenCV but no model,
Torch, Ultralytics, ONNX Runtime or RKNN Toolkit; RKNN Lite and librknnrt are
read-only host mounts. The device-side rebuild hit a Docker base-layer
`unexpected EOF`; the host-native RKNN results above are unaffected.

## Spark LAN RTSP production path

The production app was also run against Spark MediaMTX H.264 Baseline
640x640@15 (`/fall-e2e-low`). This includes OpenCV/FFmpeg decode and letterbox,
native RKNN, raw-head decode/NMS, tracker, temporal MLP, state machine, MQTT
serialization and delivery to the Mac broker. Existing voice service remained
running; no NPU reset was performed.

| Streams | MQTT received FPS | Source frame-id rate | infer mean/P95 | pipeline mean/P95 | CPU/RSS/NPU snapshot |
|---:|---:|---:|---:|---:|---|
| 1 | 4.88 | 10.82 | 69.56/94.05 ms | 70.76/96.16 ms | 64.19%; 178.8 MiB; Core0/1 30%/0% |
| 2 | 3.82 + 3.80 | 9.15 + 9.17 | 75.91/111.00; 75.12/109.30 ms | 77.41/112.49; 76.54/110.97 ms | 94.06%; 277.6 MiB; Core0/1 35%/26% |

QoS0 observation contained frame-id gaps, so `MQTT received FPS` and the
source frame-id rate are reported separately; neither is relabelled as pure
inference capacity. `pipeline_ms` starts after a decoded frame is returned.
Decoded-frame-to-broker-receive P95 was 264.05 ms (one stream) and
364.0/418.7 ms (two streams); source encode/network delay is not timestamped.

The looping GMDCSA Subject-4 Fall/01 positive stream produced 500 contract-valid
messages: 289 visible-person/pose17 messages, 410 with a retained track and
temporal classifier, 4 stream-global fall events, all four states including
`recovering`, and up to two visible people. This proves the full positive
runtime path, not dataset accuracy. Raw evidence is backed up at
`spark:/home/harvest/datasets/fall-detection/evaluation/rk-e2e/20260813/`;
summary JSON is in `results/rk3576-spark-*.json`.

## GStreamer MPP/RGA + C++ postprocess E2E (2026-08-13)

Read-only inventory on `cat-remote` found `rockchipmpp:mppvideodec`
(`gst-rockchip` 1.14.4) and no standalone RGA element. The decoder accepts
parsed AU-aligned H.264/H.265, exposes RGB/BGR/NV12/DMABuf output, and its
plugin links both `librockchip_mpp.so.1` and `librga.so.2`. The measured path
was `rtspsrc -> rtph264depay -> h264parse -> mppvideodec(640x640 RGB) ->
appsink -> RKNNLite`, followed by native C++ decode/NMS.

| Spark stream | Contract | MQTT FPS / frame-id rate | infer mean/P95 | pipeline mean/P95 | CPU / RSS / NPU snapshot | Functional output |
|---|---:|---:|---:|---:|---|---|
| `fall-e2e-low` | 200/200 | 14.90 / 14.85 | 57.93 / 65.43 ms | 58.42 / 65.74 ms | 56.57% / 192.5 MiB / Core0/1 52%/10% | no person, 0 events |
| `fall-person` | 500/500 | 12.55 / 12.55 | 64.37 / 91.12 ms | 66.70 / 96.07 ms | 35.92% / 159.9 MiB / Core0/1 33%/0% | 325 visible, 459 tracked, 8 events; normal/suspected/fallen |

Every payload reported `source_backend=gstreamer_mpp` and
`postprocess_backend=cpp`; logs confirmed RK3576 and `rga_api 1.10.1_[4]`.
During the low snapshot, a parallel `fall-trace-rk3576` extraction container
was running and was not stopped, so that snapshot is contention-affected. It
had exited before the positive snapshot. The prior OpenCV/NumPy low run was
4.88 MQTT FPS with 94.05/96.16 ms infer/pipeline P95; the prior positive run
was 4.39 MQTT FPS with 100.0/103.64 ms P95. This is a same-stream engineering
comparison, not an accuracy claim; event counts from a loop are not accuracy.

Raw evidence: `results/rk3576-mpp-cpp-low-20260813.ndjson`,
`rk3576-mpp-cpp-low-summary-20260813.json`,
`rk3576-mpp-cpp-fall-person-20260813.ndjson`, and
`rk3576-mpp-cpp-fall-person-summary-20260813.json`.

The optimized RC2 was pulled back on RK3576 with RepoDigest
`sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8`
and inspect size 258,898,465 bytes. `app.py --validate` and the complete runtime
factory/native-postprocess smoke passed without starting or stopping business
services. The external pose model is not baked and remains license HOLD.

## Aligned S INT8 RTSP capacity (2026-09-05)

The frozen 640x640 H.264@15 fixture was measured with the MQTT wall-clock
output contract and competing applications stopped. The Python control path
uses MPP NV12 followed by CPU color conversion/resize. One route passed all
three 120-second repetitions: 14.9917, 14.8333 and 15.0083 FPS; inference
means were 43.315, 43.489 and 43.809 ms, inference P95 was 49.783, 50.564
and 50.401 ms, and pipeline P95 was 50.137, 50.876 and 50.720 ms. Two routes
measured 12.8333 and 12.8083 FPS in the first formal repetition, below the
14.5 FPS/route SLA. Verified starting capacity: **1 x 15 FPS**.

`inference` excludes video preprocessing. `pipeline` starts after source read
returns and includes inference, pose decode/NMS, tracking, temporal/fall state
and payload construction; it excludes source read/preprocessing and MQTT send.
Raw evidence is under
`/home/harvest/project/edgefallkit-work/rk-aligned-20260903/`; reviewed values
and artifact hashes are frozen in
[`../../evaluation/reports/rk-s-int8-aligned-20260905.md`](../../evaluation/reports/rk-s-int8-aligned-20260905.md).
