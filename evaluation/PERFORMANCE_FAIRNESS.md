# Cross-platform performance fairness audit

Audited on 2026-09-01. The current 15 FPS RTSP rows prove that a platform can
sustain the source stream; they are not a throughput ranking. Models, output
heads, quantization and timing boundaries differ, so `pipeline_ms` must not be
compared without its platform-specific scope.

## What is comparable

- Native per-frontend GMDCSA clean S4 temporal-gate results are comparable when
  both the frontend and independently trained profile are named.
- Do not place a temporal-gate score and a deployed state-machine alert score in
  one unlabeled ranking column. Publish both scopes when both are available.
- Same-device TensorRT context/batch tests and same-device HailoRT tests are
  valid capacity A/B measurements.
- Cross-device accelerator ranking requires the same model size, input shape,
  batch, context count and precision. Jetson INT8 rows must use an explicit
  calibrator and real images; the earlier uncalibrated `trtexec --int8` rows are
  retained only as invalid historical evidence.
- A fixed 640-square, 15 FPS stream is a deployment SLA check only.

## Known measurement and implementation risks

- Jetson `inference_time_ms` includes host/device copies, CUDA preprocessing,
  TensorRT, output copy and CPU parsing; pure TensorRT is reported separately.
- RK `pipeline_ms` starts after appsink and excludes decode, tracking and MQTT.
  Its MPP production path stretches non-square input to 640 square while the
  accuracy extractor letterboxes; this must be aligned before claiming the S4
  value for non-square production streams.
- Hailo S and M application probes have different boundaries. S reports
  `pre_hailonet_to_hailonet_src`; M reports
  `appsink_enqueue_to_hailort_completion`. Neither is interchangeable with
  HailoRT hardware latency, and the two application probes must not be ranked
  against each other as if they shared a start point.
- Pro `pipeline_ms` excludes the firmware frame broker and result transport.
  Reported memory is whole-system used memory, not process RSS.
- Several capacity tests retained existing workloads; these are coexistence
  measurements, not idle peak results.

## Highest-value optimization by platform

| Platform | Priority optimization | Evidence / expected bound |
|---|---|---|
| reCamera Pro | RGA NV12→model-size resize plus letterbox before RGB/Python | preprocessing ~40.36 ms vs RKNN ~35.90 ms; reducing preprocessing below 29.3 ms is enough to sustain 15 FPS, and single-digit preprocessing would put compute near 20–23 FPS |
| Jetson | process-wide shared engine plus per-stream contexts; bounded 2–4 ms microbatch | batch-4 measured 179.94 img/s Nano and 105.96 img/s NX, but current Python ABI is batch-1 |
| RK3576/3588 | preserve aspect ratio in MPP/RGA path; remove appsink RGB copy; explicit context/core policy | multi-context already gives 1.67× on RK3576 and 2.67× on RK3588; C++ postprocess is not the main bottleneck |
| Hailo-8 | correct per-buffer latency pairing, then hardware decode/scale/color and compact C++ result element feeding Python | Hailo core is far above source rate; CPU decode/color/scale is the scaling bottleneck |

Python should remain the control plane for tracking, the tiny MLP, FSM, config
and MQTT. Do not move full RGB frames through multiprocessing. Release the GIL
inside small native hot-path extensions and cross the boundary only with compact
detections/keypoints.

## Required unified protocol

Use two leaderboards: native-best deployable systems and an ISO-frontend
hardware comparison. For each case use the same Spark LAN H.264 source, both
640-square and 1280×720 letterbox cases, blank/one-person/four-person scenes,
30 s warm-up plus 120 s × 3 runs, and separate exclusive/coexistence runs.
Record source PTS through decode, accelerator, postprocess, FSM, serialization
and broker receipt. Report processed/input/drop rate, FPS and P50/P95/P99,
power mode, clocks, temperature, model/runtime hashes, queue wait and batch
latency. A route passes at ≥14.7 FPS, <1% drop, no contract errors/reconnects or
thermal throttling for 30 minutes.

Publish timing in separate fields:

- accelerator-only: TensorRT `GPU Compute Time` or HailoRT `Latency (hw)`;
- application inference: the runtime's preprocess, copies, accelerator,
  output copy and parser interval, when that exact interval is instrumented;
- pipeline: the named start/end markers for that platform;
- output cadence: published frames divided by elapsed time.

Missing intervals are `N/A`, not substituted with a wider or narrower metric.
Batch latency is latency for the entire batch and must not be divided by batch
size and relabeled as single-frame latency.
