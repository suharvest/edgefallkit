# RK3588 measured status

Fleet `radxa` was inspected on 2026-08-13: Radxa ROCK 5T / RK3588, Debian 12,
kernel `6.1.84-8-rk2410`, 8 CPU cores and 15 GiB RAM. Host packages are
`rknpu2-rk3588 2.3.0-1` and `python3-rknnlite2 2.3.0-1`; Python is 3.11.2,
NPU driver 0.9.8 and the runtime library reports build 2.3.2. The NPU governor
is `rknpu_ondemand` with 300-1000 MHz range.

Target model: YOLO11n-Pose raw nine-head FP16, fixed 1x640x640 RGB, RKNN
Toolkit 2.3.2, size 7,647,051 bytes, SHA256
`22f00270870b25dc013e4e8e39aed98f82bcabb69b9272302280a6b8b8f48d5c`.

| Metric | Result |
|---|---|
| Native RKNN load/inference | PASS, exit 0, no `failed submit` |
| 1/2/3-context throughput | 19.25 / 38.13 / 51.40 FPS (blank 640, 100 iterations) |
| Inference mean / P95 | 51.41/58.92; 51.43/56.71; 56.68/88.01 ms |
| Pipeline mean / P95 | 51.77/59.28; 51.89/62.36; 57.29/92.40 ms |
| RSS max 1/2/3 context | 189.9 / 217.6 / 294.8 MiB |
| CPU / NPU | contention affected; NPU sampled `100@1GHz` |
| Frozen temporal-gate accuracy | 88.89% on clean GMDCSA S4 (27 clips) |

Existing `openvoicestream`, LLM and agent services were deliberately not
stopped and the NPU was not reset. Any completed benchmark must therefore be
labelled contention-affected, not an exclusive maximum. The device mirror
bootstrap was repaired so new non-login shells use
`HF_ENDPOINT=https://hf-mirror.com`; no model was downloaded from
`huggingface.co`.

These are blank-frame native runtime and decode-overhead measurements (zero
detections), not live RTSP end-to-end or accuracy data. All three runs exited
normally. Runtime logs confirmed target `rk3588`, static shape, toolkit 2.3.2,
librknnrt 2.3.2 and driver 0.9.8. Because the existing workloads remained
active, use these as contention-affected capacity evidence, not an exclusive
board maximum.

Raw JSON: `results/rk3588-native-1ctx-100.json`,
`rk3588-native-2ctx-100.json`, `rk3588-native-3ctx-100.json`, and consolidated
`rk3588-benchmark-evidence-20260813.json` (SHA256
`3fc7fadde4cfbd93e30d8954023546bd097d6643cf55acba2814c4b79cbc1f62`).

The shared ARM64 runtime image audit is recorded in `../rknn/README.md`:
The same image built on Radxa is 144,143,406 bytes unpacked and 143,349,802
bytes as streamed gzip (minor manifest metadata difference from Mac). Models and
RKNN Toolkit/Torch/CUDA are not baked.

## Spark LAN RTSP production path

The production app was run against Spark MediaMTX H.264 Baseline 640x640@15
with the existing voice/RKLLM workloads left in place and no NPU reset.

| Streams | MQTT received FPS | Source frame-id rate | infer mean/P95 | pipeline mean/P95 | CPU/RSS/NPU snapshot |
|---:|---:|---:|---:|---:|---|
| 1 | 8.56 | 14.81 | 54.36/76.00 ms | 54.76/76.45 ms | 34.73%; 169.8 MiB; Core0/1/2 47%/0%/0% |
| 3 | 5.19 + 5.41 + 5.65 | 10.64 + 10.60 + 11.00 | 72.48/136.90; 72.71/134.10; 70.78/137.60 ms | 73.21/137.86; 73.64/135.11; 71.43/138.50 ms | 101.89%; 366.4 MiB; Core0/1/2 64%/48%/29% |

QoS0 observation contained frame-id gaps, hence separate broker-received and
source-frame-id rates. Decoded-frame-to-broker-receive P95 was 232.2 ms for one
stream and 293.6-308.0 ms for three; this excludes source encode/network delay.

The looping GMDCSA Subject-4 Fall/01 stream produced 500 contract-valid
messages: 302 visible-person/pose17 messages, 398 with retained track plus
temporal classifier, and 5 stream-global fall events spanning normal,
suspected and fallen. This closes the real positive path but is not an accuracy
test. Raw evidence is backed up at
`spark:/home/harvest/datasets/fall-detection/evaluation/rk-e2e/20260813/`;
summary JSON is in `results/rk3588-spark-*.json`.

## Frozen temporal-profile accuracy

The RK3588 frontend extracted all 160 GMDCSA clips with zero failures. Subject
1-2 fit, Subject 3 selection and Subject 1-3 refit completed before Subject 4
was unlocked. The selected profile is pose features, 32 hidden units, alpha
0.01, threshold 0.80 and three consecutive positives. Subject 3 development
accuracy/F1 was 95.35%/95.45%.

The clean frozen Subject 4 test excludes the 10 earlier smoke clips: n=27,
TP=12, FN=0, TN=12, FP=3, accuracy 88.89%, recall 100%, specificity/precision
80%, F1 88.89%, zero early alerts, mean detection latency 1.525 s and median
1.10 s. False positives were ADL 06, 07 and 16. This is temporal-gate clip
accuracy from the independent RK3588 frontend, not a Jetson metric or full
deployed state-machine accuracy. Profile SHA256:
`b7213580f6f6bd1d1cafbfdecb84306c17b3ecfbe37b907a8b41215310ed4397`.

## GStreamer MPP/RGA + C++ postprocess E2E (2026-08-13)

Read-only inventory on `radxa` matched RK3576: `rockchipmpp:mppvideodec`
(`gst-rockchip` 1.14.4), no standalone RGA element, parsed AU-aligned H.264/H.265
input, RGB/BGR/NV12/DMABuf output, and plugin linkage to both MPP and RGA. The
final path uses integrated resize/RGB conversion to `appsink`, RKNNLite, then
the native C++ decode/NMS extension.

| Spark stream | Contract | MQTT FPS / frame-id rate | infer mean/P95 | pipeline mean/P95 | CPU / RSS / NPU snapshot | Functional output |
|---|---:|---:|---:|---:|---|---|
| `fall-e2e-low` | 200/200 | 14.67 / 14.24 | 53.65 / 83.00 ms | 54.12 / 83.20 ms | 27.78% / 190.8 MiB / Core0 47% | no person, 0 events |
| `fall-person` | 500/500 | 12.96 / 12.93 | 61.52 / 105.17 ms | 63.00 / 107.13 ms | 289.59% / 164.5 MiB / NPU 39% | 305 visible, 430 tracked, 8 events; normal/suspected/fallen |

All 700 payloads reported `gstreamer_mpp` and `cpp`. Existing
`openvoicestream`, LLM and agent workloads remained running; the positive
stream CPU snapshot and latency tail are contention/content affected, not an
exclusive maximum. The prior OpenCV/NumPy low run was 8.56 broker FPS and
76.0/76.45 ms infer/pipeline P95; the new run reaches 14.67 broker FPS, while
mean compute time remains similar. Positive event counts are functional only.

Raw evidence: `results/rk3588-mpp-cpp-low-raw-20260813.ndjson`,
`rk3588-mpp-cpp-low-summary-20260813.json`,
`rk3588-mpp-cpp-fall-person-raw-20260813.ndjson`, and
`rk3588-mpp-cpp-fall-person-summary-20260813.json`.

The published RC1 was pulled back on RK3588 with RepoDigest
`sha256:e13c0d3bac963ac78b2d067deee6880aa3058e65f41c700af1b1718129685dc7`
and inspect size 258,898,465 bytes. `app.py --validate` and the complete runtime
factory/native-postprocess smoke passed without starting or stopping business
services. The external pose model is not baked and remains license HOLD.
