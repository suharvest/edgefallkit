# Hailo-native temporal profile

Status: completed and integrated on 2026-08-13. The deployed Hailo runtime uses
the frozen profile trained from poses emitted by the Hailo-8
`yolov8s_pose.hef` decoder; it no longer reuses the YOLO11s profile.

## Frozen protocol

- GMDCSA Subjects 1-2 fit MLP weights.
- Subject 3 selects the feature mask, hidden width, regularization, probability
  threshold, and consecutive-confirmation count.
- After those choices are frozen, Subjects 1-3 refit the final weights.
- Subject 4 is read exactly once for the final test. The ten historical smoke
  clips listed by the shared training code remain excluded from the 27-clip
  clean test.
- Decode and sample at 15 FPS. Restart the Hailo app between clips so tracker,
  temporal window, event IDs, and frame counters cannot leak across clips.
- Keep raw MQTT captures and generated trace JSONL on Spark under
  `/home/harvest/datasets/fall-detection/traces/hailo8-yolov8s-pose/`.

The Spark backup contains 160 GMDCSA clips totaling 1,289.173 seconds:

| Subject | Clips | Video seconds | Role |
|---|---:|---:|---|
| 1 | 32 | 201.791 | train |
| 2 | 48 | 454.187 | train |
| 3 | 43 | 343.679 | validation/config selection |
| 4 | 37 | 289.515 | frozen final test |

The authorized extraction completed all 160 clips in 307 seconds with zero
failures. `mcp_face_rec` was restored immediately and verified healthy; Hailo
was not reset. CPU training then ran on Spark without occupying the accelerator.

## Trace capture

Spark exposes low-latency LAN RTSP at `192.168.3.42`; Pi-to-Spark RTT measured
3.96 ms. A person-positive smoke loop is already available as:

```text
rtsp://192.168.3.42:8554/fall-person
```

It is GMDCSA Subject 4 Fall/01 and is only for validating the positive-person
decode/tracker path, not for threshold selection. Both this stream and the
640x640 control stream were ffprobe-verified from the Pi as H.264 Constrained
Baseline 640x640@15.

For dataset extraction, expose one non-looping clip at a time, capture only its
MQTT topic, and save the clip identity separately. Convert a capture with the
dependency-free tool:

```bash
python3 tools/mqtt_to_trace.py \
  --input subject-1-ADL-01.ndjson.gz \
  --stream-id subject-1-ADL-01 \
  --output traces/subject-1/ADL/01.jsonl \
  --min-frames 30
```

The converter rejects non-monotonic frame IDs, chooses a visible valid
person, converts timestamps to clip-relative milliseconds, and emits the same
`pose17`/`features` schema consumed by the shared trainer. Inspect pose coverage
per clip before training; a successful RTSP/MQTT run with zero detections is
not a usable training trace.

## Train and freeze

Training may use NumPy/scikit-learn on Spark or a workstation. Those packages
are development-only and are never part of the edge image.

```bash
uv run python ../jetson/tools/train_temporal_for_pose.py \
  --traces /home/harvest/datasets/fall-detection/traces/hailo8-yolov8s-pose \
  --dataset /home/harvest/datasets/fall-detection/evaluation/gmdcsa24 \
  --header generated/temporal_model_weights_hailo.h \
  --report evaluation/temporal-hailo8-development.json \
  --namespace jetson_fall::temporal_hailo_weights
```

Before Subject 4 is opened, record the generated header SHA256, HEF SHA256,
selected mask/hidden/alpha/threshold/consecutive values, trace-tree digest,
and development report. Then compile the generated constants into the native
C++ runtime, rebuild the slim image, and run the unchanged frozen evaluation
rules on Subject 4. Final reports must include TP/FN/TN/FP, Accuracy, Recall,
Specificity, Precision, F1, early alerts, alarm latency, pose coverage, and the
raw per-clip results.

The deployment remains native GStreamer/HailoRT plus the tiny C++ MLP. Torch,
Ultralytics, scikit-learn, and Python are not required at runtime.

## Frozen result and artifacts

The selected configuration is `all` features, hidden width 16, alpha 0.01,
threshold 0.75 and three consecutive evaluations. Subject 3 development F1 is
97.56%. The untouched Subject 4 clean temporal-gate result is TP=12, FN=0,
TN=12, FP=3: Accuracy 88.89%, Recall 100%, Specificity 80%, Precision 80%,
F1 88.89%, no early alerts, mean/median latency 1.608/1.25 seconds. Valid
pose17 coverage is 2,743/2,981 frames (92.02%).

The 175,450-byte generated header has SHA256
`dec7237a1204cd2d9d54aa6810ca941ef82b83a18e65bb06a4eb7893bb55faf9`.
Spark stores the immutable source traces and profile at:

- `/home/harvest/datasets/fall-detection/traces/hailo8-yolov8s-pose/`
- `/home/harvest/datasets/fall-detection/profiles/hailo8-yolov8s-pose-v1/`

The local evaluation copies are `temporal-hailo8-development.json`,
`temporal-hailo8-frozen-test.json`, `temporal-hailo8-freeze-manifest.json` and
`temporal-hailo8-subject4-pose-coverage.json` under `evaluation/reports`.
