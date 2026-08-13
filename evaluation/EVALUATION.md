# TensorRT 10.3 fall-detection evaluation

This is a reproducible video-level benchmark, not a medical-safety
certification. Engines were built and evaluated on an Orin Nano (SM87), using
TensorRT 10.3, CUDA 12.6, FP16, fixed 640x640 input and the frozen reCamera
v0.2 temporal weights/thresholds for the baseline table. Video is sampled at
15 FPS and tracking plus temporal state is reset before every clip. The
optimized profiles below use the same protocol with model-specific weights
selected only on development subjects.

Two results are reported deliberately. In the baseline, the gate uses the
reCamera threshold; optimized reports use their frozen model-specific profile:

- **Temporal gate**: first frame on which any track reaches its frozen learned
  gate (all selected profiles here are probability >= 0.8 for three
  evaluations). The baseline is directly comparable with the original
  reCamera v0.2 report.
- **Deployed alert**: first `fallen/recovering` result from the complete Python
  state machine, including geometry and recovery logic. This is what MQTT
  consumers actually observe.

An alert more than 0.5 seconds before the labelled onset is an early false
alert and does not count as a true positive.

## GMDCSA-24 Subject 4 clean test

The split is exactly the original 27 untouched clips: 12 falls and 15 ADL.
The ten historical pipeline-smoke clips remain excluded. No threshold was
selected or changed using these results.

| Engine / output | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | Mean latency | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reCamera INT8 YOLO11n, temporal baseline | 10 | 2 | 10 | 5 | 74.1% | 83.3% | 66.7% | 66.7% | 74.1% | 1.75 s | — |
| TRT YOLO11s FP16, temporal gate | 9 | 3 | 12 | 3 | 77.8% | 75.0% | 80.0% | 75.0% | 75.0% | 1.83 s | 94.3% |
| TRT YOLO11s FP16, deployed alert | 8 | 4 | 11 | 4 | 70.4% | 66.7% | 73.3% | 66.7% | 66.7% | 1.69 s | 94.3% |
| TRT YOLO11m FP16, temporal gate | 11 | 1 | 12 | 3 | **85.2%** | **91.7%** | 80.0% | 78.6% | **84.6%** | 1.89 s | **96.7%** |
| TRT YOLO11m FP16, deployed alert | 9 | 3 | 11 | 4 | 74.1% | 75.0% | 73.3% | 69.2% | 72.0% | 1.70 s | 96.7% |

The m pose model materially improves the learned temporal result on this
fixed-camera test. The complete deployed state machine is worse than its own
temporal gate for both models; geometry/initial-posture handling therefore
needs independent development-set tuning before it should override the gate.

## Optimized temporal profiles and state machine

For each TensorRT pose frontend, Subjects 1–2 fit the tiny MLP, Subject 3 alone
selects the pose feature mask, hidden width, regularization, probability
threshold, and consecutive-positive count, and then Subjects 1–3 refit the
frozen configuration. Subject 4 is not read until the final test. Both selected
profiles use pelvis-centred pose features, 32 hidden units, threshold 0.8, and
three consecutive evaluations.

The production state machine is now temporal-authoritative: geometry can enter
`suspected` and handle recovery, but cannot enter `fallen`; a temporal positive
on a visible track is required. A missed/stale track cannot originate an event,
which removes one YOLO11s false alarm while preserving its true positives. A
first-frame lying pose is explicitly not an event.

| Engine / optimized deployed output | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | Mean latency | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRT YOLO11s FP16 | 10 | 2 | 12 | 3 | 81.5% | 83.3% | **80.0%** | 76.9% | 80.0% | 1.47 s | 94.3% |
| TRT YOLO11m FP16 | 12 | 0 | 11 | 4 | **85.2%** | **100%** | 73.3% | **75.0%** | **85.7%** | 1.26 s | **96.7%** |

The optimized YOLO11m profile keeps 85.2% accuracy while improving the frozen
temporal baseline from 84.6% F1 / 91.7% recall to 85.7% / 100%. More
importantly, the actual deployed state machine improves from 74.1% accuracy /
72.0% F1 / 75.0% recall to 85.2% / 85.7% / 100%. This gains one ADL false
positive while removing all fall false negatives; no threshold was changed
using Subject 4. Development, single-primary frozen-test, and complete
multi-track deployment reports are stored separately so the distinction stays
auditable.

## RealBiomFall external testing subset

The upstream testing subset contains 34 fall clips and no negative ADL clips.
It can measure recall and latency, but cannot measure accuracy, specificity,
precision, F1, or a real-world false-alarm rate.

| Engine / output | TP | FN | Recall | Early alerts | Mean latency | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|
| reCamera INT8 YOLO11n, temporal baseline | 20 | 14 | 58.8% | 9 | 1.18 s | — |
| TRT YOLO11s FP16, temporal gate | 19 | 15 | **55.9%** | 6 | 0.53 s | 68.6% |
| TRT YOLO11s FP16, deployed alert | 19 | 15 | **55.9%** | 6 | 0.66 s | 68.6% |
| TRT YOLO11m FP16, temporal gate | 18 | 16 | 52.9% | 6 | 0.96 s | **70.6%** |
| TRT YOLO11m FP16, deployed alert | 17 | 17 | 50.0% | 6 | 0.95 s | 70.6% |
| Optimized TRT YOLO11m FP16, temporal gate | 21 | 13 | **61.8%** | 9 | 0.51 s | 70.6% |
| Optimized TRT YOLO11m FP16, deployed alert | 18 | 16 | 52.9% | 7 | 0.99 s | 70.6% |

The model-specific temporal profile raises the external gate recall from 52.9%
to 61.8%, but the conservative deployed state machine reaches 52.9%. External
pose coverage remains only about 69–71%, and several long-shot clips have
nearly no person detections. The next accuracy work should target
long-shot/occlusion coverage and track continuity using development data,
rather than selecting a threshold against these 34 testing clips.

## Artifacts

Clip-level trigger times, inference latency and coverage are committed under
`reports/`. `../platforms/jetson/tools/evaluate_videos.py` reproduces the reports. The source
datasets are public GMDCSA-24 v2.1 and RealBiomFall; dataset videos are not
vendored into this repository.
