# RKNN pose-frontend-specific temporal training

Each board must extract its own RKNN pose traces. Do not reuse Jetson metrics or
read Subject 4 while choosing configuration.

On each board, after the dataset is available at `/data/gmdcsa24`:

```bash
python3 tools/extract_gmdcsa_traces.py \
  --platform rk3576 \
  --model ../rk3576/models/yolo11n_pose_rawhead_fp16.rk3576.rknn \
  --dataset /data/gmdcsa24 \
  --output /data/traces/rk3576 \
  --subjects 1,2,3 --resume
```

The extractor writes each JSONL through `.part` + atomic rename and updates
`extraction-manifest.json` after every clip. `--resume` verifies the saved trace
SHA256 before skipping it. One process loads one RKNN context and reuses it
across all clips; video is decoded continuously offline without `-re` or wall
clock throttling, while tracker state is recreated at every clip boundary.
Change platform/model paths for RK3588.

The completed 2026-08-13 run extracted 123/123 development clips and 37/37
holdout clips independently on each board, with zero failures. S1-3 extraction
measured about 9.7 clips/minute on RK3576 and 11.3 clips/minute on RK3588. The
12-candidate CPU-side MLP search plus final refit took about 41 minutes for
RK3588 and 37 minutes for RK3576 while running concurrently. The manifests,
not an estimate, remain the source of truth.

Freeze the configuration on a development host. Torch is not used; scikit-learn
is training-only and never enters the runtime image:

```bash
uv run --with numpy --with scikit-learn \
  python platforms/rknn/tools/train_temporal_profile.py \
  --platform rk3576 --traces /data/traces/rk3576 \
  --dataset /data/gmdcsa24 \
  --output platforms/rk3576/models/temporal-rk3576.npz \
  --development-report evaluation/reports/temporal-rk3576-development.json
```

Only after that report exists may Subject 4 be extracted:

```bash
python3 tools/extract_gmdcsa_traces.py \
  --platform rk3576 \
  --model ../rk3576/models/yolo11n_pose_rawhead_fp16.rk3576.rknn \
  --dataset /data/gmdcsa24 --output /data/traces/rk3576 \
  --subjects 4 --allow-holdout --resume

uv run --with numpy --with scikit-learn \
  python platforms/rknn/tools/evaluate_frozen_subject4.py \
  --platform rk3576 --traces /data/traces/rk3576 \
  --dataset /data/gmdcsa24 \
  --model platforms/rk3576/models/temporal-rk3576.npz \
  --development-report evaluation/reports/temporal-rk3576-development.json \
  --report evaluation/reports/temporal-rk3576-frozen-test.json
```

Repeat independently for RK3588. The deployed `TemporalMLP` consumes only the
resulting NPZ through NumPy; no training framework is a runtime dependency.

Frozen clean S4 results (10 prior smoke clips excluded, n=27 per platform):
both profiles produced TP=12, FN=0, TN=12 and FP=3, for 88.89% accuracy, 100%
recall and 88.89% F1. Mean detection latency was 1.492 s on RK3576 and 1.525 s
on RK3588. See `evaluation/reports/temporal-rk*-{development,frozen-test}.json`
for the immutable protocol, profile checksum, confusion matrix and exact
misclassified clip list.
