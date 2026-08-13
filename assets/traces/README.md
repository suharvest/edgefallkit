# Temporal training traces

The remote `yolo11s/` and `yolo11m/` trees contain 15 FPS JSONL pose traces
extracted with the corresponding FP16 TensorRT frontend. The onset CSV files
required by training are retained alongside the dataset backup.

Subjects 1–2 train, Subject 3 selects configuration, and Subject 4 is test-only.
Never use Subject 4 to select thresholds or hyperparameters.
See [`../ASSET_LOCATIONS.md`](../ASSET_LOCATIONS.md) for Orin/Spark paths.

RK traces are platform-specific. Complete 160-clip checkpoints (Subjects 1-4,
zero failures per board) live on `cat-remote` and `radxa`; their audited
manifests are stored in each platform `results/` directory and backed up on
Spark. Resume/freeze/test commands are documented in
[`../../platforms/rknn/TEMPORAL_TRAINING.md`](../../platforms/rknn/TEMPORAL_TRAINING.md).
