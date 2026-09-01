# Evaluation material inventory

Last verified: 2026-09-02 (Asia/Shanghai)

Large evaluation material is not stored in this source project. Orin Nano is
the evaluated working/debug machine; Spark is the durable backup.

## Storage locations

| Material | Orin Nano working copy | Spark backup |
|---|---|---|
| GMDCSA-24, RealBiomFall, manifests and prepared links | `/tmp/fall-eval-data/` | `/home/harvest/datasets/fall-detection/evaluation/` |
| YOLO11s/m 15 FPS temporal traces | `/tmp/fall-temporal-traces/` | `/home/harvest/datasets/fall-detection/traces/` |
| YOLO11s/m ONNX and SM87 TRT 10.3 engines | `/tmp/jetson-fall/` | `/home/harvest/datasets/fall-detection/models/` |
| Jetson ARM64 runtime image | local `fall-detection:jetson-slim` build source on Orin | `sensecraft-missionpack.seeed.cn/solution/fall-detection-jetson:0.1.0-rc3` |
| RK3576/RK3588 YOLO11n-Pose FP16 RKNN artifacts | WSL2 `/home/harve/fall-rknn/` and this project's ignored platform `models/` directories | `/home/harvest/datasets/fall-detection/models/rknn/{rk3576,rk3588}/` (verified 2026-08-13) |
| RK LAN E2E raw MQTT/summary | Mac `/private/tmp/rk*-spark-*` | `/home/harvest/datasets/fall-detection/evaluation/rk-e2e/20260813/` |
| RK3576 GMDCSA working copy/traces | `cat-remote:/home/cat/fall-detection-data/{gmdcsa24,traces/rk3576}` | traces `/home/harvest/datasets/fall-detection/traces/rk3576/`; profile/reports `/home/harvest/datasets/fall-detection/rk-training/rk3576/` |
| RK3588 GMDCSA working copy/traces | `radxa:/home/radxa/fall-detection-data/{gmdcsa24,traces/rk3588}` | traces `/home/harvest/datasets/fall-detection/traces/rk3588/`; profile/reports `/home/harvest/datasets/fall-detection/rk-training/rk3588/` |
| RK shared ARM64 runtime image | local source `fall-detection-rknn:2.5.0` on RK3588 | `sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc2` |
| Hailo-8 YOLOv8s-Pose 160 traces | Pi temporary extraction tree `/tmp/hailo-gmdcsa-traces/` | `/home/harvest/datasets/fall-detection/traces/hailo8-yolov8s-pose/` |
| Hailo-8 frozen temporal profile/reports | local small copies under `evaluation/reports/` | `/home/harvest/datasets/fall-detection/profiles/hailo8-yolov8s-pose-v1/` |
| Hailo-8 ARM64 runtime image | `fall-detection-rpi-hailo:4.21` build source on Pi | `sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc1` |
| reCamera Pro RV1126B GMDCSA traces | Mac archive `/tmp/recamera-pro-rv1126b-gmdcsa-traces-20260813.tar.gz` | `/home/harvest/datasets/fall-detection/traces/recamera-pro/20260813/recamera-pro-rv1126b-gmdcsa-traces-20260813.tar.gz` |
| reCamera Pro native experiment/profile reports | canonical Pro source under `apps/fall-detection/{models,evaluation}/` | `/home/harvest/datasets/fall-detection/profiles/recamera-pro-rv1126b-yolo11n-pose-v1/` |

## Verified SG2002 live traces

The following unlabelled SG2002 MQTT traces were captured on reCamera OS 0.2.2
and were copied from the Mac temporary directory to Spark on 2026-08-13. SHA256
was calculated independently on both machines after transfer.

| Material | Mac source | Spark backup | Bytes | Verified SHA256 |
|---|---|---|---:|---|
| Before valid-observation gate | `/private/tmp/recamera-fall-0.2.2-live.ndjson` | `/home/harvest/datasets/fall-detection/traces/recamera-sg2002/0.2.2/recamera-fall-0.2.2-live.ndjson` | 774012 | `045f22fa27a4a17bf5832ca0817832f8b6ecace2029e7aea526e45ea69b417d9` |
| After valid-observation gate | `/private/tmp/recamera-fall-0.2.2-visible-gate.ndjson` | `/home/harvest/datasets/fall-detection/traces/recamera-sg2002/0.2.2/recamera-fall-0.2.2-visible-gate.ndjson` | 761223 | `f3fb721c48fd103c1d6b37e3f21c20921fd790dae5246fed4cddcaa3d6ab81a4` |

These are operational regression traces, not labelled accuracy material. The
Mac source files were intentionally retained after backup.

## Verified reCamera Pro evaluation assets

The RV1126B device extracted all 160 GMDCSA clips with zero failures. The full
trace archive was copied to Spark and independently SHA256-verified on both
machines:

- trace archive: `886dcde759c9a949073beecc1d06c0785f6e02b77bb3e1bf256192311743bf23`
- native experiment profile: `4473e2af0fe5e47b03306bf9a866103a21dbdfceb794933034173bed53de6826`
- frozen S4 report: `79046f3938213f797cd3f340914e94d17e55454a45bc083c4dc01bc177f4572c`
- same-trace production-fallback report: `efc2e147c34ea9bf89f219df76b8275a6da6053d0e691182832650d11d2e8811`
- pose-coverage report: `a4ac29d118379e86f15ebda240008c03f42c7bdb21a6151f5ba1eb8b87712616`

The native experiment is retained for provenance but is not the deployed
default: its clean-S4 Accuracy/F1 were 70.37%/69.23%, versus
81.48%/81.48% for the existing fallback on the same Pro traces.

The following digests describe the earlier core backup snapshot and therefore
exclude the four per-file additions verified above. Keep them as provenance for
that snapshot; use the per-file SHA256 inventory for the new SG2002 and RKNN
artifacts.

| Earlier backup snapshot | Files | Directories | Links | Verified tree digest |
|---|---:|---:|---:|---|
| `evaluation/` | 274 | 20 | 34 | `c7eac0364dd7d1d063f76b09cf615cb15f9efcbc329f5b0ae99837915513a44c` |
| `traces/` | 320 | 27 | 0 | `664cac68c32a45ef6562389b36296863f1506063855647e8421759f9d9b6197e` |
| `models/` | 4 | 1 | 0 | `860fcb3379ab756d05996a32ee58f64118ba4cdbeba6e7b08039f47e1d3e7fde` |

## Public download sources

### GMDCSA-24 v2.1

- Upstream repository/tag:
  `https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/tree/v2.1`
- Reproducible downloader:
  `platforms/recamera-sg2002/tools/download_gmdcsa24.sh`
- Expected result: 160 MP4 files plus `ADL.csv` and `Fall.csv` for Subjects
  1–4.

```bash
platforms/recamera-sg2002/tools/download_gmdcsa24.sh /data/gmdcsa24
```

The downloader currently uses `ghproxy.net` for the GitHub v2.1 files and is
resumable. If that mirror changes, keep the upstream tag fixed.

### RealBiomFall

- DOI: `https://doi.org/10.5281/zenodo.11620083`
- Record: `https://zenodo.org/records/11620083`
- Downloader: `platforms/recamera-sg2002/tools/download_realbiomfall.sh`
- Manifest builder: `platforms/recamera-sg2002/tools/prepare_realbiomfall.py`
- Upstream archive MD5 values are embedded in the downloader.

```bash
platforms/recamera-sg2002/tools/download_realbiomfall.sh /data/realbiomfall
python3 platforms/recamera-sg2002/tools/prepare_realbiomfall.py \
  /data/realbiomfall --output /data/realbiomfall-manifest.json
```

The manifest builder uses a restricted unpickler and does not execute classes
from the downloaded pickle files.

## Rebuilding traces and engines

The published Jetson RC3 runtime was pulled back on Orin NX with RepoDigest
`sha256:a7253a5d8689607e722f9ee42c455665441ae4c553de4275605cca59ed0e01db`.
Registry-pulled inspect reports 138,045,196 bytes, Linux/ARM64. Embedded
runtime, app, config and schema hashes matched the release manifest; `app.py`
compiled and the library loaded with host TensorRT/CUDA. The prior RC2 image
was pulled back on Orin Nano and Orin NX. The image contains no ONNX or engine; the external
artifact remains subject to its upstream model provenance and is prepared with
`platforms/jetson/tools/prepare_model.sh`.

- Build TensorRT engine: `platforms/jetson/tools/build_engine.sh`
- Extract GMDCSA traces: `platforms/jetson/tools/extract_gmdcsa_traces.py`
- Train/freeze temporal profile: `platforms/jetson/tools/train_temporal_for_pose.py`
- Run video evaluation: `platforms/jetson/tools/evaluate_videos.py`
- RK resume-safe extraction: `platforms/rknn/tools/extract_gmdcsa_traces.py`
- RK independent freeze/test: `platforms/rknn/tools/train_temporal_profile.py` and
  `platforms/rknn/tools/evaluate_frozen_subject4.py`

The GMDCSA copies on Spark, RK3576 and RK3588 were independently fingerprinted
over 160 relative-path MP4 SHA256 values; all three aggregate to
`007219878d9782391ad455590a76268866687c72c2aa2ead16ad39060f88ba9d`
and contain 8 CSV files. Both boards completed all 160 traces with zero
failures. Subject 4 was unlocked only after each independent profile was
frozen. The final manifests, profiles and development/frozen-test reports are
backed up under Spark
`/home/harvest/datasets/fall-detection/rk-training/{rk3576,rk3588}/`.
The complete trace directories were transferred with Fleet stream MD5
`e2c54d32fe545dd3bc091ce17ea6d157` (RK3576) and
`c1b9c333da1a7f07636a7ec0d63e1deb` (RK3588); their final manifest SHA256
values are `d8e07a029e259d73a11b65a064467d3e6def3f650a3a639ea134aedfb1aeefad`
and `8e34c531e481950cc88ab879167c54bd6445e488d4db75ec07b8c72fa527d2a0`.
The device extractor reuses one RKNN context across clips, decodes offline
without real-time throttling, resets only per-clip tracker state, atomically
renames each JSONL and validates trace checksums on `--resume`. Subject 4 is
guarded by the explicit post-freeze `--allow-holdout` flag.

The Hailo trace backup contains 160 files (13 MB). Its independently verified
tree digest is
`0cbd4f6f50c3a9907ccc67010d895f96c5ed71fb33c916f5d534e17bf79816ff`
and transfer MD5 is `29b0cc85ee9362058f148f8b7bcc9dec`. The frozen profile
header SHA256 is
`dec7237a1204cd2d9d54aa6810ca941ef82b83a18e65bb06a4eb7893bb55faf9`;
its Spark directory also contains separate S1-3 and S4 trace manifests, the
pre-test freeze-manifest checksum, development/frozen reports and coverage.
The final Spark `SHA256SUMS` itself is
`268a84f517cd13ee8010e7d8e159f87e3bd42db646e763c187046b93396c0091`.

The published Hailo runtime was pulled back and verified with registry
RepoDigest
`sha256:1243fb26141a43f67434e1954e6f7ff227e27b8de8aabe2a50a0e3cb48f823a0`;
device-side inspect reports 144,679,212 bytes. It intentionally excludes the
HEF. Obtain `yolov8s_pose.hef` from Hailo's official fixed URL
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef`
and require SHA256
`e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d`.
The project does not re-host or assert redistribution rights for that HEF.

The published RK runtime was independently pulled on `cat-remote` and `radxa`.
Both resolve RepoDigest
`sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8`;
registry-pulled inspect reports 258,898,465 bytes. The container contains no
RKNN pose model. Its external RK3576/RK3588 model files derive from
Ultralytics YOLO11n-Pose reference weights and are on **license HOLD**: do not
publish them until AGPL-3.0 suitability or a commercial license is documented.

TensorRT engines are not portable artifacts. Rebuild them on the deployment
device when GPU architecture, JetPack/CUDA, or TensorRT version differs from
SM87 / TensorRT 10.3.

RKNN artifacts are target-specific too. The 2026-08-13 builds use the fixed
raw-head ONNX at `recamera/recamera_pro/models/rawhead/` and RKNN Toolkit 2.3.2;
rebuild with `platforms/rknn/tools/convert_pose_rknn.py`. Checksums:

- RK3576: `659519ae8179749925c3f15d978b760f6040a00ae01666c9050036077bac8bbd`
- RK3588: `22f00270870b25dc013e4e8e39aed98f82bcabb69b9272302280a6b8b8f48d5c`

They are in each platform's ignored `models/` directory and on the WSL2
conversion host. The following Spark copies were transferred and independently
SHA256-verified on 2026-08-13; the local source files were retained:

| Target | Spark backup | Bytes | Verified SHA256 |
|---|---|---:|---|
| RK3576 | `/home/harvest/datasets/fall-detection/models/rknn/rk3576/yolo11n_pose_rawhead_fp16.rk3576.rknn` | 10532939 | `659519ae8179749925c3f15d978b760f6040a00ae01666c9050036077bac8bbd` |
| RK3588 | `/home/harvest/datasets/fall-detection/models/rknn/rk3588/yolo11n_pose_rawhead_fp16.rk3588.rknn` | 7647051 | `22f00270870b25dc013e4e8e39aed98f82bcabb69b9272302280a6b8b8f48d5c` |

## Restoring from Spark

Use Fleet rather than direct SSH inventory parsing:

```bash
~/.rpty/bin/fleet pull spark \
  /home/harvest/datasets/fall-detection/evaluation /desired/local/path
```

For another edge device, prefer a device-to-device Fleet transfer and verify
the resulting file tree. Do not treat `/tmp` on Orin as durable storage.
