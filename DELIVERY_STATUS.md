# Delivery status

Last audited: 2026-09-05 (Asia/Shanghai)

This file distinguishes deployable engineering baselines from the remaining
external-validation work. Jetson, RK3576, RK3588, Hailo-8, SG2002 and Pro now have
frozen per-frontend temporal results; quote each result only with its documented
frontend, split and temporal-gate/deployed-state-machine scope.

## Platform readiness

| Platform | Runtime and configuration | Deployment mechanism | Required external artifact | Current verification |
|---|---|---|---|---|
| Jetson Orin Nano/NX | Complete | slim Docker Compose; published RC3 | target-built SM87/TRT 10.3 pose engine from the explicit-license preparation helper | RC3 registry pull-back on Orin NX, embedded hash and app/config smoke, host TensorRT/CUDA library-load smoke; prior RC2 multi-device and MQTT/RTSP checks retained |
| RK3576 | Complete | Docker Compose; published RC7 | RK3576 `.rknn` (license HOLD) and native temporal `.npz` | RC7 candidate 3-route/20-second MQTT smoke (900/900 messages observed), MPP/RGA/native bridge and dependency/OOM checks passed |
| RK3588 | Complete | Docker Compose; published RC7 | RK3588 `.rknn` (license HOLD) and native temporal `.npz` | RC7 candidate 5-route/20-second MQTT smoke (1496/1500 messages observed), MPP/RGA/native bridge and dependency/OOM checks passed |
| Raspberry Pi 5 + Hailo-8 | Complete | Docker Compose; published RC1 | Hailo-8 `yolov8s_pose.hef` from official URL | registry pull-back, metadata smoke, single/dual RTSP, positive fall path, and MQTT v1 contract verified |
| reCamera SG2002 | Complete in canonical repository | appMgr `.deb` | packaged CVI model | OS 0.2.2 live deployment, multi-person MQTT, and historical frozen accuracy verified |
| reCamera Pro | Complete in canonical repository | signed Pro app package | packaged RV1126B RKNN model and production fallback profile | firmware V1.0.4 appMgr install/signature verification, live camera/NPU/WebSocket, 60 s performance and strict native/fallback S4 comparison verified |

Jetson production uses batch 1 with one execution context and CUDA stream per
input. Static batch 4 remains a measured throughput experiment and is not
enabled by the current application ABI.

## Before deploying

Jetson's published ARM64 image is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-jetson:0.1.0-rc3`
with immutable RepoDigest
`sha256:a7253a5d8689607e722f9ee42c455665441ae4c553de4275605cca59ed0e01db`.
Orin NX pulled and validated this digest; the prior RC2 digest was validated
on Nano, NX and AGX. The image contains no ONNX or
TensorRT engine; `deploy.sh jetson` prepares the engine on the target and the
slim Compose supports the `FALL_DETECTION_IMAGE` override.

Hailo's published ARM64 image is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc1`
with immutable RepoDigest
`sha256:1243fb26141a43f67434e1954e6f7ff227e27b8de8aabe2a50a0e3cb48f823a0`.
Compose uses this tag by default and supports the `FALL_HAILO_IMAGE` override.

The shared Rockchip ARM64 release is
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc7`
with immutable RepoDigest
`sha256:8c79172138a0f510e26bd0f219f82b6a57ab98ff30f6828d96786e5131dfeae5`.
The RC7 candidate passed board-side smoke on both boards; the historical RC2
digest remains recorded in the aligned performance evidence.

1. Copy or build the platform-specific accelerator artifact described in the
   platform README. TensorRT engines, RKNN files, and HEF files are not
   interchangeable.
2. Edit the platform config with the RTSP URL, stable `stream_id`, and MQTT
   broker credentials. Do not leave the example `192.168.1.10` URL in place.
3. Validate Compose with `docker compose config --quiet`.
4. Start the service and validate captured MQTT with
   `contracts/validate_payload.py`.
5. For reCamera and Pro, use their native package/appMgr release procedure;
   they intentionally do not have Docker Compose files.

## Remaining, non-blocking validation work

- Re-run the Pro native temporal candidate after the independently deployed RGA
  preprocessing update; the current production temporal profile remains the
  stronger validated fallback.
- Evaluate the RK3576, RK3588 and Hailo-8 frozen profiles on RealBiomFall; no
  external-set result is currently claimed for these three frontends.

The split is fixed: Subjects 1–2 fit, Subject 3 selects hyperparameters,
Subjects 1–3 refit, and Subject 4 is read only by the frozen test command.
RK3576, RK3588 and Hailo-8 completed this protocol independently on 2026-08-13.

The shared RKNN 2.5.0 RC2 ARM64 image is 257,794,493 bytes by registry-pulled
device-side inspect. The same immutable digest and code hashes passed pull-back
and configuration validation on RK3576 and RK3588. Compose mounts the
verified host Rockchip MPP/RGA and H.264/H.265 parser ABI. The runtime contains
no compiler, pybind11 headers/module/cache or model; only the stripped native
postprocess `.so` crosses from the builder. Local tests are 12/12, including
real C++/NumPy parity for NCHW/NHWC plus backend/config/Compose fallback tests.
The image contains no model. The two
YOLO11n-Pose RKNN artifacts remain release HOLD until Ultralytics AGPL-3.0
suitability or a commercial model license is documented.

## Known handoff requirements

- This repository tracks the shared runtime, contract, deployment entry point,
  and evidence ledger. The linked reCamera sources remain versioned in their
  canonical repositories.
- `platforms/recamera-sg2002` and `platforms/recamera-pro/*` are relative
  symlinks to sibling canonical repositories. Preserve the documented sibling
  layout and commit those canonical changes independently.
- Large datasets, Jetson artifacts, SG2002 operational traces, and target RKNN
  artifacts are durably backed up on Spark; see `assets/ASSET_LOCATIONS.md`.
- Generated `.DS_Store`, `__pycache__`, build directories, model binaries and
  packages are ignored and must not be included in source archives.

## Evidence

- Cross-platform results: `evaluation/RESULTS.md`
- Frozen evaluation protocol: `evaluation/EVALUATION.md`
- Raw reports and hashes: `evaluation/reports/`
- MQTT schema and semantics: `contracts/`
- Dataset/model/trace locations: `assets/ASSET_LOCATIONS.md`
- Jetson ONNX restore/export: `platforms/jetson/models/README.md`
- Hailo fixed HEF download: `platforms/rpi-hailo/scripts/fetch_model.sh`
- Machine-readable Jetson RC3 manifest: `release/jetson-0.1.0-rc3.json`
- Machine-readable RKNN RC7 manifest: `release/rknn-0.1.0-rc7.json`
- Historical multi-platform RC2 manifest: `release/0.1.0-rc2.json`
- Historical RC1 manifest: `release/0.1.0-rc1.json`
