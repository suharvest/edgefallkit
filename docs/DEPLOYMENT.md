# Deployment guide

The published runtime images contain the application and native acceleration
bridge, but no pose model. The root helper dispatches to the platform-specific
model preparation flow:

```bash
./deploy.sh PLATFORM --accept-upstream-license [options]
```

The acknowledgement is mandatory before an upstream model is downloaded or
converted. It does not relicense that model. Every helper supports a dry-run;
each platform also documents its offline or local-model path.

## Published runtime images

| Platform | Image | Immutable RepoDigest |
|---|---|---|
| Jetson Orin | `sensecraft-missionpack.seeed.cn/solution/fall-detection-jetson:0.1.0-rc1` | `sha256:162824bedda86eeadb1bc265b21ae14bb264ad907f68f3ea001db745e38f32ff` |
| RK3576 / RK3588 | `sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc2` | `sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8` |
| Raspberry Pi 5 + Hailo-8 | `sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc1` | `sha256:1243fb26141a43f67434e1954e6f7ff227e27b8de8aabe2a50a0e3cb48f823a0` |

For production, pin the image by digest or mirror it into a registry you
control. The Compose files expose `FALL_DETECTION_IMAGE`, `FALL_RK_IMAGE`, and
`FALL_HAILO_IMAGE` overrides.

## Jetson Orin

Run from a development machine with `uv` and Fleet configured:

```bash
./deploy.sh jetson --model yolo11s-pose --device orin-nano \
  --accept-upstream-license --deploy
```

Use `yolo11m-pose` and `orin-nx` for the measured NX profile. The helper uses an
isolated build environment to obtain/export ONNX, sends only the ONNX and
engine-builder helper to a temporary target directory, invokes host TensorRT
10.3 on SM87, pulls back the engine, writes a provenance manifest, then deploys
only the engine and manifest. A matching manifest is a cache hit. `--offline
--onnx PATH` uses an existing ONNX without a model download.

The default remote checkout is `/home/harvest/fall-detection`; override it with
`--remote-project-dir` when the checkout differs. Deployment validates that the
remote Compose, config, model directory, and configured engine filename match
before changing the service.

## RK3576 / RK3588

RKNN Toolkit conversion is x86_64-only. Run the root helper on an x86 Mac/Linux
workstation and let Fleet transfer the prepared artifact:

```bash
./deploy.sh rk3576 --device cat-remote --accept-upstream-license
./deploy.sh rk3588 --device radxa --accept-upstream-license
```

Without an existing cache, an ephemeral model-builder obtains the official
YOLO11n-Pose weights, exports the fixed nine raw heads, and converts them with
RKNN Toolkit 2.3.2 for the selected target. Torch, Ultralytics and RKNN Toolkit
remain in the builder. The ARM runtime receives only `.rknn`, the independently
frozen temporal `.npz`, configuration, and Compose. An ARM caller must provide
an explicit x86 Fleet `--builder-host` or run the command on x86.

Use `--model-file`, `--model-url` plus `--model-sha256`, `--offline`, `--no-up`,
or `--dry-run` as documented by `platforms/rknn/deploy.sh --help`.

## Raspberry Pi 5 + Hailo-8

On the Pi checkout:

```bash
STREAMS='lobby|rtsp://camera/live' \
  ./deploy.sh hailo --accept-upstream-license
```

The helper downloads the official Hailo Model Zoo v2.15 Hailo-8
`yolov8s_pose.hef`, verifies its fixed SHA256, and atomically installs it into
the external model directory before pulling and starting the runtime. A valid
cache is reused. `--hef PATH` installs a local verified HEF; `--offline` forbids
both model and image network access.

## Configure and verify

Before accepting any deployment:

1. Configure RTSP URLs, stable stream IDs, MQTT broker credentials, and TLS.
2. Run the platform helper with `--dry-run`.
3. Validate Compose with `docker compose config --quiet`.
4. Capture MQTT and validate it with `contracts/validate_payload.py`.
5. Run a person-positive clip; an empty stream only checks the no-detection
   branch.
6. Perform a long-running reconnect and broker-restart test for production.

reCamera SG2002 and reCamera Pro use their native appMgr/package workflows and
are intentionally not handled by this Docker dispatcher.
