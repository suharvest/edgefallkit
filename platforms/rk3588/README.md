# RK3588 deployment

Target used by Fleet: `radxa` (Radxa ROCK 5T, RK3588). The compose file maps
the host RKNN runtime and DRM/NPU devices, uses host networking for RTSP/MQTT,
and contains both service and benchmark entry points.

Place the artifacts in `models/`:

```text
models/yolo11n_pose_rawhead_fp16.rk3588.rknn
models/temporal-rk3588.npz
```

This board may already run voice/RKLLM workloads. Do not stop them or reset the
NPU implicitly; record contention in benchmark reports. Before any model
download, make the non-login `HF_ENDPOINT=https://hf-mirror.com` invariant
true. See [`../rknn/README.md`](../rknn/README.md) for commands and metrics.

One-command deployment from an x86_64 workstation:

```bash
../rknn/deploy.sh --platform rk3588 --device radxa \
  --accept-upstream-license
```

Use `--dry-run`, `--offline`, `--no-up`, `--model-file`, `--model-url`, or an
explicit x86 `--builder-host` as documented in the shared README. RKNN Toolkit
conversion is never attempted on this ARM board.

The measured host exposes `rockchipmpp:mppvideodec`, not a standalone RGA
element. `libgstrockchipmpp.so` links both `librockchip_mpp.so.1` and
`librga.so.2`; Compose mounts those host ABI libraries plus the host H.264/H.265
parser plugin and codec-parser library read-only. The preferred backend is
therefore the verified integrated MPP decode/RGA resize+RGB path to `appsink`.

Default runtime release: `sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc7`
(override with `FALL_RK_IMAGE`). Published RepoDigest:
`sha256:8c79172138a0f510e26bd0f219f82b6a57ab98ff30f6828d96786e5131dfeae5`.
The RC7 candidate passed a 5-route, 20-second MQTT smoke (1496/1500 messages observed),
with no missing shared-library dependencies, no OOM, and clean exit. The measured runs used historical rc2 digest
`sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8`.
The external YOLO11n-Pose RKNN file is on license HOLD pending documented
Ultralytics AGPL-3.0 suitability or a commercial license; it is not in the
container image.
