# RK3576 deployment

Target used by Fleet: `cat-remote` (LubanCat, RK3576). The compose file maps
the host RKNN runtime and DRM/NPU devices, uses host networking for RTSP/MQTT,
and contains both service and benchmark entry points.

Place the platform-built pose model and shared temporal model in `models/`:

```text
models/yolo11n_pose_rawhead_fp16.rk3576.rknn
models/temporal-rk3576.npz
```

Before any Hugging Face/model download, verify a non-login shell prints
`https://hf-mirror.com`. Use Fleet `bootstrap cat-remote --profile edge-mirror`
when it does not. The default image has RKNN Lite 2.3.x; the host
`/usr/lib/librknnrt.so` is mounted explicitly to keep it aligned with the
kernel driver.

See [`../rknn/README.md`](../rknn/README.md) for conversion, compose and
benchmark commands.

One-command deployment from an x86_64 workstation:

```bash
../rknn/deploy.sh --platform rk3576 --device cat-remote \
  --accept-upstream-license
```

Use `--dry-run`, `--offline`, `--no-up`, `--model-file`, `--model-url`, or an
explicit x86 `--builder-host` as documented in the shared README. RKNN Toolkit
conversion is never attempted on this ARM board.

On the measured image, `gst-inspect-1.0` registers
`rockchipmpp:mppvideodec` from `libgstrockchipmpp.so`; there is no separate
RGA element. The plugin links `librockchip_mpp.so.1` and `librga.so.2`, so the
configured `width=640 height=640 format=RGB` path performs the verified
integrated MPP decode/RGA conversion before `appsink`. Compose also mounts the
host `libgstvideoparsersbad.so` and `libgstcodecparsers-1.0.so.0` ABI for
`h264parse`/`h265parse`.

Default runtime release: `sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc7`
(override with `FALL_RK_IMAGE`). Published RepoDigest:
`sha256:8c79172138a0f510e26bd0f219f82b6a57ab98ff30f6828d96786e5131dfeae5`.
The RC7 candidate passed a 3-route, 20-second MQTT smoke (900/900 messages observed),
with no missing shared-library dependencies, no OOM, and clean exit. The measured runs used historical rc2 digest
`sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8`.
The external YOLO11n-Pose RKNN file is on license HOLD pending documented
Ultralytics AGPL-3.0 suitability or a commercial license; it is not in the
container image.
