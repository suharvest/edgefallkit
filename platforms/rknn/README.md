# Rockchip RKNN shared runtime

This directory is the single implementation shared by RK3576 and RK3588.
The staged, source-aligned FP16/INT8 test procedure is documented in
[evaluation/RK_ALIGNED_TEST_PLAN.md](../../evaluation/RK_ALIGNED_TEST_PLAN.md).
The single-route compatibility path is `rtspsrc -> rtp{h264,h265}depay -> parse ->
mppvideodec(width/height/format=RGB) -> appsink uint8 -> RKNNLite`. For
multi-route production measurements, `mppvideodec` can emit NV12 and the
current Python path performs color conversion/resize on CPU. On the
actual `cat-remote` and `radxa` images there is no standalone RGA GStreamer
element: `libgstrockchipmpp.so` links both `librockchip_mpp.so.1` and
`librga.so.2`, and `mppvideodec` exposes the resize, RGB conversion and DMA
caps. We therefore use that verified integrated MPP/RGA path instead of naming
a nonexistent `rgaconvert` plugin.

The deployed Python process does not import Torch, Ultralytics or ONNX Runtime.
RKNN Lite/librknnrt performs the pose network; a small pybind11 C++ extension
performs DFL/keypoint decode and NMS. Python retains the tracker, frozen
temporal MLP, state machines and MQTT contract. OpenCV/FFmpeg plus NumPy remain
runtime fallbacks, not build-only recovery tools.

The output keeps the reCamera MQTT contract (`timestamp` in milliseconds,
`inference_time_ms`, stream-global `event_id`, aggregate state/counts,
`features`, `pose17`, and `persons`) and adds `stream_id`, `pipeline_ms` and
`coordinate_space`. `pose17` and `bbox` are normalized against the 640x640
letterboxed model canvas. Geometry is invariant in that space. If a consumer
needs original-camera pixels it must retain the source width plus OpenCV's
scale/pad values and invert the letterbox; this runtime deliberately does not
mislabel padded coordinates as original pixels.

At the aggregate level `person_count` counts currently visible detections,
while `fallen_count` includes retained tracks still in fallen/recovering state
during the configured occlusion grace. `tracking=true` can therefore coexist
with `person_detected=false`. Per-person `event_id` is that track's sequence;
top-level `event_id`/`global_event_id` and each item in `events` use the
monotonic stream-global sequence (`event_id_scope=stream_global_event_id`).

## One-command model preparation and deployment

Use the deployment entry point from an x86_64 Mac/Linux workstation. The
license acknowledgement is deliberately explicit; it is not persisted or
silently inferred from a cached file:

```bash
platforms/rknn/deploy.sh \
  --platform rk3576 --device cat-remote \
  --accept-upstream-license

platforms/rknn/deploy.sh \
  --platform rk3588 --device radxa \
  --accept-upstream-license
```

The scripts first validate `models/SHA256SUMS`. A matching pose model is a
cache hit: no model download or conversion occurs. The small platform-native
`temporal-rk3576.npz`/`temporal-rk3588.npz` files ship with the project. The
deployment then uses Fleet to push the pose and temporal artifacts plus config
and Compose, pulls the immutable RC2 runtime, verifies its RepoDigest, runs
`docker compose config` and `app.py --validate`, and finally runs
`docker compose up -d`. Add `--no-up` to stop after validation or `--dry-run`
to print every planned operation.

Offline deployment requires both a manifest-verified local pose model and the
already cached remote runtime:

```bash
platforms/rknn/deploy.sh --platform rk3576 --device cat-remote \
  --accept-upstream-license --offline
```

To supply an artifact rather than build from official weights, use a prebuilt
`.rknn` or a nine-output raw-head `.onnx` and preferably pin its checksum:

```bash
platforms/rknn/prepare_model.sh --platform rk3588 \
  --accept-upstream-license \
  --model-file /secure/path/pose.rknn \
  --model-sha256 <sha256>

# HTTPS .rknn/.onnx URLs are also accepted:
platforms/rknn/prepare_model.sh --platform rk3588 \
  --accept-upstream-license --model-url https://example/pose.rknn \
  --model-sha256 <sha256>
```

When no source is supplied, an ephemeral x86_64-only model-builder obtains
the official `yolo11n-pose.pt` through Ultralytics, exports fixed
`1x3x640x640` box/class/keypoint raw heads, and converts them with RKNN Toolkit
2.3.2. The builder contains Torch, Ultralytics and RKNN Toolkit; none is copied
to the runtime image. This flow does not upload or re-host a YOLO model.

RKNN Toolkit conversion cannot run on RK3576/RK3588 ARM hosts. An ARM call with
a missing model exits with an exact x86 command. It can also delegate to an
explicit x86 Fleet device:

```bash
platforms/rknn/prepare_model.sh --platform rk3576 \
  --accept-upstream-license --builder-host <x86-fleet-device>
```

Use `--dry-run` first when checking a new Fleet/device layout. `--offline`
never downloads a model, pulls a builder, or contacts the official source.
Passing `--accept-upstream-license` acknowledges review; it does not change the
current YOLO-derived pose artifact license HOLD or grant redistribution rights.

## Model conversion

Use RKNN Toolkit on x86; true runtime validation still happens on both boards.
The source ONNX is the fixed `1x3x640x640` nine-output raw pose head. For INT8,
provide a representative calibration text file; start with FP16 to establish
correctness.

```bash
uv run --isolated --with rknn-toolkit2==2.3.2 \
  --with 'setuptools<81' --with 'onnx==1.16.1' \
  python platforms/rknn/tools/convert_pose_rknn.py \
  --onnx yolo11n_pose_rawhead.onnx \
  --platform rk3576 --out yolo11n_pose_rawhead_fp16.rk3576.rknn
```

Repeat with `--platform rk3588`. Add `--dataset calibration.txt` for W8A8.
Never infer RK3588 correctness from an RK3576 pass (or vice versa).

The learned temporal MLP is exported from the frozen header without Torch:

```bash
uv run --with numpy python platforms/rknn/tools/export_temporal_npz.py \
  platforms/jetson/main/temporal_model_weights.h temporal-yolo11s.npz
```

The bundled `temporal-yolo11s.npz` remains only as a cross-platform fallback
and functional fixture. Production configs now select the independently
frozen `temporal-rk3576.npz` and `temporal-rk3588.npz`. Each was fit/selected/
refit on Subjects 1–3 before Subject 4 was unlocked; the clean frozen S4
results are recorded in `TEMPORAL_TRAINING.md` and `evaluation/RESULTS.md`.

## Runtime and benchmark

The backend policy is explicit in each platform config:

```json
{
  "video": {
    "backend": "gstreamer_mpp",
    "strict": false,
    "fallback": "opencv_ffmpeg",
    "codec": "h264",
    "latency_ms": 100,
    "appsink_timeout_ms": 2000,
    "failure_limit": 3
  },
  "postprocess": {
    "backend": "cpp",
    "strict": false,
    "fallback": "numpy"
  }
}
```

`strict=true` converts a missing/failing requested backend into a fatal error.
With the shipped non-strict policy, three empty/error reads switch the stream
to OpenCV/FFmpeg; a C++ import or call error permanently switches that worker
to NumPy. The active choices are emitted in startup logs and every MQTT payload
as `source_backend` and `postprocess_backend`. Per-stream `video` values can
override global video values. Set `codec=h265` for an HEVC RTSP stream.

The MPP RGB path currently produces the fixed square model canvas directly.
The NV12 path preserves decoder output and performs conversion/resize in
Python. The measured Spark streams are 640x640, so both are geometry-equivalent.
For a
non-square source that must retain letterbox geometry, select
`backend=opencv_ffmpeg` until a source-dimension-aware RGA pad stage is added;
do not silently compare stretched coordinates with letterboxed baselines.

Run from `platforms/rk3576` or `platforms/rk3588`:

```bash
docker compose build
docker compose run --rm fall-detection app.py --config /config/config.json --validate
docker compose up -d fall-detection
docker compose --profile benchmark run --rm benchmark
```

For a multi-context throughput measurement, override the benchmark command:

```bash
docker compose run --rm benchmark benchmark.py \
  --model /models/yolo11n_pose_rawhead_fp16.rk3576.rknn \
  --contexts 2 --iterations 400 --json-out /results/rk3576-2ctx.json
```

`inference_ms` measures native `rknnlite.inference()` only. `pipeline_ms`
starts after source read returns and includes inference, raw-head decode/NMS,
tracking, temporal/fall state and payload construction; it excludes source
read/preprocessing and MQTT transmission. RTSP decode latency is measured
separately in the live MQTT timestamps and must not be conflated with model
latency.

The 2026-09-05 aligned RTSP measurements establish S INT8 capacity at 1 route
on RK3576 and 5 routes on RK3588. A separate RK3588 native hybrid moved
DMA-BUF→serialized RGA→RKNN `set_io_mem` out of Python and reduced measured CPU
and RSS, but only checked inference output checksums. It did not execute pose
decode, tracking, temporal MLP or MQTT and is therefore experimental
performance-only, not a production-capacity or precision-equivalence result.

The Dockerfile is multi-stage. The compiler, Python headers and pybind11 are
present only in `builder`; runtime receives only `rknn_postprocess*.so` plus
NumPy/OpenCV and GStreamer/PyGObject. The default base image, apt source and
PyPI index use DaoCloud, Tsinghua and Aliyun mirrors respectively. Compose
mounts the board's verified `libgstrockchipmpp.so`,
`libgstvideoparsersbad.so`, `libgstcodecparsers-1.0.so.0`,
`librockchip_mpp.so.1` and `librga.so.2` read-only, alongside the host RKNN
ABI. Mounting only the H.264/H.265 parser plugin and its codec-parser ABI avoids
installing the complete Debian `plugins-bad` dependency tree in the image.

The earlier 2026-08-13 `linux/arm64` build of `fall-detection-rknn:2.3.0` is
144,143,406 bytes by the board-side `docker image inspect`; the exact
`docker image save ... | gzip -1 | wc -c` result is 143,317,758 bytes. Models
and host RKNN Lite/librknnrt mounts are excluded. Compose mounts only the host
`rknnlite`, `ruamel`, and `psutil` package directories rather than the whole
`dist-packages` tree; mounting the whole tree shadows the wheel NumPy and can
fail on missing host BLAS libraries.
An import audit found NumPy 1.26.4 and OpenCV 4.10.0, and confirmed Torch,
Ultralytics and ONNX Runtime are absent. The conversion environment is kept
outside this runtime image.

The final 2026-08-13 `fall-detection-rknn:2.4.0` artifact built on `radxa` and
loaded unchanged on `cat-remote` has image ID
`sha256:5ceaf23a73707d5ea86186c8c7d2c2bda2577aa72a82ce614beb4174cd0320fa`.
`docker image inspect` reports 257,793,213 bytes; `docker save | gzip -1` is
255,849,560 bytes with SHA256
`c3e26ce8340e7a560a077d46661aa3040e848eb8d723f5196f35b525ef89a0a3`.
The first working GStreamer build was 342,271,154 bytes (gzip 339,397,541
bytes): Debian's full `gstreamer1.0-plugins-bad` closure was the removable
84,477,941-byte image-size increase. Relative to the earlier 144,143,406-byte
OpenCV runtime, the remaining 113,649,807 bytes are Python GI plus the
GStreamer tools/base/good runtime needed for `rtspsrc`, RTP depay, caps and
appsink. Those dependencies cannot be removed without dropping the GStreamer
backend; the Rockchip decoder/parser ABI itself remains a read-only host mount.

Both boards passed the final runtime audit: no `cc`, `gcc`, `g++` or `c++`, no
pybind11 headers/module/cache, and only the stripped 200,232-byte
`rknn_postprocess` shared object is copied from the builder. Its runtime links
are limited to `libstdc++`, `libm`, `libgcc_s` and `libc`. Both boards also
resolved `rtspsrc`, `rtph264depay`, `h264parse`, `mppvideodec` and `appsink`
inside the final container.

Release candidate `0.1.0-rc6` is published at
`sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc6`, with
registry digest
`sha256:b74bbe9540bbc950f3ea3e7bb1725decab86b81af35f389cd22af6ee94783d4a`.
The arm64 image built on `spark` has local image ID
`sha256:f1071b58f79d02eae3df58eee89c71766383ba311b5cc30d2e9405b69897240e`
and inspect size 472,694,512 bytes. A pre-push container smoke imported NumPy,
OpenCV 4.10.0, GI and the native `rknn_postprocess` extension. Board-side rc3
pull validation remains pending because both RK boards went offline after the
performance run.

The preceding rc2 artifact was independently pulled and runtime-smoked on both
`cat-remote` and `radxa`; its immutable RepoDigest is
`sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8`.

The image deliberately excludes the RKNN pose models. Those models derive from
Ultralytics YOLO11n-Pose reference weights; redistribution is on **license
HOLD** until AGPL-3.0 suitability or an applicable Ultralytics commercial
license is documented. The independently trained temporal NPZ files are also
external artifacts and do not imply permission to redistribute the pose model.

Safety invariant: an invalid/missing current pose can retain or time out a
track, but cannot originate a fall event even if a cached temporal probability
is positive.
