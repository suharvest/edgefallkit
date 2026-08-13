# Jetson pose artifacts

Neither ONNX nor TensorRT engines are baked into the final runtime image.
Compose mounts this directory read-only at `/models`. This keeps the runtime
image small and prevents an engine built against the wrong TensorRT ABI from
being shipped silently.

## One-command preparation (builder-only dependencies)

The supported path for a fresh checkout is the preparation helper. It downloads
and exports the official Ultralytics reference model only after an explicit
license acknowledgement, then builds the engine on the destination Orin with
that host's TensorRT 10.3/SM87 `trtexec`:

```bash
bash tools/prepare_model.sh --model yolo11s-pose --device orin-nano \
  --accept-upstream-license --deploy
```

Use `--model yolo11m-pose` for an NX-oriented build. The optional
`--remote-project-dir` (or `FALL_DETECTION_REMOTE_PROJECT_DIR`) names the
checkout on the Jetson; it defaults to `/home/harvest/fall-detection`.
Before `--deploy`, that checkout must already contain
`platforms/jetson/{docker-compose.slim.yml,config/config.json,models/}` and its
`engine_path` must name the generated engine. The helper refuses to guess or
overwrite a remote configuration. It transfers only the target-specific
engine and manifest for deployment; the ONNX and `.pt` remain build inputs and
are never uploaded to the image registry.

For an existing Spark/CI ONNX, no upstream download or Python builder is
needed:

```bash
bash tools/prepare_model.sh --offline --onnx /path/to/yolo11s-pose.onnx \
  --device orin-nano --engine models/yolo11s-pose.sm87.trt10.3.fp16.engine
```

`--dry-run` exercises argument checks and prints the planned transfer/build,
while `--force` bypasses a matching cache. A JSON manifest beside the engine
records the official source URL, Ultralytics package/model version, SHA-256 and
size for each artifact, TensorRT/CUDA/SM information, input/profile, and the
upstream license acknowledgement. TensorRT engines are cacheable only when
the ONNX SHA, target, input name/profile, and TRT 10.3 ABI match.

The `uv run --with ...` environment is ephemeral and builder-only. The slim
runtime image intentionally contains no Torch, Ultralytics, ONNX, or ONNX
Runtime packages. Review the upstream AGPL-3.0/Enterprise terms before any
commercial redistribution; this repository does not re-host the reference
weights.

## Restore the evaluated ONNX from Spark

The exact evaluated ONNX files are durably backed up on Spark:

| File | Bytes | SHA256 |
|---|---:|---|
| `yolo11s-pose.onnx` | 39 MiB | `8918cfc11983ee4d38019308fe87be33116da21b715e1c1a0cb2f97e36ffb8d2` |
| `yolo11m-pose.onnx` | 81 MiB | `7fe8762007efd17976bdf140b99eac9a8069accd7020a5873ad20dd4caaf633c` |

From the Mac/controller:

```bash
~/.rpty/bin/fleet pull spark \
  /home/harvest/datasets/fall-detection/models/yolo11s-pose.onnx \
  platforms/jetson/models/yolo11s-pose.onnx
sha256sum platforms/jetson/models/yolo11s-pose.onnx
```

Use `yolo11s-pose.onnx` for Orin Nano and `yolo11m-pose.onnx` for Orin NX by
default.

## Re-export from the official model

Ultralytics documents the pose models and export command at
`https://docs.ultralytics.com/tasks/pose/`. In an isolated **build-only**
environment, not in the deployment image:

```bash
uv run --with ultralytics yolo export \
  model=yolo11s-pose.pt format=onnx imgsz=640 dynamic=True simplify=True
```

Repeat with `yolo11m-pose.pt` for NX. This command resolves the named official
weight through Ultralytics and exports ONNX. Record the package version and the
resulting SHA256 whenever regenerating an artifact; a new export is not assumed
bit-identical to the evaluated files above. Review the model license before
commercial distribution.

## Build the target engine

Build on the destination Orin with its TensorRT 10.3 installation:

```bash
../tools/build_engine.sh \
  yolo11s-pose.onnx yolo11s-pose.sm87.trt10.3.fp16.engine
```

Do not distribute the Spark `.engine` backup as a universal binary. TensorRT
engines depend on the GPU architecture and TensorRT/CUDA ABI. The historical
Spark engines are retained only for exact-device recovery and benchmark audit.
