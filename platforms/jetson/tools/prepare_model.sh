#!/usr/bin/env bash
set -euo pipefail

# Prepare a pose model for the Jetson runtime without putting build-only
# Python/ONNX dependencies in the production image.  The command is safe by
# default: it never uploads model assets and --dry-run performs no downloads,
# remote commands, or compose actions.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$PROJECT_DIR/models"}
MODEL_NAME=yolo11s-pose
DEVICE=
ONNX_PATH=
WEIGHTS_PATH=
ENGINE_PATH=
MANIFEST_PATH=
TRT_INPUT_NAME=${TRT_INPUT_NAME:-images}
IMG_SIZE=${IMG_SIZE:-640}
PROFILE_MIN=
PROFILE_OPT=
PROFILE_MAX=
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
UV_BIN=${UV_BIN:-uv}
ULTRALYTICS_VERSION=${ULTRALYTICS_VERSION:-8.3.0}
UPSTREAM_VERSION=${UPSTREAM_VERSION:-11}
DEPLOY=false
DRY_RUN=false
OFFLINE=false
FORCE=false
ACCEPT_LICENSE=false
REMOTE_PROJECT_DIR=${FALL_DETECTION_REMOTE_PROJECT_DIR:-/home/harvest/fall-detection}
OFFICIAL_DOWNLOAD=false
export TRTEXEC

usage() {
  cat <<'USAGE'
Usage: tools/prepare_model.sh [options]

Prepare a YOLO11-Pose ONNX and a target-specific TensorRT FP16 engine.  A
normal production run is:

  tools/prepare_model.sh --model yolo11s-pose --device orin-nano \
    --accept-upstream-license --deploy

Options:
  --model NAME             yolo11s-pose (default) or yolo11m-pose
  --device NAME            Fleet device (for example orin-nano/orin-nx)
  --onnx PATH              Existing ONNX; skips weight download/export
  --weights PATH           Existing Ultralytics .pt (requires acceptance)
  --model-dir PATH         Artifact directory (default: platforms/jetson/models)
  --engine PATH            Engine output path (default: model-specific)
  --manifest PATH          Manifest path (default: model-specific JSON)
  --input-name NAME        ONNX input name (default: images)
  --imgsz N                Square input size (default: 640)
  --min-shape SHAPE        Explicit profile, e.g. 1x3x640x640
  --opt-shape SHAPE        Explicit profile (defaults to min)
  --max-shape SHAPE        Explicit profile (defaults to opt)
  --accept-upstream-license
                           Required when downloading/exporting Ultralytics
                           weights. The upstream license remains applicable.
  --offline                No model download/export; use an existing ONNX
  --deploy                 Start production slim compose after preparation
  --no-deploy              Do not start compose (default)
  --remote-project-dir PATH
                           Project checkout on a --device (for --deploy)
  --dry-run                Print planned actions without changing state
  --force                  Rebuild even when a matching manifest is cached
  -h, --help               Show this help

The runtime image contains neither Torch, Ultralytics, nor ONNX Runtime. ONNX
is mounted from models/ and the engine is built on the destination Orin host.
USAGE
}

die() { echo "prepare_model: $*" >&2; exit 1; }
log() { printf '[prepare_model] %s\n' "$*"; }
run() {
  if "$DRY_RUN"; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
run_in_dir() {
  local directory=$1
  shift
  if "$DRY_RUN"; then
    printf '+ (cd %q && ' "$directory"
    printf '%q ' "$@"
    printf ')\n'
  else
    (cd "$directory" && "$@")
  fi
}

while (($#)); do
  case "$1" in
    --model) MODEL_NAME=${2:?missing value for --model}; shift 2 ;;
    --device) DEVICE=${2:?missing value for --device}; shift 2 ;;
    --onnx) ONNX_PATH=${2:?missing value for --onnx}; shift 2 ;;
    --weights) WEIGHTS_PATH=${2:?missing value for --weights}; shift 2 ;;
    --model-dir) MODEL_DIR=${2:?missing value for --model-dir}; shift 2 ;;
    --engine) ENGINE_PATH=${2:?missing value for --engine}; shift 2 ;;
    --manifest) MANIFEST_PATH=${2:?missing value for --manifest}; shift 2 ;;
    --input-name) TRT_INPUT_NAME=${2:?missing value for --input-name}; shift 2 ;;
    --imgsz) IMG_SIZE=${2:?missing value for --imgsz}; shift 2 ;;
    --min-shape) PROFILE_MIN=${2:?missing value for --min-shape}; shift 2 ;;
    --opt-shape) PROFILE_OPT=${2:?missing value for --opt-shape}; shift 2 ;;
    --max-shape) PROFILE_MAX=${2:?missing value for --max-shape}; shift 2 ;;
    --accept-upstream-license) ACCEPT_LICENSE=true; shift ;;
    --offline) OFFLINE=true; shift ;;
    --deploy) DEPLOY=true; shift ;;
    --no-deploy) DEPLOY=false; shift ;;
    --remote-project-dir) REMOTE_PROJECT_DIR=${2:?missing value for --remote-project-dir}; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

case "$MODEL_NAME" in
  yolo11s-pose|yolo11m-pose) ;;
  *) die "--model must be yolo11s-pose or yolo11m-pose" ;;
esac
[[ "$IMG_SIZE" =~ ^[0-9]+$ ]] || die "--imgsz must be an integer"

if [[ -z "$PROFILE_MIN" ]]; then PROFILE_MIN="1x3x${IMG_SIZE}x${IMG_SIZE}"; fi
if [[ -z "$PROFILE_OPT" ]]; then PROFILE_OPT="$PROFILE_MIN"; fi
if [[ -z "$PROFILE_MAX" ]]; then PROFILE_MAX="$PROFILE_OPT"; fi

if [[ -z "$ENGINE_PATH" ]]; then
  ENGINE_PATH="$MODEL_DIR/${MODEL_NAME}.sm87.trt10.3.fp16.engine"
fi
if [[ -z "$MANIFEST_PATH" ]]; then
  MANIFEST_PATH="$MODEL_DIR/${MODEL_NAME}.manifest.json"
fi

# Keep paths absolute so fleet and compose invocations cannot accidentally
# resolve against a different working directory.
abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}
MODEL_DIR=$(abspath "$MODEL_DIR")
ENGINE_PATH=$(abspath "$ENGINE_PATH")
MANIFEST_PATH=$(abspath "$MANIFEST_PATH")
[[ -n "$ONNX_PATH" ]] && ONNX_PATH=$(abspath "$ONNX_PATH")
[[ -n "$WEIGHTS_PATH" ]] && WEIGHTS_PATH=$(abspath "$WEIGHTS_PATH")

if [[ "$DEPLOY" == true && -z "$DEVICE" ]]; then
  die "--deploy requires --device so the compose action is explicit"
fi

if [[ "$OFFLINE" == true && ( -n "$WEIGHTS_PATH" || "$ACCEPT_LICENSE" == true ) ]]; then
  # Acceptance is harmless offline, but a requested local .pt would still
  # require an export tool/network-capable builder and is intentionally not
  # guessed at here.
  [[ -z "$WEIGHTS_PATH" ]] || die "--offline accepts an existing ONNX only (use --onnx)"
fi

if [[ -n "$WEIGHTS_PATH" || ( -z "$ONNX_PATH" && "$OFFLINE" == false ) ]]; then
  [[ "$ACCEPT_LICENSE" == true ]] || die "weight download/export requires --accept-upstream-license"
fi

mkdir_artifacts() {
  if [[ "$DRY_RUN" == true ]]; then
    log "would create $MODEL_DIR"
  else
    mkdir -p "$MODEL_DIR"
  fi
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}
bytes_file() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1"
}

OFFICIAL_URL="https://github.com/ultralytics/assets/releases/download/v${ULTRALYTICS_VERSION}/${MODEL_NAME}.pt"
ONNX_OUTPUT="$MODEL_DIR/${MODEL_NAME}.onnx"
if [[ -z "$ONNX_PATH" ]]; then ONNX_PATH="$ONNX_OUTPUT"; fi

mkdir_artifacts

if [[ "$DRY_RUN" == true ]]; then
  if [[ -n "$WEIGHTS_PATH" ]]; then
    log "would export $WEIGHTS_PATH with isolated uv builder"
  elif [[ -n "$ONNX_PATH" && -f "$ONNX_PATH" ]]; then
    log "would reuse existing ONNX $ONNX_PATH"
  elif [[ "$OFFLINE" == true ]]; then
    die "offline dry-run still needs an existing --onnx path"
  else
    log "would download official weights $OFFICIAL_URL"
    log "would export ONNX with isolated uv builder"
  fi
else
  if [[ -n "$WEIGHTS_PATH" ]]; then
    [[ -s "$WEIGHTS_PATH" ]] || die "weights not found: $WEIGHTS_PATH"
  elif [[ ! -s "$ONNX_PATH" ]]; then
    [[ "$OFFLINE" == false ]] || die "offline mode requires --onnx pointing to a non-empty file"
    log "downloading/exporting official $MODEL_NAME weights (Ultralytics v$ULTRALYTICS_VERSION)"
    WEIGHTS_PATH="$MODEL_DIR/${MODEL_NAME}.pt"
    OFFICIAL_DOWNLOAD=true
  fi

  if [[ -n "$WEIGHTS_PATH" ]]; then
    [[ "$OFFLINE" == false ]] || die "--offline cannot export a .pt; pass --onnx instead"
    command -v "$UV_BIN" >/dev/null 2>&1 || die "uv is required for isolated export"
    # The official Ultralytics CLI resolves and downloads the named asset. No
    # model is placed in the runtime image. PyPI traffic uses the caller's
    # mirror (TUNA by default); callers may override UV_INDEX_URL.
    export UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
    log "exporting ONNX in isolated builder environment"
    if [[ "$OFFICIAL_DOWNLOAD" == true ]]; then
      run_in_dir "$MODEL_DIR" env UV_INDEX_URL="$UV_INDEX_URL" "$UV_BIN" run --with "ultralytics==${ULTRALYTICS_VERSION}" --with onnx --with onnxslim -- \
        yolo export "model=${MODEL_NAME}.pt" format=onnx imgsz="$IMG_SIZE" dynamic=True simplify=True project="$MODEL_DIR" name="$MODEL_NAME"
    else
      run "$UV_BIN" run --with "ultralytics==${ULTRALYTICS_VERSION}" --with onnx --with onnxslim -- \
        yolo export "model=$WEIGHTS_PATH" format=onnx imgsz="$IMG_SIZE" dynamic=True simplify=True project="$MODEL_DIR" name="$MODEL_NAME"
    fi
    # Ultralytics may place the artifact beside the input rather than under
    # project/name; locate only the requested model and keep the output stable.
    if [[ ! -s "$ONNX_PATH" ]]; then
      candidate=$(find "$MODEL_DIR" -maxdepth 3 -type f -name "${MODEL_NAME}.onnx" -print -quit)
      [[ -n "$candidate" && -s "$candidate" ]] || die "ONNX export did not produce ${MODEL_NAME}.onnx"
      [[ "$candidate" == "$ONNX_PATH" ]] || run cp "$candidate" "$ONNX_PATH"
    fi
  fi
fi

[[ "$DRY_RUN" == true || -s "$ONNX_PATH" ]] || die "ONNX not found: $ONNX_PATH"

ONNX_SHA=
ONNX_BYTES=
if [[ "$DRY_RUN" == false ]]; then
  ONNX_SHA=$(sha256_file "$ONNX_PATH")
  ONNX_BYTES=$(bytes_file "$ONNX_PATH")
fi

# A manifest is the cache key.  Do not reuse a target engine when the ONNX,
# profile, input name, or target host changed.
if [[ "$DRY_RUN" == false && "$FORCE" == false && -s "$ENGINE_PATH" && -s "$MANIFEST_PATH" ]]; then
  if python3 - "$MANIFEST_PATH" "$ONNX_SHA" "$MODEL_NAME" "$TRT_INPUT_NAME" "$PROFILE_MIN" "$PROFILE_OPT" "$PROFILE_MAX" "$DEVICE" <<'PY'
import json, sys
path, onnx_sha, model, name, pmin, popt, pmax, device = sys.argv[1:]
try:
    m=json.load(open(path))
    e=m.get("engine", {})
    o=m.get("onnx", {})
    ok=(m.get("model")==model and o.get("sha256")==onnx_sha and
        e.get("input_name")==name and e.get("profile",{}).get("min")==pmin and
        e.get("profile",{}).get("opt")==popt and e.get("profile",{}).get("max")==pmax and
        (not device or e.get("target_device")==device) and e.get("trt_major_minor")=="10.3")
    raise SystemExit(0 if ok else 1)
except (OSError, ValueError, KeyError, TypeError):
    raise SystemExit(1)
PY
  then
    log "cache hit: $ENGINE_PATH (manifest matches ONNX/profile/target)"
    if [[ "$DEPLOY" == true ]]; then
      log "deploying cached engine on $DEVICE"
    else
      exit 0
    fi
  fi
fi

if [[ "$DRY_RUN" == true ]]; then
  if [[ -n "$DEVICE" ]]; then
    log "would push ONNX to $DEVICE and invoke host TRT10.3 trtexec (SM87)"
    log "would pull target engine and write manifest $MANIFEST_PATH"
  else
    log "would invoke $TRTEXEC via tools/build_engine.sh"
    log "would write manifest $MANIFEST_PATH"
  fi
  [[ "$DEPLOY" == true ]] && log "would run docker compose -f docker-compose.slim.yml up -d on $DEVICE"
  exit 0
fi

mkdir -p "$(dirname "$ENGINE_PATH")" "$(dirname "$MANIFEST_PATH")"
if [[ -n "$DEVICE" ]]; then
  fleet_bin=${FLEET_BIN:-}
  if [[ -z "$fleet_bin" ]]; then
    if command -v fleet >/dev/null 2>&1; then
      fleet_bin=$(command -v fleet)
    else
      fleet_bin="$HOME/.rpty/bin/fleet"
    fi
  fi
  command -v "$fleet_bin" >/dev/null 2>&1 || die "fleet CLI not found; omit --device for a local build"
  if [[ "$DEPLOY" == true ]]; then
    remote_compose="$REMOTE_PROJECT_DIR/platforms/jetson/docker-compose.slim.yml"
    remote_config="$REMOTE_PROJECT_DIR/platforms/jetson/config/config.json"
    remote_models="$REMOTE_PROJECT_DIR/platforms/jetson/models"
    run "$fleet_bin" exec "$DEVICE" -- test -f "$remote_compose"
    run "$fleet_bin" exec "$DEVICE" -- test -f "$remote_config"
    run "$fleet_bin" exec "$DEVICE" -- test -d "$remote_models"
    if [[ "$DRY_RUN" == false ]]; then
      config_engine_line=$($fleet_bin exec "$DEVICE" -- grep -E '"engine_path"[[:space:]]*:' "$remote_config" 2>/dev/null || true)
      [[ "$config_engine_line" == *"$(basename "$ENGINE_PATH")"* ]] || die "remote config engine_path does not name $(basename "$ENGINE_PATH"); update $remote_config or pass --engine matching it"
    fi
  fi
  remote_root="/tmp/fall-detection-model-prep-$$"
  remote_onnx="$remote_root/$(basename "$ONNX_PATH")"
  remote_engine="$remote_root/$(basename "$ENGINE_PATH")"
  run "$fleet_bin" exec "$DEVICE" -- mkdir -p "$remote_root"
  remote_builder="$remote_root/build_engine.sh"
  run "$fleet_bin" push "$DEVICE" "$ONNX_PATH" "$remote_onnx"
  run "$fleet_bin" push "$DEVICE" "$SCRIPT_DIR/build_engine.sh" "$remote_builder"
  run "$fleet_bin" exec "$DEVICE" -- chmod +x "$remote_builder"
  run "$fleet_bin" exec "$DEVICE" -- bash "$remote_builder" "$remote_onnx" "$remote_engine" \
    "--minShapes=${TRT_INPUT_NAME}:${PROFILE_MIN}" \
    "--optShapes=${TRT_INPUT_NAME}:${PROFILE_OPT}" \
    "--maxShapes=${TRT_INPUT_NAME}:${PROFILE_MAX}"
  run "$fleet_bin" pull "$DEVICE" "$remote_engine" "$ENGINE_PATH"
else
  run "$SCRIPT_DIR/build_engine.sh" "$ONNX_PATH" "$ENGINE_PATH" \
    "--minShapes=${TRT_INPUT_NAME}:${PROFILE_MIN}" \
    "--optShapes=${TRT_INPUT_NAME}:${PROFILE_OPT}" \
    "--maxShapes=${TRT_INPUT_NAME}:${PROFILE_MAX}"
fi

[[ -s "$ENGINE_PATH" ]] || die "engine build did not produce $ENGINE_PATH"
ENGINE_SHA=$(sha256_file "$ENGINE_PATH")
ENGINE_BYTES=$(bytes_file "$ENGINE_PATH")
HOST_TRT="10.3"
HOST_SM="87"
HOST_CUDA=
if [[ -n "$DEVICE" ]]; then
  host_info=$($fleet_bin exec --literal "$DEVICE" -- 'printf "trt="; dpkg-query -W -f="${Version}" tensorrt-libs 2>/dev/null || true; printf " cuda="; sed -n "s/^# R[0-9].*CUDA Version \([^ ]*\).*$/\1/p" /etc/nv_tegra_release | head -1 || true; printf " sm=87\n"' 2>/dev/null || true)
  HOST_TRT=$(printf '%s\n' "$host_info" | sed -n 's/.*trt=\([^ ]*\).*/\1/p' | head -1)
  [[ -n "$HOST_TRT" ]] || HOST_TRT="10.3"
  HOST_CUDA=$(printf '%s\n' "$host_info" | sed -n 's/.*cuda=\([^ ]*\).*/\1/p' | head -1)
fi

WEIGHTS_SHA=
if [[ -n "$WEIGHTS_PATH" && -s "$WEIGHTS_PATH" ]]; then
  WEIGHTS_SHA=$(sha256_file "$WEIGHTS_PATH")
fi
export MANIFEST_PATH MODEL_NAME UPSTREAM_VERSION ULTRALYTICS_VERSION OFFICIAL_URL
export WEIGHTS_SHA ACCEPT_LICENSE ONNX_PATH ONNX_SHA ONNX_BYTES ENGINE_PATH ENGINE_SHA ENGINE_BYTES
export DEVICE HOST_SM HOST_TRT HOST_CUDA TRT_INPUT_NAME IMG_SIZE PROFILE_MIN PROFILE_OPT PROFILE_MAX
python3 - "$MANIFEST_PATH" <<'PY'
import json, os
def env(name, default=""):
    return os.environ.get(name, default)
def boolean(name):
    return env(name).lower() == "true"
manifest = {
  "schema_version": 1,
  "model": env("MODEL_NAME"),
  "upstream": {
    "provider": "Ultralytics",
    "package_version": env("ULTRALYTICS_VERSION"),
    "model_version": env("UPSTREAM_VERSION"),
    "weights_url": env("OFFICIAL_URL"),
    "weights_sha256": env("WEIGHTS_SHA") or None,
    "license": "AGPL-3.0-or-Enterprise",
    "license_accepted": boolean("ACCEPT_LICENSE")
  },
  "onnx": {"path": env("ONNX_PATH"), "sha256": env("ONNX_SHA"), "bytes": int(env("ONNX_BYTES", "0"))},
  "engine": {
    "path": env("ENGINE_PATH"), "sha256": env("ENGINE_SHA"), "bytes": int(env("ENGINE_BYTES", "0")),
    "target_device": env("DEVICE") or None, "sm": env("HOST_SM"),
    "trt_major_minor": "10.3", "trt_package": env("HOST_TRT"), "cuda": env("HOST_CUDA") or None,
    "precision": "fp16", "input_name": env("TRT_INPUT_NAME"),
    "input": [1, 3, int(env("IMG_SIZE")), int(env("IMG_SIZE"))],
    "profile": {"min": env("PROFILE_MIN"), "opt": env("PROFILE_OPT"), "max": env("PROFILE_MAX")}
  },
  "runtime": {"model_external": True, "runtime_image": "sensecraft-missionpack.seeed.cn/solution/fall-detection-jetson:0.1.0-rc1"}
}
with open(os.sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PY
log "wrote manifest $MANIFEST_PATH"

if [[ "$DEPLOY" == true ]]; then
  # The ONNX is a build input and is deliberately not copied to the
  # deployment checkout. Only the target-specific engine and its newly
  # generated manifest are transferred for compose to mount from models/.
  run "$fleet_bin" push "$DEVICE" "$ENGINE_PATH" "$remote_models/$(basename "$ENGINE_PATH")"
  run "$fleet_bin" push "$DEVICE" "$MANIFEST_PATH" "$remote_models/$(basename "$MANIFEST_PATH")"
fi

if [[ "$DEPLOY" == true ]]; then
  compose="$PROJECT_DIR/docker-compose.slim.yml"
  [[ -f "$compose" ]] || die "slim compose not found: $compose"
  if [[ -n "$DEVICE" ]]; then
    [[ -n "$REMOTE_PROJECT_DIR" ]] || die "--deploy with --device requires --remote-project-dir or FALL_DETECTION_REMOTE_PROJECT_DIR"
    run "$fleet_bin" exec --literal "$DEVICE" -- "cd $(printf '%q' "$REMOTE_PROJECT_DIR") && docker compose -f platforms/jetson/docker-compose.slim.yml up -d"
  else
    run docker compose -f "$compose" up -d
  fi
fi

log "ready: ONNX=$ONNX_PATH ENGINE=$ENGINE_PATH"
