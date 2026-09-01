#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL.onnx OUTPUT.engine [trtexec args...]" >&2
  echo "Builds on the Jetson host, where TensorRT 10.3 and SM87 are installed." >&2
  exit 2
fi

ONNX_PATH=$1
ENGINE_PATH=$2
shift 2
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

[[ -f "$ONNX_PATH" ]] || { echo "ONNX model not found: $ONNX_PATH" >&2; exit 1; }
[[ -x "$TRTEXEC" ]] || { echo "trtexec not executable: $TRTEXEC" >&2; exit 1; }
mkdir -p "$(dirname "$ENGINE_PATH")"

echo "== TensorRT builder =="
"$TRTEXEC" --version || true  # trtexec 10.3 exits 1 on --version
echo "ONNX:    $ONNX_PATH"
echo "Engine:  $ENGINE_PATH"
echo "Target:  Orin SM87 / host TensorRT 10.3"

# YOLO11 exports are often dynamic. Without a profile trtexec can select the
# parser's placeholder 1x1 shape and either fail tactic selection or produce an
# unusable engine. Use the conventional `images` input and a fixed 640x640
# profile by default; set TRT_INPUT_NAME or pass an explicit shape argument
# when the exported graph uses a different name/profile.
TRT_INPUT_NAME=${TRT_INPUT_NAME:-images}
TRT_STATIC_SHAPE=${TRT_STATIC_SHAPE:-false}
shape_args=()
has_explicit_shapes=false
for argument in "$@"; do
  case "$argument" in
    --minShapes=*|--optShapes=*|--maxShapes=*|--shapes=*|--explicitBatch)
      has_explicit_shapes=true
      ;;
  esac
done
if [[ "$TRT_STATIC_SHAPE" != true && "$TRT_STATIC_SHAPE" != false ]]; then
  echo "TRT_STATIC_SHAPE must be true or false" >&2
  exit 2
fi
if [[ "$has_explicit_shapes" == false && "$TRT_STATIC_SHAPE" == false ]]; then
  shape_args=(
    "--minShapes=${TRT_INPUT_NAME}:1x3x640x640"
    "--optShapes=${TRT_INPUT_NAME}:1x3x640x640"
    "--maxShapes=${TRT_INPUT_NAME}:1x3x640x640"
  )
fi

trtexec_args=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--fp16"
  "--skipInference"
  "--builderOptimizationLevel=3"
  "--timingCacheFile=${ENGINE_PATH}.timing.cache"
)
if [[ "$has_explicit_shapes" == false && "$TRT_STATIC_SHAPE" == false ]]; then
  trtexec_args+=("${shape_args[@]}")
fi
trtexec_args+=("$@")
"$TRTEXEC" "${trtexec_args[@]}"

test -s "$ENGINE_PATH"
echo "Engine written: $ENGINE_PATH"
