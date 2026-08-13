#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

usage() {
    cat <<'EOF'
Usage: ./deploy.sh PLATFORM --accept-upstream-license [platform options]

Platforms:
  jetson   Prepare YOLO11s/m Pose ONNX, build TensorRT on Orin, optionally deploy
  rk3576   Build/provide RKNN on x86, transfer to RK3576, and deploy
  rk3588   Build/provide RKNN on x86, transfer to RK3588, and deploy
  hailo    Download/verify the official Hailo-8 HEF and deploy on Raspberry Pi

Examples:
  ./deploy.sh jetson --accept-upstream-license --device orin-nano --deploy
  ./deploy.sh rk3576 --accept-upstream-license --device cat-remote
  ./deploy.sh rk3588 --accept-upstream-license --device radxa
  ./deploy.sh hailo --accept-upstream-license

Run `./deploy.sh PLATFORM --help` for platform-specific offline, cache, model,
and dry-run options. Models are prepared outside the published runtime images.
EOF
}

[ "$#" -gt 0 ] || { usage >&2; exit 2; }
platform=$1
shift

case "$platform" in
    jetson)
        exec bash "$ROOT/platforms/jetson/tools/prepare_model.sh" "$@"
        ;;
    rk3576|rk3588)
        exec sh "$ROOT/platforms/rknn/deploy.sh" --platform "$platform" "$@"
        ;;
    hailo)
        exec sh "$ROOT/platforms/rpi-hailo/deploy.sh" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "unknown platform: $platform" >&2
        usage >&2
        exit 2
        ;;
esac
