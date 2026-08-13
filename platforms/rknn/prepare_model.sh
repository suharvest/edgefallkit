#!/bin/sh
set -eu

usage() {
  cat <<'EOF'
Usage: prepare_model.sh --platform rk3576|rk3588 --accept-upstream-license [options]
  --output-dir DIR        platform models directory
  --manifest FILE         SHA256SUMS file (default: OUTPUT_DIR/SHA256SUMS)
  --model-file FILE       existing .rknn or raw-head .onnx
  --model-url HTTPS_URL   download an existing .rknn or raw-head .onnx
  --model-sha256 SHA256   verify a caller-supplied model before accepting it
  --builder-host DEVICE   delegate missing-model build to an x86_64 Fleet device
  --offline               prohibit network/download/build-image operations
  --dry-run               print actions without changing state

The official-source path downloads yolo11n-pose.pt through Ultralytics inside
an ephemeral x86_64 model-builder and exports nine raw heads before RKNN 2.3.2
conversion. It never runs on ARM and never enters the runtime image.
EOF
}

platform='' output_dir='' manifest='' model_file='' model_url='' model_sha='' builder_host=''
accept=no offline=no dry_run=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform) platform=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --manifest) manifest=$2; shift 2 ;;
    --model-file) model_file=$2; shift 2 ;;
    --model-url) model_url=$2; shift 2 ;;
    --model-sha256) model_sha=$2; shift 2 ;;
    --builder-host) builder_host=$2; shift 2 ;;
    --accept-upstream-license) accept=yes; shift ;;
    --offline) offline=yes; shift ;;
    --dry-run) dry_run=yes; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ "$accept" = yes ] || { echo "license gate: pass --accept-upstream-license after reviewing Ultralytics AGPL-3.0/commercial terms" >&2; exit 3; }
case "$platform" in rk3576|rk3588) ;; *) echo "--platform must be rk3576 or rk3588" >&2; exit 2;; esac
[ -z "$model_file" ] || [ -z "$model_url" ] || { echo "choose only one of --model-file/--model-url" >&2; exit 2; }
if [ -n "$model_url" ]; then
  case "$model_url" in https://*) ;; *) echo "--model-url must use HTTPS" >&2; exit 2;; esac
  case "$model_url" in https://huggingface.co/*)
    echo "direct huggingface.co downloads are disabled; use HF_ENDPOINT=https://hf-mirror.com with hf download, then --model-file" >&2; exit 2;; esac
  clean_url=${model_url%%\?*}
  case "$clean_url" in *.rknn) url_suffix=.rknn;; *.onnx) url_suffix=.onnx;; *) echo "--model-url must end in .rknn or .onnx" >&2; exit 2;; esac
fi
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
if [ -z "$output_dir" ]; then
  platform_dir=$(CDPATH='' cd -- "$script_dir/../$platform" && pwd)
  output_dir=$platform_dir/models
fi
manifest=${manifest:-$output_dir/SHA256SUMS}
name=yolo11n_pose_rawhead_fp16.$platform.rknn
output=$output_dir/$name

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
record_sha() {
  digest=$1; tmp=$manifest.tmp.$$
  mkdir -p "$output_dir"
  if [ -f "$manifest" ]; then awk -v name="$name" '$2!=name && $2!=("*" name)' "$manifest" > "$tmp"
  else : > "$tmp"; fi
  echo "$digest  $name" >> "$tmp"; mv "$tmp" "$manifest"
}
check_supplied_sha() {
  actual=$1
  [ -z "$model_sha" ] || [ "$actual" = "$model_sha" ] || {
    echo "SHA256 mismatch: expected $model_sha, got $actual" >&2; exit 6; }
}
expected_sha() {
  [ -f "$manifest" ] || return 1
  awk -v name="$name" '$2==name || $2==("*" name) {print $1; exit}' "$manifest"
}
expected=$(expected_sha || true)
if [ -z "$model_file" ] && [ -z "$model_url" ] && [ -f "$output" ] && \
   [ -n "$expected" ] && [ "$(sha_file "$output")" = "$expected" ]; then
  echo "cache hit: $output ($expected)"
  exit 0
fi
if [ "$offline" = yes ] && [ -z "$model_file" ]; then
  echo "offline mode: verified cached model or --model-file is required" >&2
  exit 4
fi

# A caller-supplied, already converted RKNN is architecture-independent.
case "$model_file" in
  *.rknn)
    if [ "$dry_run" = yes ]; then echo "DRY-RUN: copy verified prebuilt RKNN $model_file to $output"; exit 0; fi
    mkdir -p "$output_dir"
    if [ ! "$model_file" -ef "$output" ]; then cp "$model_file" "$output"; fi
    actual=$(sha_file "$output")
    check_supplied_sha "$actual"; record_sha "$actual"
    echo "prepared: $output ($actual)"; exit 0 ;;
esac

arch=${RKNN_TEST_ARCH:-$(uname -m)}
case "$arch" in x86_64|amd64) ;; *)
  if [ -z "$builder_host" ]; then
    echo "RKNN Toolkit 2.3.2 conversion is x86_64-only; this host is $arch." >&2
    echo "Run on an x86_64 workstation: ./deploy.sh --platform $platform --device DEVICE --accept-upstream-license" >&2
    echo "Or delegate explicitly: ./prepare_model.sh --platform $platform --builder-host X86_FLEET_DEVICE --accept-upstream-license" >&2
    exit 5
  fi
  fleet=${FLEET_BIN:-$HOME/.rpty/bin/fleet}
  work=/tmp/fall-detection-rknn-builder-$platform
  remote_model_arg=
  [ -z "$model_url" ] || remote_model_arg="--model-url $model_url"
  if [ -n "$model_file" ]; then remote_model_arg="--model-file $work/input/$(basename "$model_file")"; fi
  [ -z "$model_sha" ] || remote_model_arg="$remote_model_arg --model-sha256 $model_sha"
  if [ "$dry_run" = yes ]; then
    echo "DRY-RUN: $fleet exec $builder_host -- mkdir -p $work/input $work/output"
    echo "DRY-RUN: fleet push scripts${model_file:+ and model} to $builder_host:$work"
    echo "DRY-RUN: fleet exec $builder_host -- $work/prepare_model.sh --platform $platform --output-dir $work/output --accept-upstream-license $remote_model_arg"
    echo "DRY-RUN: fleet pull $builder_host:$work/output/$name $output"
    exit 0
  fi
  "$fleet" exec "$builder_host" -- mkdir -p "$work/input" "$work/output" "$work/tools"
  "$fleet" push "$builder_host" "$script_dir/prepare_model.sh" "$work/prepare_model.sh"
  "$fleet" push "$builder_host" "$script_dir/Dockerfile.model-builder" "$work/Dockerfile.model-builder"
  "$fleet" push "$builder_host" "$script_dir/tools/convert_pose_rknn.py" "$work/tools/convert_pose_rknn.py"
  "$fleet" push "$builder_host" "$script_dir/tools/export_pose_rawhead_onnx.py" "$work/tools/export_pose_rawhead_onnx.py"
  if [ -n "$model_file" ]; then "$fleet" push "$builder_host" "$model_file" "$work/input/$(basename "$model_file")"; fi
  set -- sh "$work/prepare_model.sh" --platform "$platform" --output-dir "$work/output" --accept-upstream-license
  [ -z "$model_url" ] || set -- "$@" --model-url "$model_url"
  [ -z "$model_file" ] || set -- "$@" --model-file "$work/input/$(basename "$model_file")"
  [ -z "$model_sha" ] || set -- "$@" --model-sha256 "$model_sha"
  "$fleet" exec --timeout 3600 "$builder_host" -- "$@"
  mkdir -p "$output_dir"
  "$fleet" pull "$builder_host" "$work/output/$name" "$output"
  actual=$(sha_file "$output"); check_supplied_sha "$actual"; record_sha "$actual"
  echo "prepared through x86_64 Fleet builder: $output ($actual)"
  exit 0
esac

if [ "$dry_run" = yes ]; then
  echo "DRY-RUN: prepare $output on x86_64"
  if [ -n "$model_file" ]; then echo "DRY-RUN: use local input $model_file";
  elif [ -n "$model_url" ]; then echo "DRY-RUN: download $model_url";
  else echo "DRY-RUN: export official yolo11n-pose.pt as nine raw heads"; fi
  echo "DRY-RUN: convert with RKNN Toolkit 2.3.2 target=$platform"
  exit 0
fi
mkdir -p "$output_dir"
input=$model_file
if [ -n "$model_url" ]; then
  input=$output_dir/model-download$url_suffix
  python3 - "$model_url" "$input" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
fi
case "$input" in
  *.rknn) cp "$input" "$output" ;;
  *)
    [ "$offline" = no ] || { echo "offline mode cannot build/pull the disposable builder" >&2; exit 4; }
    docker build -f "$script_dir/Dockerfile.model-builder" -t fall-detection-rknn-model-builder:2.3.2 "$script_dir"
    if [ -n "$input" ]; then
      case "$input" in *.onnx) ;; *) echo "model input must end in .rknn or .onnx" >&2; exit 2;; esac
      docker run --rm -v "$input":/input/model.onnx:ro -v "$output_dir":/output \
        fall-detection-rknn-model-builder:2.3.2 \
        "python /tools/convert_pose_rknn.py --onnx /input/model.onnx --platform $platform --out /output/$name"
    else
      docker run --rm -v "$output_dir":/output fall-detection-rknn-model-builder:2.3.2 \
        "python /tools/export_pose_rawhead_onnx.py --out /output/yolo11n_pose_rawhead.onnx && python /tools/convert_pose_rknn.py --onnx /output/yolo11n_pose_rawhead.onnx --platform $platform --out /output/$name"
    fi ;;
esac
actual=$(sha_file "$output")
check_supplied_sha "$actual"; record_sha "$actual"
echo "prepared: $output ($actual)"
