#!/bin/sh
set -eu

usage() {
  cat <<'EOF'
Usage: deploy.sh --platform rk3576|rk3588 --accept-upstream-license [options]
  --device DEVICE         Fleet target (defaults: cat-remote/radxa)
  --builder-host DEVICE   x86_64 Fleet builder when invoked from ARM
  --model-file FILE       prebuilt .rknn or raw-head .onnx
  --model-url HTTPS_URL   prebuilt .rknn or raw-head .onnx URL
  --model-sha256 SHA256   verify a caller-supplied model
  --models-dir DIR        model/profile cache (default: platform models directory)
  --offline               require cached local model and cached remote runtime
  --dry-run               print every action without changing local/remote state
  --no-up                 prepare, push and validate but do not compose up
EOF
}

platform='' device='' builder_host='' model_file='' model_url='' model_sha='' models_dir=''
accept=no offline=no dry_run=no up=yes
while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform) platform=$2; shift 2;; --device) device=$2; shift 2;;
    --builder-host) builder_host=$2; shift 2;; --model-file) model_file=$2; shift 2;;
    --model-url) model_url=$2; shift 2;; --accept-upstream-license) accept=yes; shift;;
    --model-sha256) model_sha=$2; shift 2;;
    --models-dir) models_dir=$2; shift 2;;
    --offline) offline=yes; shift;; --dry-run) dry_run=yes; shift;; --no-up) up=no; shift;;
    -h|--help) usage; exit 0;; *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$accept" = yes ] || { echo "license gate: pass --accept-upstream-license after reviewing upstream terms" >&2; exit 3; }
case "$platform" in rk3576) device=${device:-cat-remote}; home=/home/cat;; rk3588) device=${device:-radxa}; home=/home/radxa;; *) echo "--platform must be rk3576 or rk3588" >&2; exit 2;; esac
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
platform_dir=$(CDPATH='' cd -- "$script_dir/../$platform" && pwd)
models=${models_dir:-$platform_dir/models}
pose=yolo11n_pose_rawhead_fp16.$platform.rknn
temporal=temporal-$platform.npz
image=sensecraft-missionpack.seeed.cn/solution/fall-detection-rknn:0.1.0-rc3
digest=sha256:43d767f5927e6a4ebc00013c24ebd9f10c692c9aa0d7615520a4823d6367ffa8
fleet=${FLEET_BIN:-$HOME/.rpty/bin/fleet}

set -- sh "$script_dir/prepare_model.sh" --platform "$platform" --output-dir "$models" --accept-upstream-license
[ -z "$builder_host" ] || set -- "$@" --builder-host "$builder_host"
[ -z "$model_file" ] || set -- "$@" --model-file "$model_file"
[ -z "$model_url" ] || set -- "$@" --model-url "$model_url"
[ -z "$model_sha" ] || set -- "$@" --model-sha256 "$model_sha"
[ "$offline" = no ] || set -- "$@" --offline
[ "$dry_run" = no ] || set -- "$@" --dry-run
"$@"

[ "$dry_run" = yes ] || { [ -f "$models/$pose" ] && [ -f "$models/$temporal" ]; } || {
  echo "required artifacts missing: $models/$pose or $models/$temporal" >&2; exit 7; }
verify_local() {
  artifact=$1
  expected=$(awk -v name="$artifact" '$2==name || $2==("*" name) {print $1; exit}' "$models/SHA256SUMS")
  [ -n "$expected" ] || { echo "manifest has no entry for $artifact" >&2; exit 7; }
  if command -v sha256sum >/dev/null 2>&1; then actual=$(sha256sum "$models/$artifact" | awk '{print $1}')
  else actual=$(shasum -a 256 "$models/$artifact" | awk '{print $1}'); fi
  [ "$actual" = "$expected" ] || { echo "manifest mismatch for $artifact: $actual != $expected" >&2; exit 7; }
}
if [ "$dry_run" = no ]; then verify_local "$pose"; verify_local "$temporal"; fi
remote=$home/fall-detection/platforms/$platform
run() { if [ "$dry_run" = yes ]; then echo "DRY-RUN: $*"; else "$@"; fi; }
run "$fleet" exec "$device" -- mkdir -p "$remote/models" "$remote/config"
push_if_changed() {
  source=$1 destination=$2
  if command -v sha256sum >/dev/null 2>&1; then local_sha=$(sha256sum "$source" | awk '{print $1}')
  else local_sha=$(shasum -a 256 "$source" | awk '{print $1}'); fi
  if [ "$dry_run" = yes ]; then
    echo "DRY-RUN: verify remote SHA256 or fleet push $device $source $destination"; return
  fi
  remote_sum=$("$fleet" exec "$device" -- sha256sum "$destination" 2>/dev/null || true)
  case "$remote_sum" in "$local_sha "*) echo "remote cache hit: $destination ($local_sha)";;
    *) "$fleet" push "$device" "$source" "$destination";; esac
}
push_if_changed "$models/$pose" "$remote/models/$pose"
push_if_changed "$models/$temporal" "$remote/models/$temporal"
run "$fleet" push "$device" "$platform_dir/config/config.json" "$remote/config/config.json"
run "$fleet" push "$device" "$platform_dir/docker-compose.yml" "$remote/docker-compose.yml"
if [ "$offline" = yes ]; then
  run "$fleet" exec "$device" -- docker image inspect "$image"
else
  run "$fleet" exec --timeout 1200 "$device" -- docker pull "$image"
fi
if [ "$dry_run" = yes ]; then
  echo "DRY-RUN: verify RepoDigests contains $digest"
else
  image_metadata=$("$fleet" exec "$device" -- docker image inspect "$image")
  case "$image_metadata" in *"$digest"*) ;; *) echo "remote RepoDigest mismatch" >&2; exit 8;; esac
fi
run "$fleet" exec "$device" -- env FALL_RK_IMAGE="$image" docker compose -f "$remote/docker-compose.yml" config --quiet
run "$fleet" exec "$device" -- env FALL_RK_IMAGE="$image" docker compose -f "$remote/docker-compose.yml" run --rm --no-deps fall-detection app.py --config /config/config.json --validate
if [ "$up" = yes ]; then
  run "$fleet" exec "$device" -- env FALL_RK_IMAGE="$image" docker compose -f "$remote/docker-compose.yml" up -d fall-detection
fi
if [ "$dry_run" = yes ]; then echo "dry-run complete: $device $platform $image@$digest"
else echo "deployment ready: $device $platform $image@$digest"; fi
