#!/bin/sh
set -eu

BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR=${FALL_HAILO_MODEL_DIR:-"$BASE/models"}
COMPOSE_FILE=${FALL_HAILO_COMPOSE_FILE:-"$BASE/docker-compose.yml"}
DOCKER=${FALL_HAILO_DOCKER_BIN:-docker}
CURL=${FALL_HAILO_CURL_BIN:-curl}
MODEL_URL=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef
EXPECTED_SHA256=e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d
MODEL_PATH=$MODEL_DIR/yolov8s_pose.hef
SHA_PATH=$MODEL_DIR/yolov8s_pose.hef.sha256

accepted=0
offline=0
dry_run=0
prepare_only=0
local_hef=

usage() {
    cat <<'EOF'
Usage: ./deploy.sh --accept-upstream-license [options]

Required acknowledgement:
  --accept-upstream-license  Accept the upstream terms applicable to the Hailo HEF.

Options:
  --hef PATH                 Install a local/offline HEF instead of downloading.
  --offline                  Forbid model and image network access; use local HEF/cache/image.
  --dry-run                  Print the validated deployment plan without writing or starting.
  --prepare-only             Prepare and verify the HEF without pulling or starting containers.
  -h, --help                 Show this help.

The acknowledgement records consent; it does not grant redistribution rights.
The HEF is downloaded only from the fixed official Hailo URL and is never baked
into the runtime image.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --accept-upstream-license) accepted=1 ;;
        --hef)
            [ "$#" -ge 2 ] || { echo "--hef requires a path" >&2; exit 2; }
            local_hef=$2
            shift
            ;;
        --offline) offline=1 ;;
        --dry-run) dry_run=1 ;;
        --prepare-only) prepare_only=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ "$accepted" -ne 1 ]; then
    echo "refusing deployment: pass --accept-upstream-license after reviewing upstream Hailo model terms" >&2
    exit 2
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

verify_hef() {
    actual=$(sha256_file "$1")
    if [ "$actual" != "$EXPECTED_SHA256" ]; then
        echo "HEF checksum mismatch: expected $EXPECTED_SHA256, got $actual ($1)" >&2
        return 1
    fi
}

if [ "$dry_run" -eq 1 ]; then
    echo "DRY RUN: upstream license acknowledgement supplied"
    if [ -n "$local_hef" ]; then
        echo "DRY RUN: verify local HEF $local_hef -> $MODEL_PATH"
    elif [ -f "$MODEL_PATH" ] && verify_hef "$MODEL_PATH" >/dev/null 2>&1; then
        echo "DRY RUN: use verified cached HEF $MODEL_PATH"
    elif [ "$offline" -eq 1 ]; then
        echo "DRY RUN ERROR: offline mode requires --hef or a verified cache" >&2
        exit 1
    else
        echo "DRY RUN: download $MODEL_URL -> $MODEL_PATH atomically and verify $EXPECTED_SHA256"
    fi
    if [ "$prepare_only" -eq 1 ]; then
        echo "DRY RUN: model preparation only; no container action"
    elif [ "$offline" -eq 1 ]; then
        echo "DRY RUN: $DOCKER compose -f $COMPOSE_FILE config --quiet"
    else
        echo "DRY RUN: $DOCKER compose -f $COMPOSE_FILE pull fall-detection"
        echo "DRY RUN: $DOCKER compose -f $COMPOSE_FILE config --quiet"
    fi
    if [ "$prepare_only" -eq 0 ]; then
        echo "DRY RUN: $DOCKER compose -f $COMPOSE_FILE up -d --pull never --no-build fall-detection"
    fi
    exit 0
fi

mkdir -p "$MODEL_DIR"
tmp=$MODEL_DIR/.yolov8s_pose.hef.tmp.$$
sha_tmp=$SHA_PATH.tmp.$$
cleanup() { rm -f "$tmp" "$sha_tmp"; }
trap cleanup EXIT HUP INT TERM

if [ -n "$local_hef" ]; then
    [ -f "$local_hef" ] || { echo "local HEF not found: $local_hef" >&2; exit 1; }
    verify_hef "$local_hef"
    if [ "$local_hef" != "$MODEL_PATH" ]; then
        cp "$local_hef" "$tmp"
        verify_hef "$tmp"
        chmod 0644 "$tmp"
        mv -f "$tmp" "$MODEL_PATH"
    fi
    echo "installed verified local HEF: $MODEL_PATH"
elif [ -f "$MODEL_PATH" ] && verify_hef "$MODEL_PATH"; then
    echo "verified HEF cache hit: $MODEL_PATH"
elif [ "$offline" -eq 1 ]; then
    echo "offline mode requires --hef PATH or a verified cache at $MODEL_PATH" >&2
    exit 1
else
    "$CURL" --fail --location --retry 3 --connect-timeout 20 --output "$tmp" "$MODEL_URL"
    verify_hef "$tmp"
    chmod 0644 "$tmp"
    mv -f "$tmp" "$MODEL_PATH"
    echo "downloaded and verified official HEF: $MODEL_PATH"
fi
printf '%s  %s\n' "$EXPECTED_SHA256" yolov8s_pose.hef > "$sha_tmp"
mv -f "$sha_tmp" "$SHA_PATH"

if [ "$prepare_only" -eq 1 ]; then
    echo "model preparation complete; containers were not changed"
    exit 0
fi

if [ "$offline" -eq 0 ]; then
    "$DOCKER" compose -f "$COMPOSE_FILE" pull fall-detection
fi
"$DOCKER" compose -f "$COMPOSE_FILE" config --quiet
"$DOCKER" compose -f "$COMPOSE_FILE" up -d --pull never --no-build fall-detection
echo "deployment started; validate health and MQTT output before acceptance"
