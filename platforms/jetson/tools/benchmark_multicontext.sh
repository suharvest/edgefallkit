#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 ENGINE OUTPUT_DIR [CONTEXTS...]" >&2
    exit 2
fi

engine=$1
output_dir=$2
shift 2
if [[ $# -eq 0 ]]; then
    contexts=(1 2 4 6)
else
    contexts=("$@")
fi
trtexec_bin=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
duration=${BENCH_DURATION:-15}
warmup=${BENCH_WARMUP:-2000}

mkdir -p "$output_dir"
test -r "$engine"
test -x "$trtexec_bin"

uname -a >"$output_dir/system.txt"
dpkg-query -W tensorrt-libs libnvinfer10 >>"$output_dir/system.txt" 2>&1 || true
nvpmodel -q >>"$output_dir/system.txt" 2>&1 || true
free -m >>"$output_dir/system.txt" 2>&1 || true
ps -eo pid,rss,comm,args --sort=-rss | head -20 >>"$output_dir/system.txt" 2>&1 || true

for count in "${contexts[@]}"; do
    case "$count" in
        ''|*[!0-9]*) echo "invalid context count: $count" >&2; exit 2 ;;
    esac
    run_dir="$output_dir/contexts-$count"
    mkdir -p "$run_dir"
    tegrastats --interval 1000 --logfile "$run_dir/tegrastats.log" >/dev/null 2>&1 &
    stats_pid=$!
    set +e
    "$trtexec_bin" \
        --loadEngine="$engine" \
        --infStreams="$count" \
        --duration="$duration" \
        --warmUp="$warmup" \
        --useCudaGraph \
        --noDataTransfers \
        --percentile=50,90,95,99 \
        >"$run_dir/trtexec.log" 2>&1 &
    infer_pid=$!
    while kill -0 "$infer_pid" >/dev/null 2>&1; do
        ps -p "$infer_pid" -o pid=,rss=,%cpu=,nlwp=,etime= >>"$run_dir/process.log" 2>/dev/null || true
        sleep 1
    done
    wait "$infer_pid"
    status=$?
    set -e
    kill "$stats_pid" >/dev/null 2>&1 || true
    wait "$stats_pid" >/dev/null 2>&1 || true
    printf '%s\n' "$status" >"$run_dir/exit-status.txt"
    if [[ $status -ne 0 ]]; then
        echo "contexts=$count failed with exit $status" >&2
        exit "$status"
    fi
done
