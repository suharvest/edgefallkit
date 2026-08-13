#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tool="$root/tools/prepare_model.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/jetson-prepare-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

printf 'offline-test-input\n' > "$tmp/input.onnx"

bash -n "$tool"

if bash "$tool" --model-dir "$tmp/models" --dry-run > "$tmp/missing-license.out" 2>&1; then
  echo "expected missing license failure" >&2
  exit 1
fi
grep -q -- '--accept-upstream-license' "$tmp/missing-license.out"

bash "$tool" --offline --onnx "$tmp/input.onnx" --model-dir "$tmp/models" \
  --engine "$tmp/models/test.engine" --dry-run > "$tmp/offline.out"
grep -q 'would reuse existing ONNX' "$tmp/offline.out"
! grep -q 'uv run' "$tmp/offline.out"
! test -e "$tmp/models/test.engine"

bash "$tool" --offline --onnx "$tmp/input.onnx" --device orin-nano \
  --model-dir "$tmp/models" --engine "$tmp/models/test.engine" --dry-run > "$tmp/remote.out"
grep -q 'would push ONNX to orin-nano' "$tmp/remote.out"
grep -q 'host TRT10.3 trtexec (SM87)' "$tmp/remote.out"

echo "prepare_model shell tests passed"
