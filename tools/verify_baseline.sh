#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

compose_files=(
    "$root/platforms/jetson/docker-compose.yml"
    "$root/platforms/jetson/docker-compose.slim.yml"
    "$root/platforms/rk3576/docker-compose.yml"
    "$root/platforms/rk3588/docker-compose.yml"
    "$root/platforms/rpi-hailo/docker-compose.yml"
    "$root/tools/rtsp-fixture/docker-compose.yml"
)
for file in "${compose_files[@]}"; do
    docker compose -f "$file" config --quiet
done

python3 -m json.tool "$root/contracts/mqtt-result.schema.json" >/dev/null
python3 -m json.tool "$root/release/0.1.0-rc1.json" >/dev/null
python3 -m json.tool "$root/platforms/jetson/config/config.json" >/dev/null
python3 -m json.tool "$root/platforms/rk3576/config/config.json" >/dev/null
python3 -m json.tool "$root/platforms/rk3588/config/config.json" >/dev/null
python3 "$root/contracts/validate_payload.py" \
    "$root/platforms/rpi-hailo/tests/fixtures/recamera_payload.json"

sh -n "$root/platforms/rpi-hailo/scripts/fetch_model.sh"
bash -n "$root/platforms/jetson/tools/build_engine.sh"
bash -n "$root/tools/rtsp-fixture/publish.sh"

(
    cd "$root/evaluation/reports"
    sha256sum -c SHA256SUMS
)

python3 - "$root" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
broken = []
for document in root.rglob("*.md"):
    text = document.read_text(encoding="utf-8", errors="replace")
    for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        path = (document.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append((document.relative_to(root), target))
if broken:
    for document, target in broken:
        print(f"broken Markdown link: {document} -> {target}", file=sys.stderr)
    raise SystemExit(1)
print("Markdown links passed")
PY

if rg -n -i 'batch[ =_-]*8' "$root" --glob '*.md' --glob '*.json'; then
    echo "batch 8 remains in delivery documentation" >&2
    exit 1
fi

echo "baseline verification passed"
