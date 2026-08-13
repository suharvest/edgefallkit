#!/bin/sh
set -eu
BASE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
echo "fetch_model.sh is retained for compatibility; use deploy.sh for deployment" >&2
exec "$BASE/deploy.sh" "$@" --prepare-only
