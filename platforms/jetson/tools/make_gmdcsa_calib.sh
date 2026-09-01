#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 GMDCSA_VIDEO_ROOT OUTPUT_DIR [IMAGE_COUNT]" >&2
  exit 2
fi

source_root=${1%/}
output_dir=${2%/}
image_count=${3:-64}
frames_dir=$output_dir/frames
candidate_manifest=$output_dir/candidates.txt
calibration_manifest=$output_dir/calibration.txt

[[ -d "$source_root" ]] || { echo "Video root not found: $source_root" >&2; exit 1; }
[[ "$image_count" =~ ^[1-9][0-9]*$ ]] || { echo "IMAGE_COUNT must be positive" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

mkdir -p "$frames_dir"
if find "$frames_dir" -type f -print -quit | grep -q .; then
  echo "Refusing to mix calibration runs; frames directory is not empty: $frames_dir" >&2
  exit 1
fi

find "$source_root" -type f -name '*.mp4' | sort > "$output_dir/source-videos.txt"
video_count=$(wc -l < "$output_dir/source-videos.txt")
[[ "$video_count" -gt 0 ]] || { echo "No MP4 videos found under $source_root" >&2; exit 1; }

index=0
while IFS= read -r src; do
  relative=${src#"$source_root"/}
  stem=${relative%.mp4}
  stem=${stem//\//_}
  ffmpeg -loglevel error -y -ss 0 -i "$src" -frames:v 1 "$frames_dir/${stem}_t0.jpg"
  if [[ "$index" -lt 80 ]]; then
    ffmpeg -loglevel error -y -ss 1 -i "$src" -frames:v 1 "$frames_dir/${stem}_t1.jpg"
  fi
  index=$((index + 1))
done < "$output_dir/source-videos.txt"

find "$frames_dir" -type f -name '*.jpg' | sort > "$candidate_manifest"
candidate_count=$(wc -l < "$candidate_manifest")
if [[ "$candidate_count" -lt "$image_count" ]]; then
  echo "Need $image_count candidates, extracted $candidate_count" >&2
  exit 1
fi

# Select evenly across the sorted candidate list. For the 2026-09-01 run this
# reproduces the 64 indices 1,5,9,...,238 selected from 240 candidates.
awk -v count="$image_count" -v total="$candidate_count" '
  BEGIN {
    for (i = 0; i < count; i++) {
      selected[int((i * total + count - 1) / count) + 1] = 1
    }
  }
  selected[NR]
' "$candidate_manifest" > "$calibration_manifest"

[[ "$(wc -l < "$calibration_manifest")" -eq "$image_count" ]]
while IFS= read -r image; do sha256sum "$image"; done \
  < "$calibration_manifest" > "$output_dir/calibration-images.sha256"
sha256sum "$calibration_manifest" "$output_dir/calibration-images.sha256"
echo "videos=$video_count candidates=$candidate_count calibration_images=$image_count"
