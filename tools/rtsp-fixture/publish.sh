#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 VIDEO [RTSP_HOST=127.0.0.1] [PATH=fall-e2e-low]" >&2
    exit 2
fi

video=$1
host=${2:-127.0.0.1}
path=${3:-fall-e2e-low}

exec ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$video" \
    -an \
    -vf 'fps=15,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:-1:-1:color=black' \
    -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline \
    -level 3.1 -b:v 1200k -maxrate 1200k -bufsize 1200k \
    -g 30 -keyint_min 30 -bf 0 \
    -x264-params 'repeat-headers=1:scenecut=0' -pix_fmt yuv420p \
    -f rtsp -rtsp_transport tcp "rtsp://${host}:8554/${path}"
