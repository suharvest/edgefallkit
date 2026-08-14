# The builder exists only to obtain h264parse. Installing
# gstreamer1.0-plugins-bad in the runtime stage pulls ~300 packages and doubles
# the image (533 MB -> 1.07 GB) for one element; even the shared-library
# package alone drags in libllvm19 + mesa + flite (~207 MB) via libgstgl.
# h264parse lives in libgstvideoparsersbad.so, whose only dependency outside
# gstreamer1.0-plugins-base is libgstcodecparsers. Both are copied directly.
FROM debian:trixie-slim AS gstbad
RUN apt-get update && apt-get install -y --no-install-recommends \
    gstreamer1.0-plugins-bad && rm -rf /var/lib/apt/lists/*

FROM debian:trixie-slim AS runtime
# The RTSP branch in main.cpp is rtspsrc ! rtph264depay ! h264parse ! decodebin.
# Without h264parse gst_parse_launch drops the link with `no element
# "h264parse"` and the process exits 139 — and since RTSP is the default
# STREAMS value, the shipped compose cannot ingest a camera at all. test:// and
# file:// sources do not hit this path, which is why synthetic benchmarks passed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-libav libmosquitto1 && rm -rf /var/lib/apt/lists/*
COPY --from=gstbad /usr/lib/aarch64-linux-gnu/libgstcodecparsers-1.0.so.0* \
     /usr/lib/aarch64-linux-gnu/
COPY --from=gstbad /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstvideoparsersbad.so \
     /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstvideoparsersbad.so
COPY platforms/rpi-hailo/build/fall-hailo /usr/local/bin/fall-hailo
ENTRYPOINT ["/usr/local/bin/fall-hailo"]
