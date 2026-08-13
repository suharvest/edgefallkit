FROM scratch AS artifact
COPY platforms/rpi-hailo/build/fall-hailo /fall-hailo

FROM debian:trixie-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-libav libmosquitto1 && rm -rf /var/lib/apt/lists/*
COPY --from=artifact /fall-hailo /usr/local/bin/fall-hailo
ENTRYPOINT ["/usr/local/bin/fall-hailo"]
