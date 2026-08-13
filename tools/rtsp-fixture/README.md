# Reproducible RTSP fixture

This is the LAN test source used for end-to-end decoder, accelerator,
post-processing, tracking, temporal state, and MQTT verification. It publishes
H.264 Constrained Baseline, 640×640 at 15 FPS, approximately 1.2 Mbit/s, no B
frames, and a two-second GOP over RTSP/TCP.

On the LAN source machine:

```bash
docker compose up -d
chmod +x publish.sh
./publish.sh /path/to/video.mp4
```

The client URL is `rtsp://SOURCE_LAN_IP:8554/fall-e2e-low`. To exercise the
positive person/fall path without modifying the training split, publish a
frozen-test clip under the second configured path:

```bash
./publish.sh /data/gmdcsa24/subject-4/Fall/01.mp4 127.0.0.1 fall-person
```

Subject 4 is for testing only and must never be read by development/training
commands. The source currently used on Spark is `192.168.3.42`; deployment
documentation should not assume that address outside the test LAN.

Verify actual DESCRIBE/readability, not merely the open TCP port:

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,profile,width,height,r_frame_rate \
  -of compact rtsp://127.0.0.1:8554/fall-e2e-low
```
