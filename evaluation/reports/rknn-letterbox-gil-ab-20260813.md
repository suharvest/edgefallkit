# RK MPP letterbox and C++ postprocess A/B — 2026-08-13

The production path now decodes and aspect-resizes in `mppvideodec`/RGA, then performs the only required mapped-buffer copy directly into a 640×640 RGB canvas padded with value 114. The mapped Gst buffer never escapes its map lifetime. OpenCV/FFmpeg and NumPy postprocess remain fallbacks.

The 1280×720 Spark RTSP stream is the decisive geometry test: the previous forced 640×640 pipeline failed negotiation on both boards; the new pipeline negotiated 640×360 and produced the expected 140-pixel top/bottom padding. Unit parity also covers 640×480 (640×480 + 80-pixel padding) and 640×640.

| Device | 720p contract | Video CPU ms/frame | Video RSS | C++ 2-context throughput old → new | MQTT strict contract | E2E inference / pipeline mean |
|---|---:|---:|---:|---:|---:|---:|
| RK3576 | PASS | 14.744 | 71.4 MiB | 29.60 → 29.42 fps | 200/200 | 62.47 / 63.09 ms |
| RK3588 | PASS | 7.030 | 69.9 MiB | 37.34 → 38.16 fps | 200/200 | 54.18 / 54.45 ms |

RK3588 retained `openvoicestream`, `ovs-llm`, `ovs-agent`, and `wyoming-slv`; its p95/max values therefore represent coexistence, not an isolated ceiling. The 15 fps video FPS is source-limited. The minimal MQTT collector includes startup in its observed rate, so use `inference_ms`/`pipeline_ms` for platform comparison. Exact machine-readable values are in each platform's `*-letterbox-gil-ab-20260813.json` result.
