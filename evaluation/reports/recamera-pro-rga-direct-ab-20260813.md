# reCamera Pro RGA direct-preprocess A/B — 2026-08-13

The optimized official-frame path preserves the original 1280×720 geometry
for OSD/result normalization while sending a separate 640×640 letterboxed RGB
model input. RGA performs NV12 aspect resize to 640×360 and color conversion;
Python only fills the gray border. Any RGA ABI or driver error permanently
latches the process back to the established full-resolution path.

| Path | WS FPS | App metrics FPS | Preprocess | Inference | Postprocess |
|---|---:|---:|---:|---:|---:|
| RGA direct model input | 18.13 | 17.5–18.9 | 0.0 ms | 42.0–45.3 ms | 1.9–2.4 ms |
| Full RGB + Python letterbox | 12.14 | 11.4–12.4 | 38.2–43.1 ms | 35.3–37.9 ms | 1.3–1.6 ms |

The direct path improved observed WebSocket throughput by 49.3% and removed
roughly 40 ms of Python preprocessing per frame. The higher inference envelope
on the faster path reflects increased NPU work cadence; total throughput still
improved substantially. Logs explicitly reported the RGA direct path with no
fallback.

The test used appMgr single-active switching. After A/B, the device was restored
to `retail-vision` (PID 1631); rkipc remained active and retail detections
continued. This is performance/geometry evidence, not accuracy evidence.
