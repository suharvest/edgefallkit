# Demo asset provenance

## `recamera-pro-live-fall-demo.gif`

- UI: production reCamera Pro App Center `Live Preview` page.
- Video: GMDCSA-24 v2.1, `subject-4/Fall/01.mp4`.
- Upstream: <https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos>
- Upstream repository license: MIT. Cite the GMDCSA-24 dataset and paper as
  requested by its authors.
- Source video SHA-256: `be235fef9bda633f5f63ccbc28c1dbb6c0138728c5b1696d86f7a6e0c66f4516`.
- Pose overlay: matching RV1126B/yolo11n-pose trace from
  `recamera-pro-rv1126b-gmdcsa-traces-20260813.tar.gz`, archived on Spark at
  `/home/harvest/datasets/fall-detection/traces/recamera-pro/20260813/`.
- Trace archive SHA-256: `886dcde759c9a949073beecc1d06c0785f6e02b77bb3e1bf256192311743bf23`.
- GIF SHA-256: `005893ab60a7085734827c1cc05b4f87a07e911c65593231707720f734d0bbd9`.

The state labels are replayed to demonstrate the panel transition and are not
evaluation evidence. Frozen accuracy and device-performance claims live under
`evaluation/`.

## `recamera-pro-debug-fall-demo.gif`

This is a synthetic WebSocket-result injection into the production Debug page;
it contains no source video. SHA-256:
`907417b72823c426d12be3d107b1c4b540734837ef7fb8d280d4d201d86dfef0`.
