# Changelog

The public project name is **EdgeFallKit**. Existing `fall-detection-*` image
repositories and MQTT identifiers remain stable deployment interfaces.

All notable changes are recorded here. The project follows semantic versioning
for published runtime images and release candidates.

## 0.1.0-rc1 - 2026-08-13

- Added multi-stream, multi-person fall detection for Jetson Orin, RK3576,
  RK3588, Raspberry Pi 5 + Hailo-8, reCamera SG2002, and reCamera Pro.
- Standardized the reCamera-compatible MQTT v1 result contract.
- Added per-track temporal MLP and fall state with frozen evaluation reports.
- Published slim ARM64 runtime images for Jetson, RKNN, and Hailo platforms.
- Added hardware-accelerated RTSP paths and native pose postprocessing.
- Added explicit-license, cache-aware model preparation and deployment helpers.
- Added reproducible RTSP fixtures, performance evidence, checksums, and asset
  backup locations.

This is an engineering release candidate, not a medical or safety
certification.
