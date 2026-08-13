# Model assets and provenance

Published runtime images intentionally contain no pose model. Model preparation
is explicit, cached, checksum-aware, and performed by `deploy.sh` or the
platform helper it dispatches to.

## License boundary

The repository's Apache-2.0 license applies to original project code and
documentation. It does not cover models downloaded by the helpers.

- Ultralytics YOLO11 Pose weights and their ONNX, TensorRT, and RKNN conversions
  retain their upstream AGPL-3.0 or Enterprise provenance.
- The Hailo Model Zoo HEF retains Hailo's upstream terms and is downloaded from
  Hailo's official fixed URL; this project does not re-host it.
- Dataset downloads retain their dataset terms.

Passing `--accept-upstream-license` confirms that the operator reviewed the
applicable terms. It does not grant redistribution or commercial rights.

## What is published

- Slim ARM64 runtime images in the Seeed Registry.
- Original source, configuration, MQTT schema, deployment helpers, evaluation
  reports, and small project-owned temporal profiles/source headers.

## What is not published by this project

- Ultralytics weights, ONNX exports, RKNN files, or TensorRT engines.
- Hailo's HEF.
- GMDCSA-24 or RealBiomFall videos.
- Vendor SDKs, host drivers, or firmware.

## Manifests and caching

Jetson records the upstream/export version, source/ONNX SHA, input/profile,
target device, SM, TensorRT and engine SHA. RK records the target-specific RKNN
SHA alongside the temporal profile. Hailo pins the official HEF SHA directly in
the downloader. A cache is accepted only when its recorded digest and target
parameters match.

Internal benchmark assets and durable backups are indexed in
`assets/ASSET_LOCATIONS.md`; those locations are provenance records, not public
redistribution URLs.
