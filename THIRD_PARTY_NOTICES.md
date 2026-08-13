# Third-party assets and licenses

The repository's Apache-2.0 license covers original source code and
documentation in this project. It does not relicense third-party software,
models, datasets, firmware, SDKs, or the contents of linked repositories.

## Pose models

Pose models are intentionally excluded from Git and published runtime images.
The deployment helpers download them from their upstream provider or consume a
local file only after the user explicitly acknowledges the upstream terms.

- Ultralytics YOLO11 Pose weights and artifacts derived from them remain subject
  to the Ultralytics AGPL-3.0 or Enterprise terms. TensorRT and RKNN conversion
  does not change that provenance.
- The Hailo-8 YOLOv8s-Pose HEF is downloaded from the Hailo Model Zoo's fixed
  official URL and verified by SHA256. This project does not redistribute it or
  claim a right to relicense it.

See the platform deployment guide before enabling model download. The
`--accept-upstream-license` flag records an explicit acknowledgement; it does
not grant additional rights.

## Datasets

GMDCSA-24 and RealBiomFall are not distributed by this repository. Their
download helpers preserve the official source and checksums. Users must review
and comply with each dataset's terms.

## Linked reCamera repositories

`platforms/recamera-sg2002` and `platforms/recamera-pro/*` are relative symbolic
links to sibling canonical repositories. Their contents retain the licenses
and notices of those repositories and are not relicensed by this project.

## Vendor runtimes

TensorRT, CUDA, RKNN Toolkit/Lite, Rockchip MPP/RGA, HailoRT, GStreamer, and
device firmware are supplied by their respective vendors or Linux
distributions. Runtime images either install distribution packages or mount
version-matched host libraries as documented by each platform.
