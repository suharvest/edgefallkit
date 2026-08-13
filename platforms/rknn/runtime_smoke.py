#!/usr/bin/env python3
"""Dependency and native-extension smoke for the final runtime image."""
import json

import cv2
import numpy as np
import rknn_postprocess

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)
factories = {name: Gst.ElementFactory.find(name) is not None
             for name in ("rtspsrc", "rtph264depay", "h264parse", "mppvideodec", "appsink")}
if not all(factories.values()):
    raise SystemExit(f"missing GStreamer factories: {factories}")
outputs = []
for grid in (8, 4, 2):
    outputs.extend((np.zeros((1, 64, grid, grid), np.float32),
                    np.zeros((1, 1, grid, grid), np.float32),
                    np.zeros((1, 51, grid, grid), np.float32)))
print(json.dumps({"ok": True, "numpy": np.__version__, "opencv": cv2.__version__,
                  "gstreamer": Gst.version_string(), "factories": factories,
                  "postprocess_detections": len(rknn_postprocess.decode_pose(outputs))}))
