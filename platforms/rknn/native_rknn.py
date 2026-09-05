"""Optional RKNN/RGA adapter used by the ARM64 native profile.

RC7 supplies ``libhybrid_rga_rknn.so`` in the image and permits an explicit
``RKNN_NATIVE_LIBRARY`` override. This module checks the bridge ABI and never
pretends that x86 or a missing library can execute the native path.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

EXPECTED_POSE_SHAPES = tuple((1, c, s, s) for s in (80, 40, 20) for c in (64, 1, 51))


def validate_pose_outputs(outputs):
    if outputs is None or len(outputs) != 9:
        raise RuntimeError(f"native pose requires exactly 9 outputs, got {0 if outputs is None else len(outputs)}")
    arrays = []
    expected_shapes = sorted(EXPECTED_POSE_SHAPES)
    actual_shapes = sorted(tuple(getattr(value, "shape", ())) for value in outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"native pose output shapes mismatch: {actual_shapes}")
    for value in outputs:
        arrays.append(np.ascontiguousarray(value, dtype=np.float32))
    return arrays


def load_host_runtime():
    """Check that both host ABI libraries are loadable, without bundling them."""
    try:
        rknn = ctypes.CDLL("librknnrt.so")
        rga = ctypes.CDLL("librga.so.2")
    except OSError as exc:
        raise RuntimeError(f"host native runtime unavailable: {exc}") from exc
    return rknn, rga


class NativeRuntime:
    """ctypes bridge to the board-built RKNN/RGA hot-path library."""

    def __init__(self, model, core_mask=None, library=None):
        library = library or os.environ.get(
            "RKNN_NATIVE_LIBRARY", "/opt/fall-detection/libhybrid_rga_rknn.so")
        self.lib = ctypes.CDLL(library)
        self.lib.hybrid_last_error.restype = ctypes.c_char_p
        self.lib.hybrid_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        self.lib.hybrid_create.restype = ctypes.c_void_p
        self.lib.hybrid_model_width.argtypes = [ctypes.c_void_p]
        self.lib.hybrid_model_width.restype = ctypes.c_int
        self.lib.hybrid_model_height.argtypes = [ctypes.c_void_p]
        self.lib.hybrid_model_height.restype = ctypes.c_int
        self.lib.hybrid_output_count.argtypes = [ctypes.c_void_p]
        self.lib.hybrid_output_count.restype = ctypes.c_uint32
        self.lib.hybrid_output_ndims.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.hybrid_output_ndims.restype = ctypes.c_uint32
        self.lib.hybrid_output_dim.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.hybrid_output_dim.restype = ctypes.c_uint32
        self.lib.hybrid_output_elems.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.hybrid_output_elems.restype = ctypes.c_uint32
        self.lib.hybrid_infer_pose_nv12_fd.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        self.lib.hybrid_infer_pose_nv12_fd.restype = ctypes.c_int
        self.lib.hybrid_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.hybrid_create(str(model).encode(), int(core_mask or 0))
        if not self.handle: raise RuntimeError(self.error())
        try:
            self.width = self.lib.hybrid_model_width(self.handle)
            self.height = self.lib.hybrid_model_height(self.handle)
            count = self.lib.hybrid_output_count(self.handle)
            self.shapes = []
            for index in range(count):
                ndims = self.lib.hybrid_output_ndims(self.handle, index)
                self.shapes.append(tuple(self.lib.hybrid_output_dim(self.handle, index, d)
                                         for d in range(ndims)))
            self.sizes = [self.lib.hybrid_output_elems(self.handle, i) for i in range(count)]
            self.flat = np.empty(sum(self.sizes), dtype=np.float32)
            validate_pose_outputs([np.empty(shape, np.float32) for shape in self.shapes])
        except BaseException:
            self.lib.hybrid_destroy(self.handle); self.handle = None
            raise

    def error(self):
        value = self.lib.hybrid_last_error()
        return value.decode(errors="replace") if value else "unknown native error"

    def infer(self, frame):
        if frame.closed or len(frame.planes) != 2:
            raise RuntimeError("native frame is closed or lacks two NV12 planes")
        y, uv = frame.planes
        if (y["fd"] != uv["fd"] or y["offset"] != 0 or
                uv["offset"] != y["stride"] * frame.visible_height or
                y["stride"] != uv["stride"]):
            raise RuntimeError("native backend requires contiguous single-fd NV12")
        if (frame.visible_width, frame.visible_height) != (self.width, self.height):
            raise RuntimeError("native backend currently requires 640x640 decoded NV12")
        rga_ms = ctypes.c_double(); rknn_ms = ctypes.c_double()
        started = time.perf_counter()
        ret = self.lib.hybrid_infer_pose_nv12_fd(
            self.handle, int(y["fd"]), frame.visible_width, frame.visible_height,
            int(y["stride"]), ctypes.byref(rga_ms), ctypes.byref(rknn_ms),
            self.flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.flat.size)
        if ret: raise RuntimeError(f"native pose inference {ret}: {self.error()}")
        # Borrowed views are safe because each context is owned by one worker
        # thread and pose decoding completes synchronously before its next run.
        # Avoiding nine extra copies removes ~3.9 MiB/frame of host traffic.
        outputs = []; offset = 0
        for shape, size in zip(self.shapes, self.sizes):
            outputs.append(self.flat[offset:offset + size].reshape(shape))
            offset += size
        return validate_pose_outputs(outputs), (time.perf_counter() - started) * 1000.0

    def close(self):
        if self.handle:
            self.lib.hybrid_destroy(self.handle); self.handle = None
