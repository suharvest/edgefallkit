"""Native RKNN Lite pose inference with C++/NumPy pose decode fallback."""
from __future__ import annotations

import time
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _nms(boxes, scores, threshold):
    order = scores.argsort()[::-1]; keep = []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= threshold]
    return keep


def _nchw(output):
    a = np.asarray(output)
    if a.ndim == 3: a = a[None]
    channels = (1, 51, 64)
    if a.ndim == 4 and a.shape[1] not in channels and a.shape[-1] in channels:
        a = a.transpose(0, 3, 1, 2)
    return a


def decode_pose_numpy(outputs, confidence=0.35, nms_threshold=0.45, input_size=640):
    outs = [_nchw(x) for x in outputs]
    groups = {1: [], 51: [], 64: []}
    for x in outs:
        if x.ndim == 4 and x.shape[1] in groups: groups[x.shape[1]].append(x)
    for values in groups.values(): values.sort(key=lambda x: -x.shape[-1])
    boxes, scores_all, points = [], [], []
    proj = np.arange(16, dtype=np.float32)
    for bd, cl, kp in zip(groups[64], groups[1], groups[51]):
        _, _, h, w = bd.shape; stride = input_size // h; n = h * w
        score = cl.reshape(-1, n).max(0).astype(np.float32)
        if score.min(initial=0) < 0 or score.max(initial=0) > 1: score = _sigmoid(score)
        selected = np.flatnonzero(score >= confidence)
        if not selected.size: continue
        gx = (selected % w).astype(np.float32); gy = (selected // w).astype(np.float32)
        dist = (_softmax(bd.reshape(4, 16, n)[:, :, selected].astype(np.float32)) * proj[None, :, None]).sum(1)
        boxes.append(np.stack(((gx + .5 - dist[0]) * stride, (gy + .5 - dist[1]) * stride,
                               (gx + .5 + dist[2]) * stride, (gy + .5 + dist[3]) * stride), 1))
        raw = kp.reshape(17, 3, n)[:, :, selected].astype(np.float32)
        x = (raw[:, 0] * 2 + gx[None] - .5) * stride
        y = (raw[:, 1] * 2 + gy[None] - .5) * stride
        c = _sigmoid(raw[:, 2])
        points.append(np.stack((x, y, c), -1).transpose(1, 0, 2)); scores_all.append(score[selected])
    if not boxes: return []
    boxes = np.concatenate(boxes); scores = np.concatenate(scores_all); points = np.concatenate(points)
    return [{"box": boxes[i].clip(0, input_size).astype(float).tolist(), "score": float(scores[i]),
             "keypoints": points[i].astype(float).tolist()}
            for i in _nms(boxes, scores, nms_threshold)]


class PoseDecoder:
    """Select the native decoder once and fail over without changing its result schema."""

    def __init__(self, config=None):
        config = config or {}
        self.requested_backend = str(config.get("backend", "cpp"))
        self.strict = bool(config.get("strict", False))
        self.fallback = str(config.get("fallback", "numpy"))
        if self.requested_backend not in ("auto", "cpp", "numpy"):
            raise ValueError(f"unsupported postprocess backend: {self.requested_backend}")
        if self.fallback not in ("none", "numpy"):
            raise ValueError(f"unsupported postprocess fallback: {self.fallback}")
        self._cpp = None
        self.active_backend = "numpy"
        if self.requested_backend in ("auto", "cpp"):
            try:
                import rknn_postprocess
                self._cpp = rknn_postprocess
                self.active_backend = "cpp"
            except Exception as exc:
                if self.strict or self.fallback == "none":
                    raise RuntimeError("C++ pose postprocess is unavailable") from exc
                print(f"[postprocess] C++ unavailable, using NumPy: {exc}", flush=True)

    def decode(self, outputs, confidence=0.35, nms_threshold=0.45, input_size=640):
        if self._cpp is not None:
            try:
                return self._cpp.decode_pose(outputs, confidence=confidence,
                                              nms_threshold=nms_threshold,
                                              input_size=input_size)
            except Exception as exc:
                if self.strict or self.fallback == "none":
                    raise
                print(f"[postprocess] C++ failed, switching to NumPy: {exc}", flush=True)
                self._cpp = None
                self.active_backend = "numpy"
        return decode_pose_numpy(outputs, confidence, nms_threshold, input_size)


_DEFAULT_DECODER = None


def decode_pose(outputs, confidence=0.35, nms_threshold=0.45, size=640,
                *, input_size=None, backend=None):
    """Backward-compatible functional API; ``backend`` is primarily for tests/tools."""
    global _DEFAULT_DECODER
    if input_size is not None:
        size = input_size
    if backend is not None:
        return PoseDecoder({"backend": backend}).decode(outputs, confidence, nms_threshold, size)
    if _DEFAULT_DECODER is None:
        _DEFAULT_DECODER = PoseDecoder()
    return _DEFAULT_DECODER.decode(outputs, confidence, nms_threshold, size)


class RKNNPose:
    def __init__(self, model: str, core_mask: int | None = None):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model)
        if ret: raise RuntimeError(f"load_rknn({model})={ret}")
        ret = self.rknn.init_runtime(**({"core_mask": core_mask} if core_mask is not None else {}))
        if ret: raise RuntimeError(f"init_runtime({model})={ret}")

    def infer(self, rgb_640: np.ndarray):
        x = np.ascontiguousarray(rgb_640[None], dtype=np.uint8)
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[x])
        elapsed = (time.perf_counter() - start) * 1000
        if outputs is None: raise RuntimeError("RKNN inference returned None")
        return outputs, elapsed

    def close(self):
        self.rknn.release()
