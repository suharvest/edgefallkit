#!/usr/bin/env python3
"""Readable Python control plane for the Jetson fall detector.

Python owns RTSP reconnection, stream orchestration, MQTT and the per-track
fall state machine.  Above ``runtime.max_streams_per_worker`` streams the
process shards itself: one OS process per group, because every stream's
per-frame Python work (payload building, JSON, MQTT) contends for a single
GIL and threads alone stop scaling well before the GPU saturates.  The hot data plane stays in ``libjetson_fall_trt.so``:
OpenCV/GStreamer hands a BGR pointer to C++, where resize/letterbox, TensorRT
enqueueV3 and YOLO parsing happen.  Only compact detections/keypoints cross
the ctypes boundary; no image tensor or per-pixel Python loop is used.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
    import numpy as np
except ImportError:  # --validate and host config tests do not need video deps.
    cv2 = None
    np = None

try:
    import paho.mqtt.client as mqtt
except ImportError:  # The validate/config tests do not need an MQTT client.
    mqtt = None


RUNNING = True
JOINT_COUNT = 17
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


class CDetection(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("score", ctypes.c_float),
        ("keypoint_offset", ctypes.c_uint32),
        ("keypoint_count", ctypes.c_uint32),
    ]


class CKeypoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


class CFrameMeta(ctypes.Structure):
    _fields_ = [
        ("detection_count", ctypes.c_uint32),
        ("keypoint_count", ctypes.c_uint32),
        ("inference_ms", ctypes.c_float),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


class CTemporalResult(ctypes.Structure):
    _fields_ = [
        ("evaluated", ctypes.c_int32),
        ("positive", ctypes.c_int32),
        ("probability", ctypes.c_float),
    ]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    keypoints: list[Keypoint]

    def iou(self, other: "Detection | Track") -> float:
        left = max(self.x - self.w * 0.5, other.x - other.w * 0.5)
        top = max(self.y - self.h * 0.5, other.y - other.h * 0.5)
        right = min(self.x + self.w * 0.5, other.x + other.w * 0.5)
        bottom = min(self.y + self.h * 0.5, other.y + other.h * 0.5)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.w * self.h + other.w * other.h - intersection
        return intersection / union if union > 1e-8 else 0.0

    def center_distance(self, other: "Detection | Track") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class TrtBridge:
    """ctypes facade over the native TensorRT + YOLO parser library."""

    def __init__(self, config: dict[str, Any]):
        if np is None:
            raise RuntimeError("numpy and OpenCV are required for runtime inference")
        library_path = config.get("trt_library", "/app/libjetson_fall_trt.so")
        self.library = ctypes.CDLL(library_path)
        self._bind_api()
        configured_profile = str(config.get("temporal_profile", "auto"))
        if configured_profile == "auto":
            engine_name = Path(str(config["engine_path"])).name.lower()
            configured_profile = "yolo11m-pose" if "yolo11m" in engine_name else "yolo11s-pose"
        self.temporal_profile = configured_profile
        self.handle = self.library.jf_trt_create(
            os.fsencode(config["engine_path"]),
            int(config.get("input", {}).get("width", 640)),
            int(config.get("input", {}).get("height", 640)),
            float(config.get("score_threshold", 0.35)),
            float(config.get("keypoint_threshold", 0.25)),
            float(config.get("nms_threshold", 0.45)),
        )
        if not self.handle:
            raise RuntimeError(f"TensorRT engine failed to load: {config['engine_path']}")
        self.detection_capacity = max(1, int(config.get("max_detections", 128)))
        self.keypoint_capacity = self.detection_capacity * JOINT_COUNT
        self.det_buffer = (CDetection * self.detection_capacity)()
        self.kp_buffer = (CKeypoint * self.keypoint_capacity)()

    def _bind_api(self) -> None:
        void_p = ctypes.c_void_p
        byte_p = ctypes.POINTER(ctypes.c_uint8)
        det_p = ctypes.POINTER(CDetection)
        kp_p = ctypes.POINTER(CKeypoint)
        self.library.jf_trt_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.library.jf_trt_create.restype = void_p
        self.library.jf_trt_destroy.argtypes = [void_p]
        self.library.jf_trt_destroy.restype = None
        self.library.jf_trt_infer.argtypes = [void_p, byte_p, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_size_t, det_p, ctypes.c_size_t,
                                              kp_p, ctypes.c_size_t, ctypes.POINTER(CFrameMeta)]
        self.library.jf_trt_infer.restype = ctypes.c_int
        self.library.jf_trt_last_error.argtypes = [void_p]
        self.library.jf_trt_last_error.restype = ctypes.c_char_p
        self.library.jf_temporal_create.argtypes = []
        self.library.jf_temporal_create.restype = void_p
        self.library.jf_temporal_create_profile.argtypes = [ctypes.c_char_p]
        self.library.jf_temporal_create_profile.restype = void_p
        self.library.jf_temporal_destroy.argtypes = [void_p]
        self.library.jf_temporal_destroy.restype = None
        self.library.jf_temporal_update.argtypes = [void_p, ctypes.POINTER(CKeypoint), ctypes.c_size_t,
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                                    ctypes.c_int32, ctypes.c_double,
                                                    ctypes.POINTER(CTemporalResult)]
        self.library.jf_temporal_update.restype = ctypes.c_int

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("GStreamer appsink must provide an uint8 BGR frame")
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        meta = CFrameMeta()
        pointer = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        status = self.library.jf_trt_infer(
            self.handle, pointer, width, height, frame.strides[0],
            self.det_buffer, self.detection_capacity,
            self.kp_buffer, self.keypoint_capacity, ctypes.byref(meta),
        )
        if status != 0:
            detail = self.library.jf_trt_last_error(self.handle)
            raise RuntimeError(detail.decode("utf-8", "replace") if detail else f"status={status}")
        detections: list[Detection] = []
        for index in range(meta.detection_count):
            item = self.det_buffer[index]
            keypoints = [
                Keypoint(self.kp_buffer[item.keypoint_offset + offset].x,
                         self.kp_buffer[item.keypoint_offset + offset].y,
                         self.kp_buffer[item.keypoint_offset + offset].confidence)
                for offset in range(item.keypoint_count)
            ]
            detections.append(Detection(item.x, item.y, item.w, item.h, item.score, keypoints))
        return detections, float(meta.inference_ms)

    def create_temporal(self) -> ctypes.c_void_p:
        handle = self.library.jf_temporal_create_profile(self.temporal_profile.encode("ascii"))
        if not handle:
            raise RuntimeError(f"could not allocate temporal classifier profile={self.temporal_profile}")
        return handle

    def temporal_update(self, handle: ctypes.c_void_p, keypoints: Iterable[Keypoint],
                        width: int, height: int, hip_y: float, torso_angle: float,
                        aspect: float, score: float, valid: bool, timestamp: float) -> CTemporalResult:
        points = list(keypoints)
        c_points = (CKeypoint * len(points))(*[
            CKeypoint(point.x, point.y, point.confidence) for point in points
        ]) if points else None
        result = CTemporalResult()
        status = self.library.jf_temporal_update(
            handle, c_points, len(points), width, height, hip_y, torso_angle,
            aspect, score, int(valid), timestamp, ctypes.byref(result),
        )
        if status != 0:
            raise RuntimeError(f"temporal classifier status={status}")
        return result

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.library.jf_trt_destroy(self.handle)
            self.handle = None

    def close_temporal(self, handle: ctypes.c_void_p) -> None:
        if handle:
            self.library.jf_temporal_destroy(handle)


class MqttPublisher:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client = None
        if not config.get("enabled", True):
            return
        if mqtt is None:
            raise RuntimeError("MQTT is enabled but python3-paho-mqtt is not installed")
        self.client = mqtt.Client(client_id=str(config.get("client_id", "jetson-fall-detection")))
        username = config.get("username", "")
        if username:
            self.client.username_pw_set(username, config.get("password", ""))
        if config.get("tls", False):
            self.client.tls_set(ca_certs=config.get("ca_file") or None,
                                certfile=config.get("cert_file") or None,
                                keyfile=config.get("key_file") or None)
        self.client.connect(config.get("host", "127.0.0.1"), int(config.get("port", 1883)),
                            int(config.get("keepalive_sec", 30)))
        self.client.loop_start()

    def publish(self, topic: str, payload: str, retain: bool = False) -> None:
        if self.client is not None:
            qos = max(0, min(2, int(self.config.get("qos", 0))))
            info = self.client.publish(topic, payload, qos=qos, retain=retain)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"MQTT publish failed: {info.rc}")

    def close(self) -> None:
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None


@dataclass
class FallFeatures:
    valid: bool = False
    hip_y: float = 0.0
    torso_angle_deg: float = 0.0
    bbox_aspect_ratio: float = 0.0
    person_score: float = 0.0
    hip_drop_speed: float = 0.0
    hip_drop_distance: float = 0.0
    lying_posture: bool = False
    upright_posture: bool = False
    evidence_features: int = 0
    temporal_probability: float = 0.0
    temporal_positive: bool = False


class Track:
    """One person's geometry + learned temporal state, isolated by track_id."""

    def __init__(self, track_id: int, detection: Detection, bridge: TrtBridge,
                 config: dict[str, Any]):
        self.track_id = track_id
        self.box = detection
        self.score = detection.score
        self.bridge = bridge
        self.temporal = bridge.create_temporal()
        self.config = config
        self.fall_cfg = config.get("fall", {})
        self.state = "normal"
        self.event_id = 0
        self.fall_event = False
        self.age = 0
        self.missed = 0
        self.initialized = False
        self.previous_hip_y: Optional[float] = None
        self.previous_timestamp: Optional[float] = None
        self.baseline_hip_y: Optional[float] = None
        self.last_fast_drop = -1.0
        self.max_drop_distance = 0.0
        self.suspected_since = -1.0
        self.last_strong_evidence = -1.0
        self.motion_triggered = False
        self.recovery_since = -1.0
        self.cooldown_until = -1.0
        self.last_timestamp = 0.0
        self.features = FallFeatures()

    @property
    def temporal_confirmation_required(self) -> bool:
        """Whether learned temporal evidence is required to enter ``fallen``.

        The deployed default deliberately keeps the temporal classifier as the
        only fall confirmation signal.  A false value is an explicit
        compatibility switch for the old geometry-only path; it is useful for
        bring-up but should not be used for production evaluation.
        """
        return bool(self.fall_cfg.get("temporal_confirmation_required", True))

    def close(self) -> None:
        self.bridge.close_temporal(self.temporal)
        self.temporal = None

    @property
    def x(self) -> float:
        return self.box.x

    @property
    def y(self) -> float:
        return self.box.y

    @property
    def w(self) -> float:
        return self.box.w

    @property
    def h(self) -> float:
        return self.box.h

    def _number(self, name: str, default: float) -> float:
        return float(self.fall_cfg.get(name, default))

    def _keypoint(self, points: list[Keypoint], index: int) -> Optional[Keypoint]:
        threshold = float(self.config.get("keypoint_threshold", 0.25))
        return points[index] if index < len(points) and points[index].confidence >= threshold else None

    @staticmethod
    def _midpoint(points: list[Keypoint], a: int, b: int, threshold: float) -> Optional[tuple[float, float]]:
        first = points[a] if a < len(points) and points[a].confidence >= threshold else None
        second = points[b] if b < len(points) and points[b].confidence >= threshold else None
        if first and second:
            return (0.5 * (first.x + second.x), 0.5 * (first.y + second.y))
        if first:
            return first.x, first.y
        if second:
            return second.x, second.y
        return None

    def _features(self, detection: Detection, timestamp: float) -> FallFeatures:
        threshold = float(self.config.get("keypoint_threshold", 0.25))
        hip = self._midpoint(detection.keypoints, LEFT_HIP, RIGHT_HIP, threshold)
        shoulder = self._midpoint(detection.keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, threshold)
        if hip is None:
            visible = [p for p in detection.keypoints if p.confidence >= threshold]
            hip = (sum(p.x for p in visible) / len(visible), sum(p.y for p in visible) / len(visible)) if visible else None
        valid = hip is not None and bool(detection.keypoints)
        hip_y = hip[1] if hip else 0.0
        torso = 0.0
        if hip and shoulder:
            torso = math.degrees(math.atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1])))
        aspect = detection.w / detection.h if detection.h > 1e-6 else 0.0
        speed = 0.0
        if valid and self.previous_hip_y is not None and self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            if 1e-4 < dt < 10.0:
                speed = (hip_y - self.previous_hip_y) / dt
        torso_threshold = self._number("torso_angle_threshold_deg", 55.0)
        aspect_threshold = self._number("bbox_aspect_ratio_threshold", 1.25)
        evidence = int(speed >= self._number("hip_drop_speed_threshold", 0.25))
        evidence += int(torso >= torso_threshold)
        evidence += int(aspect >= aspect_threshold)
        lying = torso >= torso_threshold and aspect >= aspect_threshold
        upright = torso <= self._number("recovery_torso_angle_deg", 35.0) and aspect <= self._number("recovery_aspect_ratio", 1.10)
        return FallFeatures(valid, hip_y, torso, aspect, detection.score, speed,
                            self.max_drop_distance, lying, upright, evidence)

    def _trigger(self, timestamp: float) -> None:
        self.state = "fallen"
        self.fall_event = True
        self.event_id += 1
        self.recovery_since = -1.0
        self.cooldown_until = timestamp + self._number("cooldown_sec", 3.0)

    def update(self, detection: Detection, timestamp: float, frame_width: int, frame_height: int) -> None:
        self.box = detection
        self.score = detection.score
        self.age += 1 if self.age else 1
        self.missed = 0
        current = self._features(detection, timestamp)
        if self.state == "normal" and not current.lying_posture:
            self.baseline_hip_y = current.hip_y
        if self.state == "suspected" and self.baseline_hip_y is not None:
            self.max_drop_distance = max(self.max_drop_distance, current.hip_y - self.baseline_hip_y)
            current.hip_drop_distance = self.max_drop_distance
        if current.hip_drop_speed >= self._number("hip_drop_speed_threshold", 0.25):
            self.last_fast_drop = timestamp
        temporal = self.bridge.temporal_update(
            self.temporal, detection.keypoints, frame_width, frame_height,
            current.hip_y, current.torso_angle_deg, current.bbox_aspect_ratio,
            current.person_score, current.valid, timestamp,
        )
        current.temporal_probability = float(temporal.probability)
        current.temporal_positive = bool(temporal.positive)
        self._advance(current, timestamp)
        self.features = current
        self.last_timestamp = timestamp
        self.previous_hip_y = current.hip_y
        self.previous_timestamp = timestamp
        self.initialized = True

    def update_missing(self, timestamp: float, frame_width: int, frame_height: int) -> None:
        self.fall_event = False
        self.missed += 1
        temporal = self.bridge.temporal_update(
            self.temporal, [], frame_width, frame_height, 0.0, 0.0, 0.0,
            0.0, False, timestamp,
        )
        current = FallFeatures(valid=False, temporal_probability=float(temporal.probability),
                               temporal_positive=bool(temporal.positive),
                               hip_drop_distance=self.max_drop_distance)
        if self.state == "suspected" and self.suspected_since >= 0.0:
            # Keep the temporal window warm, but do not create an event from a
            # missed/stale track. A reacquired visible pose can confirm using
            # the retained history on its next update.
            age = timestamp - self.suspected_since
            recent = self.last_strong_evidence >= 0.0 and timestamp - self.last_strong_evidence <= self._number("occlusion_grace_sec", 0.75)
            if (not self.temporal_confirmation_required and self._geometry_confirmation_ready(timestamp, recent=recent)):
                # Legacy compatibility only.  The normal configuration never
                # reaches this branch: geometry may arm suspicion, but cannot
                # confirm a fall.
                self._trigger(timestamp)
            elif age > self._number("suspected_timeout_sec", 1.5):
                self._reset_suspicion()
        current.lying_posture = False
        self.features = current
        self.last_timestamp = timestamp

    def _reset_suspicion(self) -> None:
        self.state = "normal"
        self.suspected_since = -1.0
        self.last_strong_evidence = -1.0
        self.motion_triggered = False
        self.last_fast_drop = -1.0
        self.max_drop_distance = 0.0

    def _geometry_confirmation_ready(self, timestamp: float, *, recent: Optional[bool] = None) -> bool:
        """Return the old geometry confirmation predicate.

        This is intentionally isolated behind ``temporal_confirmation_required``
        so the production path cannot accidentally regress to a geometry-only
        fall alarm when thresholds are changed.
        """
        if self.temporal_confirmation_required or self.suspected_since < 0.0:
            return False
        if recent is None:
            recent = (self.last_strong_evidence >= 0.0 and
                      timestamp - self.last_strong_evidence <= self._number("occlusion_grace_sec", 0.75))
        return bool(
            self.motion_triggered and recent and
            self.max_drop_distance >= self._number("hip_drop_distance_threshold", 0.02) and
            timestamp - self.suspected_since >= self._number("confirmation_sec", 0.80)
        )

    def _temporal_confirmation_ready(self, feature: FallFeatures) -> bool:
        """Require learned positivity after at least one prior observation."""
        # The native temporal classifier normally withholds ``positive`` until
        # its learned window is populated.  Keeping this explicit guard makes
        # the first-frame/no-history invariant hold for fake or future bridges
        # too, while still allowing a positive result during short occlusion.
        return bool(feature.temporal_positive and self.age > 1)

    def _advance(self, current: FallFeatures, timestamp: float) -> None:
        self.fall_event = False
        if not self.initialized:
            # A first-frame lying pose is not a fall event.  There is no prior
            # motion context yet.  Even an optimistic/future temporal bridge
            # must observe at least one prior frame before it may confirm.
            if not self.temporal_confirmation_required and self._geometry_confirmation_ready(timestamp):
                self._trigger(timestamp)
            elif self._temporal_confirmation_ready(current) and timestamp >= self.cooldown_until:
                self._trigger(timestamp)
            return
        cooldown = timestamp < self.cooldown_until
        if self.state == "normal":
            if not cooldown and self._temporal_confirmation_ready(current):
                self._trigger(timestamp)
            elif not cooldown and self.last_fast_drop >= 0.0 and timestamp - self.last_fast_drop <= self._number("motion_window_sec", 0.75) and (current.torso_angle_deg >= self._number("torso_angle_threshold_deg", 55.0) or current.bbox_aspect_ratio >= self._number("bbox_aspect_ratio_threshold", 1.25)):
                self.state = "suspected"
                self.suspected_since = self.last_fast_drop
                self.last_strong_evidence = timestamp if current.lying_posture else -1.0
                self.motion_triggered = True
                self.max_drop_distance = max(0.0, current.hip_y - (self.baseline_hip_y or current.hip_y))
        elif self.state == "suspected":
            if not cooldown and self._temporal_confirmation_ready(current):
                self._trigger(timestamp)
            elif current.lying_posture and current.evidence_features >= int(self.fall_cfg.get("min_suspected_features", 2)):
                self.last_strong_evidence = timestamp
                if self._geometry_confirmation_ready(timestamp):
                    self._trigger(timestamp)
            elif current.upright_posture or timestamp - self.suspected_since > self._number("suspected_timeout_sec", 1.5):
                self._reset_suspicion()
        elif self.state == "fallen":
            if current.upright_posture:
                self.recovery_since = timestamp
                self.state = "recovering" if self._number("recovery_window_sec", 2.0) > 0 else "normal"
                if self.state == "normal": self._reset_suspicion()
        elif self.state == "recovering":
            if not current.upright_posture:
                self.state = "fallen"
                self.recovery_since = -1.0
            elif timestamp - self.recovery_since >= self._number("recovery_window_sec", 2.0):
                self._reset_suspicion()

    def as_json(self) -> dict[str, Any]:
        feature = self.features
        pose17 = [[round(point.x, 5), round(point.y, 5), round(point.confidence, 5)] for point in self.box.keypoints]
        return {
            "track_id": self.track_id,
            "person_detected": feature.valid,
            "person_score": round(self.score, 4),
            "fall_detected": self.state in ("fallen", "recovering"),
            "fall_event": self.fall_event,
            "event_id": self.event_id,
            "state": self.state,
            "tracking": feature.valid,
            "missed_frames": self.missed,
            "features": {
                "valid": feature.valid,
                "hip_y": round(feature.hip_y, 4),
                "person_score": round(feature.person_score, 4),
                "hip_drop_speed": round(feature.hip_drop_speed, 4),
                "hip_drop_distance": round(feature.hip_drop_distance, 4),
                "torso_angle_deg": round(feature.torso_angle_deg, 2),
                "bbox_aspect_ratio": round(feature.bbox_aspect_ratio, 3),
                "evidence_features": feature.evidence_features,
                "evidence_score": round(feature.evidence_features / 3.0, 3),
                "lying_posture": feature.lying_posture,
                "upright_posture": feature.upright_posture,
                "in_cooldown": self.last_timestamp < self.cooldown_until,
                "temporal_probability": round(feature.temporal_probability, 4),
                "temporal_positive": feature.temporal_positive,
                "suspected_for_sec": round(max(0.0, self.last_timestamp - self.suspected_since), 3) if self.suspected_since >= 0 else 0.0,
                "recovery_for_sec": round(max(0.0, self.last_timestamp - self.recovery_since), 3) if self.recovery_since >= 0 else 0.0,
            },
            "bbox": [round(self.box.x, 5), round(self.box.y, 5), round(self.box.w, 5), round(self.box.h, 5)],
            # `keypoints` is retained for reCamera compatibility while
            # `pose17` is the normalized cross-platform COCO representation.
            "keypoints": pose17,
            "pose17": pose17,
        }


class MultiPersonTracker:
    def __init__(self, bridge: TrtBridge, config: dict[str, Any]):
        self.bridge = bridge
        self.config = config
        tracker_config = config.get("tracker", {})
        self.iou_threshold = float(tracker_config.get("iou_threshold", 0.20))
        self.distance_threshold = float(tracker_config.get("center_distance_threshold", 0.25))
        self.max_missed = int(tracker_config.get("max_missed_frames", 8))
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[Detection], timestamp: float,
               frame_width: int, frame_height: int) -> list[Track]:
        matches: list[tuple[float, int, int]] = []
        for d_index, detection in enumerate(detections):
            for t_index, track in enumerate(self.tracks):
                iou = detection.iou(track)
                distance = detection.center_distance(track)
                if iou >= self.iou_threshold or distance <= self.distance_threshold:
                    matches.append((iou + 1.0 - min(1.0, distance), d_index, t_index))
        matches.sort(reverse=True)
        detection_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, d_index, t_index in matches:
            if d_index not in detection_to_track and t_index not in used_tracks:
                detection_to_track[d_index] = t_index
                used_tracks.add(t_index)
        for d_index, detection in enumerate(detections):
            if d_index in detection_to_track:
                self.tracks[detection_to_track[d_index]].update(detection, timestamp, frame_width, frame_height)
            else:
                track = Track(self.next_id, detection, self.bridge, self.config)
                self.next_id += 1
                # Track.__init__ has a synthetic timestamp only to initialize
                # geometry; immediately process the real frame for temporal time.
                track.update(detection, timestamp, frame_width, frame_height)
                self.tracks.append(track)
        # Explicitly mark original unmatched tracks. New tracks are appended
        # after this snapshot and must never receive a first-frame miss.
        original_tracks = list(self.tracks)
        new_start = len(original_tracks) - (len(detections) - len(detection_to_track))
        for index, track in enumerate(original_tracks[:max(0, new_start)]):
            if index not in used_tracks:
                track.update_missing(timestamp, frame_width, frame_height)
        survivors: list[Track] = []
        for track in self.tracks:
            if track.missed <= self.max_missed:
                survivors.append(track)
            else:
                track.close()
        self.tracks = survivors
        # Keep short-lived unmatched tracks in the result so a post-impact
        # pose loss can still publish the person's fall event.  Their
        # `tracking`/`person_detected` fields are false until reacquired.
        return list(self.tracks)


def gstreamer_pipeline(stream: dict[str, Any], config: dict[str, Any]) -> str:
    url = stream["rtsp_url"]
    codec = str(stream.get("codec", "h264")).lower()
    if codec in ("h265", "hevc"):
        depay, parser, decoder = "rtph265depay", "h265parse", "nvv4l2decoder"
    else:
        depay, parser, decoder = "rtph264depay", "h264parse", "nvv4l2decoder"
    latency = int(stream.get("latency_ms", config.get("rtsp_latency_ms", 100)))
    # nvv4l2decoder + nvvidconv are intentionally explicit: Jetson hardware
    # decode/convert stays on the data plane before appsink hands BGR to C++.
    return (f"rtspsrc location={url} protocols=tcp latency={latency} ! {depay} ! {parser} ! "
            f"{decoder} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


class StreamWorker(threading.Thread):
    def __init__(self, stream: dict[str, Any], config: dict[str, Any], publisher: MqttPublisher):
        super().__init__(name=f"stream-{stream['id']}", daemon=True)
        self.stream = stream
        self.config = config
        self.publisher = publisher
        self.bridge: Optional[TrtBridge] = None
        self.tracker: Optional[MultiPersonTracker] = None
        # Stream-global event counter. Per-track counters remain inside each
        # persons[] entry and must not be used as the top-level event id.
        self.global_event_id = 0

    def run(self) -> None:
        if cv2 is None:
            print(f"[{self.stream['id']}] python3-opencv/numpy are not installed", file=sys.stderr)
            return
        reconnect_ms = max(100, int(self.stream.get("reconnect_delay_ms", 1000)))
        try:
            self.bridge = TrtBridge(self.config)
            self.tracker = MultiPersonTracker(self.bridge, self.config)
            while RUNNING:
                pipeline = gstreamer_pipeline(self.stream, self.config)
                capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not capture.isOpened():
                    print(f"[{self.stream['id']}] GStreamer RTSP open failed; retrying", file=sys.stderr)
                    time.sleep(reconnect_ms / 1000.0)
                    continue
                print(f"[{self.stream['id']}] RTSP connected", file=sys.stderr)
                try:
                    self._read_loop(capture)
                finally:
                    capture.release()
                    if self.tracker:
                        for track in self.tracker.tracks:
                            track.close()
                        self.tracker.tracks.clear()
                if RUNNING:
                    time.sleep(reconnect_ms / 1000.0)
        except Exception as error:
            print(f"[{self.stream['id']}] worker stopped: {error}", file=sys.stderr)
        finally:
            if self.bridge:
                self.bridge.close()

    def _read_loop(self, capture: cv2.VideoCapture) -> None:
        frame_id = 0
        max_fps = int(self.config.get("max_fps", 0))
        while RUNNING:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                print(f"[{self.stream['id']}] RTSP read failed; reconnecting", file=sys.stderr)
                return
            detections, infer_ms = self.bridge.infer(frame)
            timestamp = time.monotonic()
            persons = self.tracker.update(detections, timestamp, frame.shape[1], frame.shape[0])
            frame_id += 1
            publish_track = any(track.features.valid or track.fall_event or track.state in ("fallen", "recovering") for track in persons)
            if publish_track or self.config.get("publish_empty_frames", False):
                payload = self._payload(persons, frame_id, infer_ms)
                topic = str(self.config.get("mqtt", {}).get("topic", "recamera/fall-detection/results"))
                topic = topic.replace("{stream_id}", str(self.stream["id"]))
                self.publisher.publish(topic, json.dumps(payload, separators=(",", ":")),
                                        bool(self.config.get("mqtt", {}).get("retain", False)))
            if max_fps > 0:
                delay = 1.0 / max_fps - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)

    def _payload(self, persons: list[Track], frame_id: int, infer_ms: float) -> dict[str, Any]:
        active_persons = [track for track in persons if track.features.valid]
        fallen = sum(track.state in ("fallen", "recovering") for track in persons)
        primary = max(persons, key=lambda track: track.score, default=None)
        if any(track.fall_event for track in persons):
            self.global_event_id += 1
        state = "normal"
        for track in persons:
            if track.state == "fallen":
                state = "fallen"
                break
            if track.state == "recovering" and state == "normal":
                state = "recovering"
            elif track.state == "suspected" and state == "normal":
                state = "suspected"
        primary_json = primary.as_json() if primary else None
        empty_features = {
            "valid": False,
            "hip_drop_speed": 0.0,
            "hip_drop_distance": 0.0,
            "torso_angle_deg": 0.0,
            "bbox_aspect_ratio": 0.0,
        }
        return {
            "timestamp": int(time.time() * 1000),
            "frame_id": frame_id,
            "inference_time_ms": round(infer_ms, 2),
            "stream_id": self.stream["id"],
            "fall_detected": fallen > 0,
            "fall_event": any(track.fall_event for track in persons),
            "event_id": self.global_event_id,
            "global_event_id": self.global_event_id,
            "event_id_scope": "stream_global_event_id",
            "state": state,
            "person_detected": bool(active_persons),
            "person_count": len(active_persons),
            "fallen_count": fallen,
            "tracking": bool(persons),
            "persons": [track.as_json() for track in persons],
            "features": primary_json["features"] if primary_json else empty_features,
            "keypoints": primary_json["keypoints"] if primary_json else [],
            "pose17": primary_json["pose17"] if primary_json else [],
        }


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    if not config.get("engine_path"):
        raise ValueError("engine_path must not be empty")
    temporal_profile = config.setdefault("temporal_profile", "auto")
    if temporal_profile not in ("auto", "yolo11s-pose", "yolo11m-pose"):
        raise ValueError("temporal_profile must be auto, yolo11s-pose, or yolo11m-pose")
    input_config = config.get("input", {})
    width = int(input_config.get("width", 640))
    height = int(input_config.get("height", 640))
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise ValueError("input.width/height must be in 1..8192")
    for name in ("score_threshold", "keypoint_threshold", "nms_threshold"):
        if name in config and not (0.0 <= float(config[name]) <= 1.0):
            raise ValueError(f"{name} must be in [0,1]")
    fall_config = config.setdefault("fall", {})
    if not isinstance(fall_config, dict):
        raise ValueError("fall must be an object")
    temporal_required = fall_config.setdefault("temporal_confirmation_required", True)
    if not isinstance(temporal_required, bool):
        raise ValueError("fall.temporal_confirmation_required must be boolean")
    runtime_config = config.setdefault("runtime", {})
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime must be an object")
    workers = runtime_config.setdefault("workers", "auto")
    if workers != "auto" and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError('runtime.workers must be "auto" or a positive integer')
    per_worker = runtime_config.setdefault("max_streams_per_worker", 0)
    if isinstance(per_worker, bool) or not isinstance(per_worker, int) or per_worker < 0:
        raise ValueError("runtime.max_streams_per_worker must be a non-negative integer")
    streams = config.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must contain at least one RTSP source")
    for index, stream in enumerate(streams):
        if stream.get("enabled", True) and not stream.get("rtsp_url"):
            raise ValueError(f"streams[{index}].rtsp_url must not be empty")
    mqtt_config = config.get("mqtt", {})
    if mqtt_config.get("enabled", True):
        if not mqtt_config.get("topic"):
            raise ValueError("mqtt.topic must not be empty")
        if not mqtt_config.get("host", "127.0.0.1") or not (1 <= int(mqtt_config.get("port", 1883)) <= 65535):
            raise ValueError("mqtt.host/port is invalid")
    return config


def worker_count(config: dict[str, Any], stream_count: int) -> int:
    """How many OS processes should own the enabled streams.

    ``runtime.workers`` accepts "auto" or a positive integer.  "auto" divides
    the streams by ``runtime.max_streams_per_worker``, which is a per-device
    calibration: it is the number of streams one Python process sustains at
    the target frame rate before the GIL, not the accelerator, becomes the
    limit.  A count of 1 keeps the historical single-process behaviour.
    """
    runtime = config.get("runtime", {})
    requested = runtime.get("workers", "auto")
    if requested == "auto":
        per_worker = int(runtime.get("max_streams_per_worker", 0))
        if per_worker <= 0:
            return 1
        count = math.ceil(stream_count / per_worker)
    else:
        count = int(requested)
    return max(1, min(count, stream_count))


def shard_streams(streams: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    """Split streams into `shard_count` groups whose sizes differ by at most 1."""
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, stream in enumerate(streams):
        shards[index % shard_count].append(stream)
    return shards


def run_streams(config: dict[str, Any], streams: list[dict[str, Any]],
                shard_index: int, shard_count: int) -> int:
    """Run one shard's streams in this process.  Also the single-shard path."""
    global RUNNING
    RUNNING = True

    def stop_handler(_signal: int, _frame: Any) -> None:
        global RUNNING
        RUNNING = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    mqtt_config = dict(config.get("mqtt", {}))
    if shard_count > 1:
        # Every process needs its own broker session. Sharing one client id
        # makes the broker evict the previous session, after which each
        # publish fails with rc=4 and the shard goes silent.
        base = str(mqtt_config.get("client_id", "jetson-fall-detection"))
        mqtt_config["client_id"] = f"{base}-{shard_index}"
    try:
        publisher = MqttPublisher(mqtt_config)
    except (OSError, RuntimeError) as error:
        # Keep camera inference alive when an optional broker is temporarily
        # offline; the error is explicit and the next process restart retries.
        print(f"MQTT disabled after setup failure: {error}", file=sys.stderr)
        publisher = MqttPublisher({"enabled": False})
    workers = [StreamWorker(stream, config, publisher) for stream in streams]
    if not workers:
        print(f"[shard {shard_index}] no enabled RTSP streams", file=sys.stderr)
        publisher.close()
        return 2
    for worker in workers:
        worker.start()
    try:
        while RUNNING:
            time.sleep(0.5)
    finally:
        RUNNING = False
        for worker in workers:
            worker.join(timeout=10.0)
        publisher.close()
    return 0


def supervise(config: dict[str, Any], shards: list[list[dict[str, Any]]]) -> int:
    """Start one process per shard and restart any that exits early."""
    global RUNNING
    # spawn, not fork: a forked child inherits the parent's CUDA and GStreamer
    # state, which is not valid to use after fork.
    context = multiprocessing.get_context("spawn")
    shard_count = len(shards)
    processes: dict[int, Any] = {}
    next_restart: dict[int, float] = {}

    def start(index: int) -> None:
        process = context.Process(
            target=run_streams, name=f"shard-{index}",
            args=(config, shards[index], index, shard_count), daemon=False)
        process.start()
        processes[index] = process
        print(f"[shard {index}] started pid={process.pid} "
              f"streams={[stream['id'] for stream in shards[index]]}", file=sys.stderr)

    def stop_handler(_signal: int, _frame: Any) -> None:
        global RUNNING
        RUNNING = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    for index in range(shard_count):
        start(index)
    try:
        while RUNNING:
            time.sleep(0.5)
            now = time.monotonic()
            for index, process in list(processes.items()):
                if process.is_alive():
                    next_restart.pop(index, None)
                    continue
                # Back off so a shard that cannot start (bad engine, missing
                # camera) does not spin the supervisor.
                due = next_restart.get(index)
                if due is None:
                    print(f"[shard {index}] exited rc={process.exitcode}; restarting in 5s",
                          file=sys.stderr)
                    next_restart[index] = now + 5.0
                elif now >= due:
                    start(index)
    finally:
        RUNNING = False
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=15.0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson TensorRT multi-stream fall detection")
    parser.add_argument("--config", default="/app/config/config.json")
    parser.add_argument("--validate", action="store_true", help="validate config and exit")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    enabled = [stream for stream in config["streams"] if stream.get("enabled", True)]
    if args.validate:
        count = worker_count(config, len(enabled)) if enabled else 0
        print(f"configuration valid: {len(config.get('streams', []))} stream(s), "
              f"{len(enabled)} enabled, {count} worker process(es)")
        return 0
    if not enabled:
        print("No enabled RTSP streams", file=sys.stderr)
        return 2

    count = worker_count(config, len(enabled))
    if count == 1:
        return run_streams(config, enabled, 0, 1)
    return supervise(config, shard_streams(enabled, count))


if __name__ == "__main__":
    raise SystemExit(main())
