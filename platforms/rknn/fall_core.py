"""Dependency-light multi-person fall logic shared by RK3576 and RK3588."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    box: list[float]
    score: float
    keypoints: list[list[float]]


@dataclass
class Observation:
    valid: bool = False
    timestamp: float = 0.0
    hip_y: float = 0.0
    torso_angle: float = 0.0
    aspect: float = 0.0
    person_score: float = 0.0
    temporal_available: bool = False
    temporal_positive: bool = False
    temporal_probability: float = 0.0


def make_observation(det: Optional[Detection], timestamp: float, width: int, height: int,
                     keypoint_threshold: float = 0.25) -> Observation:
    if det is None or width <= 0 or height <= 0 or len(det.keypoints) < 17:
        return Observation(timestamp=timestamp)
    kp = det.keypoints
    needed = (5, 6, 11, 12)
    if any(len(kp[i]) < 3 or kp[i][2] < keypoint_threshold for i in needed):
        return Observation(timestamp=timestamp, person_score=det.score)
    shoulder_x = (kp[5][0] + kp[6][0]) * 0.5
    shoulder_y = (kp[5][1] + kp[6][1]) * 0.5
    hip_x = (kp[11][0] + kp[12][0]) * 0.5
    hip_y = (kp[11][1] + kp[12][1]) * 0.5
    dx, dy = hip_x - shoulder_x, hip_y - shoulder_y
    angle = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))
    x1, y1, x2, y2 = det.box
    aspect = max(0.0, x2 - x1) / max(1e-6, y2 - y1)
    return Observation(True, timestamp, hip_y / height, angle, aspect, det.score)


@dataclass
class Track:
    track_id: int
    box: list[float]
    last_seen: float
    detection: Optional[Detection] = None
    missed: int = 0


class IoUTracker:
    def __init__(self, threshold: float = 0.2, max_lost_sec: float = 0.75):
        self.threshold = threshold
        self.max_lost_sec = max_lost_sec
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.expired_ids: list[int] = []

    @staticmethod
    def iou(a: list[float], b: list[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(aa + ab - inter, 1e-9)

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        self.expired_ids = [tid for tid, tr in self.tracks.items()
                            if now - tr.last_seen > self.max_lost_sec]
        for tid in self.expired_ids:
            del self.tracks[tid]
        for tr in self.tracks.values():
            tr.detection = None; tr.missed += 1
        pairs = sorted((self.iou(tr.box, det.box), tid, di)
                       for tid, tr in self.tracks.items()
                       for di, det in enumerate(detections))
        used_t, used_d = set(), set()
        for score, tid, di in reversed(pairs):
            if score < self.threshold or tid in used_t or di in used_d:
                continue
            tr = self.tracks[tid]
            tr.box, tr.last_seen, tr.detection, tr.missed = detections[di].box, now, detections[di], 0
            used_t.add(tid); used_d.add(di)
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, det.box, now, det, 0)
        return [self.tracks[k] for k in sorted(self.tracks)]


class TemporalMLP:
    """Frozen 48-frame NumPy MLP; model is exported from the Jetson C++ header."""
    def __init__(self, path: str):
        data = np.load(path)
        for key in ("frame_mask", "mean", "scale", "w1", "b1", "w2"):
            setattr(self, key, data[key].astype(np.float32))
        self.b2 = float(data["b2"])
        self.threshold = float(data["threshold"])
        self.consecutive = int(data["consecutive"])
        self.window = int(data.get("window", 48))
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.last_eval = -1.0
        self.positive_run = 0
        self.probability = 0.0
        self.positive = False

    @staticmethod
    def pose_frame(det: Optional[Detection], obs: Observation, width: int, height: int) -> np.ndarray:
        out = np.zeros(56, np.float32)
        if det is None or not obs.valid:
            return out
        pts = np.asarray(det.keypoints[:17], np.float32)
        conf = np.clip(pts[:, 2], 0, 1)
        xy = pts[:, :2] / np.array([width, height], np.float32)
        weights = conf[11] + conf[12]
        if weights < 0.1:
            visible = conf >= 0.1
            if not visible.any():
                return out
            hip = np.average(xy[visible], axis=0, weights=conf[visible])
        else:
            hip = (xy[11] * conf[11] + xy[12] * conf[12]) / weights
        sw = conf[5] + conf[6]
        shoulder = (xy[5] * conf[5] + xy[6] * conf[6]) / max(sw, 1e-6)
        scale = float(np.linalg.norm(shoulder - hip)) if sw >= 0.1 else 0.0
        if scale < 0.04:
            visible = conf >= 0.1
            span = np.ptp(xy[visible], axis=0) if visible.any() else np.zeros(2)
            scale = max(float(span.max()) * 0.35, 0.04)
        visible = conf >= 0.1
        centered = np.clip((xy - hip) / scale, -4, 4)
        for i in range(17):
            if visible[i]:
                out[i * 2:i * 2 + 2] = centered[i]
        out[34:51] = conf
        out[-5:] = [obs.hip_y, obs.torso_angle / 90, min(obs.aspect, 4) / 4,
                    obs.person_score, 1]
        return out

    def update(self, frame: np.ndarray, now: float) -> tuple[bool, float]:
        self.frames.append((now, frame * self.frame_mask))
        while len(self.frames) > 1 and self.frames[1][0] < now - 3.7:
            self.frames.popleft()
        if self.last_eval >= 0 and now - self.last_eval < 0.2 - 1e-6:
            return self.positive, self.probability
        self.last_eval = now
        ts = np.array([x[0] for x in self.frames])
        values = np.stack([x[1] for x in self.frames])
        wanted = now - np.arange(self.window - 1, -1, -1, dtype=np.float64) / 15.0
        seq = values[np.searchsorted(ts, wanted, side="right").clip(1) - 1]
        bins = seq.reshape(6, self.window // 6, 56).mean(axis=1).reshape(-1)
        feat = np.concatenate((bins, seq.std(axis=0), seq[-1] - seq[0], np.ptp(seq, axis=0)))
        norm = (feat - self.mean) / np.maximum(self.scale, 1e-12)
        hidden = np.maximum(0, norm @ self.w1 + self.b1)
        logit = float(hidden @ self.w2 + self.b2)
        self.probability = 1 / (1 + math.exp(-max(-80.0, min(80.0, logit))))
        self.positive_run = self.positive_run + 1 if self.probability >= self.threshold else 0
        self.positive = self.positive_run >= self.consecutive
        return self.positive, self.probability


@dataclass
class FallConfig:
    temporal_confirmation_required: bool = True
    hip_drop_speed: float = 0.25
    hip_drop_distance: float = 0.02
    motion_window: float = 0.75
    torso_angle: float = 55.0
    aspect: float = 1.25
    confirmation: float = 0.8
    suspected_timeout: float = 1.5
    recovery_angle: float = 35.0
    recovery_aspect: float = 1.1
    recovery_window: float = 2.0
    cooldown: float = 3.0


class FallDetector:
    def __init__(self, cfg: FallConfig):
        self.cfg = cfg; self.state = "normal"; self.event_id = 0
        self.prev_hip = None; self.prev_time = None; self.baseline = None
        self.last_drop = -1.0; self.suspected_since = -1.0
        self.max_drop = 0.0; self.recovery_since = -1.0; self.cooldown_until = -1.0

    def update(self, o: Observation) -> dict:
        event = False; speed = 0.0
        if not o.valid:
            # Hard invariant: cached temporal output or disappearance can retain state,
            # but an invalid/missing observation can never originate a fall event.
            if self.state == "suspected" and o.timestamp - self.suspected_since > self.cfg.suspected_timeout:
                self.state = "normal"; self.suspected_since = -1.0; self.max_drop = 0.0
            return self.result(o, speed, event)
        if self.prev_time is not None and 1e-4 < o.timestamp - self.prev_time < 10:
            speed = (o.hip_y - self.prev_hip) / (o.timestamp - self.prev_time)
        horizontal = o.torso_angle >= self.cfg.torso_angle or o.aspect >= self.cfg.aspect
        lying = o.torso_angle >= self.cfg.torso_angle and o.aspect >= self.cfg.aspect
        upright = o.torso_angle <= self.cfg.recovery_angle and o.aspect <= self.cfg.recovery_aspect
        if self.state == "normal" and not horizontal:
            self.baseline = o.hip_y
        if speed >= self.cfg.hip_drop_speed:
            self.last_drop = o.timestamp
        if self.state == "normal":
            if o.timestamp >= self.cooldown_until and o.temporal_available and o.temporal_positive:
                self.state = "fallen"; event = True
            elif (o.timestamp >= self.cooldown_until and horizontal and self.last_drop >= 0 and
                  o.timestamp - self.last_drop <= self.cfg.motion_window):
                self.state = "suspected"; self.suspected_since = self.last_drop
                self.max_drop = max(0.0, o.hip_y - (self.baseline if self.baseline is not None else o.hip_y))
        elif self.state == "suspected":
            if self.baseline is not None:
                self.max_drop = max(self.max_drop, o.hip_y - self.baseline)
            if o.temporal_available and o.temporal_positive:
                self.state = "fallen"; event = True
            elif (not self.cfg.temporal_confirmation_required and lying and
                  self.max_drop >= self.cfg.hip_drop_distance and
                  o.timestamp - self.suspected_since >= self.cfg.confirmation):
                self.state = "fallen"; event = True
            elif upright or o.timestamp - self.suspected_since > self.cfg.suspected_timeout:
                self.state = "normal"; self.suspected_since = -1.0; self.max_drop = 0.0
        elif self.state == "fallen" and upright:
            self.state = "recovering"; self.recovery_since = o.timestamp
        elif self.state == "recovering":
            if not upright:
                self.state = "fallen"; self.recovery_since = -1.0
            elif o.timestamp - self.recovery_since >= self.cfg.recovery_window:
                self.state = "normal"; self.recovery_since = -1.0; self.max_drop = 0.0
        if event:
            self.event_id += 1; self.cooldown_until = o.timestamp + self.cfg.cooldown
        self.prev_hip, self.prev_time = o.hip_y, o.timestamp
        return self.result(o, speed, event)

    def result(self, o: Observation, speed: float, event: bool) -> dict:
        return {"state": self.state, "fall_detected": self.state in ("fallen", "recovering"),
                "fall_event": event, "event_id": self.event_id,
                "features": {"valid": o.valid, "hip_y": round(o.hip_y, 5),
                             "person_score": round(o.person_score, 4),
                             "hip_drop_speed": round(speed, 5),
                             "hip_drop_distance": round(self.max_drop, 5),
                             "torso_angle_deg": round(o.torso_angle, 3),
                             "bbox_aspect_ratio": round(o.aspect, 4),
                             "lying_posture": o.torso_angle >= self.cfg.torso_angle and o.aspect >= self.cfg.aspect,
                             "upright_posture": o.torso_angle <= self.cfg.recovery_angle and o.aspect <= self.cfg.recovery_aspect,
                             "in_cooldown": o.timestamp < self.cooldown_until},
                "temporal": {"available": o.temporal_available,
                             "positive": o.temporal_positive,
                             "probability": round(o.temporal_probability, 6)}}
