"""Dependency-light multi-person fall logic shared by RK3576 and RK3588."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
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


def _midpoint(kp, a: int, b: int, threshold: float) -> Optional[tuple[float, float]]:
    """Average of joints a, b; one visible side alone; None when neither is."""
    va = len(kp[a]) >= 3 and kp[a][2] >= threshold
    vb = len(kp[b]) >= 3 and kp[b][2] >= threshold
    if va and vb:
        return ((kp[a][0] + kp[b][0]) * 0.5, (kp[a][1] + kp[b][1]) * 0.5)
    if va:
        return (kp[a][0], kp[a][1])
    if vb:
        return (kp[b][0], kp[b][1])
    return None


def make_observation(det: Optional[Detection], timestamp: float, width: int, height: int,
                     keypoint_threshold: float = 0.25) -> Observation:
    """Same rules as kit/logic/geometry.py make_observation (reCamera Pro)."""
    if det is None or width <= 0 or height <= 0 or len(det.keypoints) < 17:
        return Observation(timestamp=timestamp)
    invalid = Observation(timestamp=timestamp, person_score=det.score)
    x1, y1, x2, y2 = det.box
    if y2 - y1 <= 1e-4:
        return invalid
    kp = det.keypoints
    hip = _midpoint(kp, 11, 12, keypoint_threshold)
    shoulder = _midpoint(kp, 5, 6, keypoint_threshold)
    if hip is None or shoulder is None:
        return invalid
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return invalid
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    if not math.isfinite(angle):
        return invalid
    aspect = (x2 - x1) / (y2 - y1)
    return Observation(True, timestamp, hip[1] / height, angle, aspect, det.score)


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


# Radius of a "same place" match, as a multiple of the larger box side.
LOCATION_RADIUS_SCALE = 0.75
# Upper bound on remembered emitted events (expired ones are dropped first).
RECENT_FALLS_MAX = 32


def boxes_near(a, b) -> bool:
    """Box centres (x1, y1, x2, y2) within 0.75 x the larger box side."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    side = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 0.0)
    dx = (ax1 + ax2 - bx1 - bx2) * 0.5
    dy = (ay1 + ay2 - by1 - by2) * 0.5
    return math.hypot(dx, dy) <= LOCATION_RADIUS_SCALE * side


class LocationCooldown:
    """Suppress a fall edge that re-detects a recent event under a new id.

    Same rule as reCamera Pro apps/fall-detection/app.py: an edge is
    suppressed only when an event EMITTED within the last ``window_sec`` lies
    nearby AND that event's track is not visible in the current frame (its
    identity was lost).  An event whose track is still visible, or one from
    this same frame, is a different person.  The window runs from the emitted
    event only; suppressed edges do not extend it.
    """

    def __init__(self, window_sec: float = 10.0):
        self.window_sec = max(0.0, float(window_sec))
        self.recent: list = []

    def hit(self, track_id, box, now: float, visible_ids) -> bool:
        if self.window_sec <= 0.0:
            self.recent = []
            return False
        self.recent = [f for f in self.recent if 0.0 <= now - f[0] <= self.window_sec]
        return any(ts < now and tid != track_id and tid not in visible_ids and boxes_near(b, box)
                   for ts, b, tid in self.recent)

    def remember(self, track_id, box, now: float) -> None:
        if self.window_sec <= 0.0:
            return
        self.recent.append((now, list(box), track_id))
        del self.recent[:-RECENT_FALLS_MAX]


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
        self.reset()

    def reset(self) -> None:
        self.frames.clear()
        self.last_time = -1.0
        self.last_eval = -1.0
        self.positive_run = 0
        self.probability = 0.0
        self.positive = False

    @staticmethod
    def pose_frame(det: Optional[Detection], obs: Observation, width: int, height: int) -> np.ndarray:
        """56-value pelvis-centred COCO-17 frame, as reCamera Pro app.py.

        A detected person whose pose is not a valid observation still fills
        the keypoint slots; only the valid flag (last value) is 0."""
        out = np.zeros(56, np.float32)
        if det is None or width <= 0 or height <= 0 or len(det.keypoints) < 17:
            return out
        try:
            pts = np.asarray(det.keypoints[:17], np.float32)
        except (TypeError, ValueError):
            return out
        if pts.shape != (17, 3) or not np.all(np.isfinite(pts)):
            return out
        conf = np.clip(pts[:, 2], 0, 1)
        xy = pts[:, :2] / np.array([width, height], np.float32)

        def midpoint(a: int, b: int):
            weight = float(conf[a] + conf[b])
            if weight < 0.1:
                return np.zeros(2, np.float32), 0.0
            return (xy[a] * conf[a] + xy[b] * conf[b]) / weight, weight * 0.5

        hip, hip_conf = midpoint(11, 12)
        shoulder, shoulder_conf = midpoint(5, 6)
        visible = conf >= 0.1
        if hip_conf < 0.1:
            weight = float(conf[visible].sum())
            if weight <= 0.0:
                return out
            hip = (xy[visible] * conf[visible, None]).sum(axis=0) / weight
        scale = float(np.linalg.norm(shoulder - hip)) if shoulder_conf >= 0.1 else 0.0
        if scale < 0.04 and visible.any():
            scale = float(np.ptp(xy[visible], axis=0).max()) * 0.35
        scale = max(scale, 0.04)
        centered = np.clip((xy - hip) / scale, -4, 4)
        out[:34].reshape(17, 2)[visible] = centered[visible]
        out[34:51] = conf
        out[-5:] = [obs.hip_y, obs.torso_angle / 90, min(obs.aspect, 4) / 4,
                    obs.person_score, 1.0 if obs.valid else 0.0]
        return out

    def update(self, frame: np.ndarray, now: float) -> tuple[bool, float]:
        now = float(now)
        if self.last_time >= 0.0 and now < self.last_time:
            self.reset()
        self.last_time = now
        self.frames.append((now, frame * self.frame_mask))
        cutoff = now - (self.window - 1) / 15.0 - 0.5
        while len(self.frames) > 1 and self.frames[1][0] < cutoff:
            self.frames.popleft()
        if self.last_eval >= 0 and now - self.last_eval < 0.2 - 1e-6:
            return self.positive, self.probability
        self.last_eval = now
        ts = np.array([x[0] for x in self.frames])
        values = np.stack([x[1] for x in self.frames])
        wanted = now - np.arange(self.window - 1, -1, -1, dtype=np.float64) / 15.0
        seq = values[np.searchsorted(ts, wanted + 1e-6, side="right").clip(1) - 1]
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
    min_suspected_features: int = 2
    confirmation: float = 0.8
    suspected_timeout: float = 1.5
    occlusion_grace: float = 0.75
    # Extra time after suspected_timeout during which a still-lying candidate
    # may be confirmed by a late temporal positive (48-frame window at 15 fps).
    late_confirmation: float = 3.2
    recovery_angle: float = 35.0
    recovery_aspect: float = 1.1
    recovery_window: float = 2.0
    cooldown: float = 3.0

    def clamp(self) -> "FallConfig":
        """Same bounds as kit/logic/temporal.py FallConfig.clamp()."""
        self.hip_drop_speed = max(0.0, self.hip_drop_speed)
        self.hip_drop_distance = max(0.0, self.hip_drop_distance)
        self.motion_window = max(0.0, self.motion_window)
        self.torso_angle = min(89.0, max(1.0, self.torso_angle))
        self.aspect = max(1.0, self.aspect)
        self.min_suspected_features = min(3, max(1, int(self.min_suspected_features)))
        self.confirmation = max(0.0, self.confirmation)
        self.suspected_timeout = max(self.confirmation, self.suspected_timeout)
        self.occlusion_grace = max(0.0, self.occlusion_grace)
        self.recovery_angle = min(89.0, max(1.0, self.recovery_angle))
        self.recovery_aspect = max(1.0, self.recovery_aspect)
        self.recovery_window = max(0.0, self.recovery_window)
        self.cooldown = max(0.0, self.cooldown)
        self.late_confirmation = max(0.0, self.late_confirmation)
        return self


class FallDetector:
    """normal -> suspected -> fallen -> recovering -> normal.

    Only geometry arms ``suspected``: a horizontal cue plus either a fast hip
    drop within ``motion_window`` or, across an occluded gap, a hip
    displacement of at least ``hip_drop_distance`` below the last upright
    baseline seen at most ``motion_window + occlusion_grace`` earlier.  A
    temporal-positive result alone never leaves ``normal``; it confirms only a
    ``suspected`` candidate whose current valid pose is lying with at least
    ``min_suspected_features`` evidence features (the arming motion counts).
    """

    def __init__(self, cfg: FallConfig):
        # Per-detector copy: callers may share one FallConfig across tracks.
        self.cfg = replace(cfg).clamp(); self.state = "normal"; self.event_id = 0
        self.prev_hip = None; self.prev_time = None
        self.baseline = None; self.baseline_time = -1.0; self.gap_since_baseline = False
        self.last_drop = -1.0; self.suspected_since = -1.0
        self.last_strong = -1.0; self.motion_triggered = False
        self.max_drop = 0.0; self.recovery_since = -1.0; self.cooldown_until = -1.0

    def _to_normal(self):
        self.state = "normal"; self.suspected_since = -1.0
        self.last_strong = -1.0; self.motion_triggered = False
        self.last_drop = -1.0; self.max_drop = 0.0; self.recovery_since = -1.0

    def _in_late_latch(self, timestamp: float) -> bool:
        return (self.motion_triggered and self.suspected_since >= 0.0 and
                timestamp - self.suspected_since <= self.cfg.suspected_timeout + self.cfg.late_confirmation)

    def update(self, o: Observation) -> dict:
        cfg = self.cfg
        event = False; speed = 0.0
        if not o.valid:
            # Hard invariant: cached temporal output or disappearance can retain state,
            # but an invalid/missing observation can never originate a fall event.
            if self.baseline is not None:
                self.gap_since_baseline = True
            if self.state == "suspected" and self.suspected_since >= 0.0:
                suspected_for = o.timestamp - self.suspected_since
                recent_strong = (self.last_strong >= 0.0 and
                                 o.timestamp - self.last_strong <= cfg.occlusion_grace)
                if (suspected_for > cfg.suspected_timeout and
                        not (recent_strong and self._in_late_latch(o.timestamp))):
                    # A late-confirmation candidate survives only a short
                    # occlusion right after a lying frame.  Missing/invalid
                    # observations never originate an event.
                    self._to_normal()
            return self.result(o, speed, event)
        if self.prev_time is not None and 1e-4 < o.timestamp - self.prev_time < 10:
            speed = (o.hip_y - self.prev_hip) / (o.timestamp - self.prev_time)
        horizontal = o.torso_angle >= cfg.torso_angle or o.aspect >= cfg.aspect
        lying = o.torso_angle >= cfg.torso_angle and o.aspect >= cfg.aspect
        upright = o.torso_angle <= cfg.recovery_angle and o.aspect <= cfg.recovery_aspect
        if self.state == "normal" and not horizontal:
            self.baseline = o.hip_y; self.baseline_time = o.timestamp
            self.gap_since_baseline = False
        if self.state == "suspected" and self.baseline is not None:
            self.max_drop = max(self.max_drop, o.hip_y - self.baseline)
        if speed >= cfg.hip_drop_speed:
            self.last_drop = o.timestamp
        # Confirmation evidence: posture features plus the motion feature,
        # which a motion-armed suspected candidate has latched (a victim lying
        # still after impact has ~0 current hip speed).
        motion = speed >= cfg.hip_drop_speed or (self.state == "suspected" and self.motion_triggered)
        confirmation_features = (int(o.torso_angle >= cfg.torso_angle) +
                                 int(o.aspect >= cfg.aspect) + int(motion))
        enough = confirmation_features >= cfg.min_suspected_features
        cooldown = o.timestamp < self.cooldown_until
        if self.state == "normal":
            # Geometry is the only way out of normal.  A temporal-positive
            # window on its own (e.g. somebody sitting down quickly) must not
            # raise an alarm without hip-drop + horizontal arming.
            fast_drop = self.last_drop >= 0 and o.timestamp - self.last_drop <= cfg.motion_window
            displaced = (self.baseline is not None and self.gap_since_baseline and
                         o.timestamp - self.baseline_time <= cfg.motion_window + cfg.occlusion_grace and
                         o.hip_y - self.baseline >= cfg.hip_drop_distance)
            if not cooldown and horizontal and (fast_drop or displaced):
                self.state = "suspected"
                self.suspected_since = self.last_drop if fast_drop else o.timestamp
                self.last_strong = o.timestamp if lying else -1.0
                self.motion_triggered = True
                self.max_drop = max(0.0, o.hip_y - (self.baseline if self.baseline is not None else o.hip_y))
        elif self.state == "suspected":
            if lying and enough:
                self.last_strong = o.timestamp
            if (not cooldown and o.temporal_available and o.temporal_positive and lying and enough):
                # Learned confirmation also needs the current pose lying.
                self.state = "fallen"; event = True
            elif (not cfg.temporal_confirmation_required and self.motion_triggered and lying and
                  enough and self.max_drop >= cfg.hip_drop_distance and
                  o.timestamp - self.suspected_since >= cfg.confirmation):
                self.state = "fallen"; event = True
            elif (upright or (o.timestamp - self.suspected_since > cfg.suspected_timeout and
                              not (lying and self._in_late_latch(o.timestamp)))):
                # Past the timeout only a still-lying candidate inside the
                # bounded late-confirmation latch is kept.
                self._to_normal()
        elif self.state == "fallen" and upright:
            if cfg.recovery_window <= 0.0:
                self._to_normal()
            else:
                self.state = "recovering"; self.recovery_since = o.timestamp
        elif self.state == "recovering":
            if not upright:
                self.state = "fallen"; self.recovery_since = -1.0
            elif o.timestamp - self.recovery_since >= cfg.recovery_window:
                self._to_normal()
        if event:
            self.event_id += 1; self.cooldown_until = o.timestamp + cfg.cooldown
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
