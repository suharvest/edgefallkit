"""BASE-1 AppHooks implementation for fall-detection."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure retail-vision / vision_base can be found if not installed in current env
try:
    from vision_base.hooks import AppHooks, Outgoing, StreamContext
    from vision_base.types import FrameResult, Detection as VbDetection, StreamSpec, Event
except ImportError:
    for candidate in [
        Path(__file__).resolve().parents[3] / "retail-vision" / "core-py",
        Path(__file__).resolve().parents[3] / "epv-public" / "third_party",
    ]:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from vision_base.hooks import AppHooks, Outgoing, StreamContext
        from vision_base.types import FrameResult, Detection as VbDetection, StreamSpec, Event
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class Outgoing:
            topic_suffix: str
            payload: dict | bytes
            qos: int = 0
            retain: bool = False

        class AppHooks:  # type: ignore[no-redef]
            pass

        class StreamContext:  # type: ignore[no-redef]
            pass

        class FrameResult:  # type: ignore[no-redef]
            pass

        class VbDetection:  # type: ignore[no-redef]
            pass

        class StreamSpec:  # type: ignore[no-redef]
            pass

        class Event:  # type: ignore[no-redef]
            pass

try:
    from .fall_core import (
        Detection as CoreDetection,
        Observation,
        make_observation,
        IoUTracker,
        TemporalMLP,
        FallConfig,
        FallDetector,
    )
except ImportError:
    from fall_core import (  # type: ignore[no-redef]
        Detection as CoreDetection,
        Observation,
        make_observation,
        IoUTracker,
        TemporalMLP,
        FallConfig,
        FallDetector,
    )

__all__ = [
    "FallApp",
    "build_payload",
    "CoreDetection",
    "Observation",
    "make_observation",
    "IoUTracker",
    "TemporalMLP",
    "FallConfig",
    "FallDetector",
    "Outgoing",
]


def build_payload(
    stream_id: str,
    frame_id: int,
    now: float,
    inference_ms: float,
    pipeline_ms: float,
    read_ms: float,
    people: list[dict],
    events: list[dict],
    global_event_id: int,
    source_backend: str | None = None,
    postprocess_backend: str | None = None,
) -> dict:
    """Build the shared reCamera-compatible MQTT document."""
    visible_people = [x for x in people if x.get("person_detected")]
    fallen = [x for x in people if x.get("state") in ("fallen", "recovering")]
    order = {"normal": 0, "suspected": 1, "recovering": 2, "fallen": 3}
    state = max((x["state"] for x in people), key=lambda x: order.get(x, 0), default="normal")
    primary = max(
        people,
        key=lambda x: (x.get("person_detected", False), x.get("person_score", 0.0)),
        default=None,
    )
    empty_features = {
        "valid": False,
        "hip_y": 0.0,
        "person_score": 0.0,
        "hip_drop_speed": 0.0,
        "hip_drop_distance": 0.0,
        "torso_angle_deg": 0.0,
        "bbox_aspect_ratio": 0.0,
        "lying_posture": False,
        "upright_posture": False,
        "in_cooldown": False,
    }
    return {
        "type": "fall_detection",
        "version": "0.3.0",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "timestamp": int(round(now * 1000.0)),
        "inference_time_ms": int(inference_ms + 0.5),
        "inference_ms": round(inference_ms, 3),
        "read_ms": round(read_ms, 3),
        "pipeline_ms": round(pipeline_ms, 3),
        "coordinate_space": "letterbox_model_input_normalized",
        "person_detected": bool(visible_people),
        "person_count": len(visible_people),
        "fallen_count": len(fallen),
        "tracking": bool(people),
        "persons": people,
        "state": state,
        "fall_detected": bool(fallen),
        "fall_event": bool(events),
        "event_id": global_event_id,
        "global_event_id": global_event_id,
        "event_id_scope": "stream_global_event_id",
        "features": primary["features"] if primary else empty_features,
        "keypoints": [],
        "pose17": (
            primary["pose17"]
            if primary and primary.get("person_detected") and len(primary.get("pose17", [])) == 17
            else []
        ),
        "events": events,
        "source_backend": source_backend,
        "postprocess_backend": postprocess_backend,
    }


class FallApp:
    name: str = "fall-detection"
    wants_frames: bool = True

    def __init__(self, options: dict | None = None, device_id: str = "") -> None:
        self.options: dict = {}
        self.device_id: str = ""
        self.temporal_model: str | None = None
        self.topic_template: str = "recomputer/fall-detection/results/{stream_id}"
        self.keypoint_threshold: float = 0.25
        self.fall_config: FallConfig = FallConfig()
        if options is not None:
            self.configure(options, device_id)

    def configure(self, options: dict, device_id: str = "") -> None:
        self.options = dict(options or {})
        self.device_id = device_id
        self.temporal_model = self.options.get("temporal_model")
        self.topic_template = self.options.get(
            "topic_template", "recomputer/fall-detection/results/{stream_id}"
        )
        self.keypoint_threshold = float(self.options.get("keypoint_threshold", 0.25))
        self.fall_config = FallConfig(**self.options.get("fall", {}))

    def analyzers(self, spec: StreamSpec) -> list[dict]:
        return []

    def plugins(self) -> list[str]:
        return []

    def on_stream_added(self, ctx: StreamContext) -> None:
        ctx.state["detectors"] = {}
        ctx.state["temporal"] = {}
        ctx.state["frame_id"] = 0
        ctx.state["global_event_id"] = 0

    def on_stream_removed(self, ctx: StreamContext) -> None:
        ctx.state.clear()

    def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]:
        return []

    def on_frame(self, ctx: StreamContext, res: FrameResult) -> list[Outgoing]:
        if "detectors" not in ctx.state:
            self.on_stream_added(ctx)

        now = res.wall_ms / 1000.0
        ctx.state["frame_id"] = ctx.state.get("frame_id", 0) + 1
        frame_id = ctx.state["frame_id"]

        people: list[dict] = []
        events: list[dict] = []

        for idx, d in enumerate(res.detections):
            # Keypoints handling
            kpts_raw = getattr(d, "keypoints", ())
            if len(kpts_raw) == 51:
                kpts = [
                    [
                        float(kpts_raw[i * 3]) * 640.0,
                        float(kpts_raw[i * 3 + 1]) * 640.0,
                        float(kpts_raw[i * 3 + 2]),
                    ]
                    for i in range(17)
                ]
            else:
                kpts = []

            # Box in pixels [x1, y1, x2, y2]
            cx = float(d.cx)
            cy = float(d.cy)
            w = float(d.w)
            h = float(d.h)
            score = float(d.score)
            box = [
                (cx - w / 2.0) * 640.0,
                (cy - h / 2.0) * 640.0,
                (cx + w / 2.0) * 640.0,
                (cy + h / 2.0) * 640.0,
            ]

            core_det = CoreDetection(box=box, score=score, keypoints=kpts)

            raw_tid = getattr(d, "track_id", 0)
            tid = raw_tid if (isinstance(raw_tid, int) and raw_tid > 0) else (idx + 1)

            if tid not in ctx.state["detectors"]:
                ctx.state["detectors"][tid] = FallDetector(self.fall_config)

            obs = make_observation(core_det, now, 640, 640, self.keypoint_threshold)

            if self.temporal_model:
                if tid not in ctx.state["temporal"]:
                    ctx.state["temporal"][tid] = TemporalMLP(self.temporal_model)
                pose_f = ctx.state["temporal"][tid].pose_frame(core_det, obs, 640, 640)
                pos, prob = ctx.state["temporal"][tid].update(pose_f, now)
                obs.temporal_available = True
                obs.temporal_positive = pos
                obs.temporal_probability = prob

            result = ctx.state["detectors"][tid].update(obs)

            if result["fall_event"]:
                ctx.state["global_event_id"] += 1
                events.append({
                    "kind": "fall",
                    "track_id": tid,
                    **result,
                    "per_track_event_id": result["event_id"],
                    "event_id": ctx.state["global_event_id"],
                    "global_event_id": ctx.state["global_event_id"],
                })

            pose17 = (
                [
                    [
                        round(p[0] / 640.0, 4),
                        round(p[1] / 640.0, 4),
                        round(p[2], 4),
                    ]
                    for p in kpts
                ]
                if len(kpts) == 17
                else []
            )

            bbox = [round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)]

            person_item = {
                "track_id": tid,
                "person_detected": True,
                "person_score": round(score, 4),
                "tracking": True,
                "missed_frames": 0,
                "bbox": bbox,
                "pose17": pose17,
                "keypoints": [],
                **result,
            }
            people.append(person_item)

        stream_id = getattr(ctx, "stream_id", getattr(res, "stream_id", "default"))
        inference_ms = getattr(res, "inference_ms", 0.0)
        read_ms = getattr(res, "queue_delay_ms", 0.0)
        pipeline_ms = inference_ms

        payload = build_payload(
            stream_id=stream_id,
            frame_id=frame_id,
            now=now,
            inference_ms=inference_ms,
            pipeline_ms=pipeline_ms,
            read_ms=read_ms,
            people=people,
            events=events,
            global_event_id=ctx.state["global_event_id"],
        )

        topic = self.topic_template.replace("{stream_id}", stream_id)
        # If full topic root was included in template, strip it for topic_suffix
        prefix = "recomputer/fall-detection/"
        if topic.startswith(prefix):
            topic = topic[len(prefix):]
        return [Outgoing(topic, payload)]
