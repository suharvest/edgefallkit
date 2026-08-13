#!/usr/bin/env python3
"""Multi-RTSP RKNN fall detector with hardware video and postprocess fallbacks."""
from __future__ import annotations

import argparse, json, os, signal, socket, ssl, struct, threading, time
from pathlib import Path

from fall_core import Detection, FallConfig, FallDetector, IoUTracker, TemporalMLP, make_observation
from rknn_pose import PoseDecoder, RKNNPose
from video_source import create_video_source, validate_backend_config

RUNNING = True


def _remaining(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n % 128; n //= 128
        if n: byte |= 128
        out.append(byte)
        if not n: return bytes(out)


def _mqtt_string(value: str) -> bytes:
    data = value.encode(); return struct.pack("!H", len(data)) + data


class MqttPublisher:
    """Small MQTT 3.1.1 QoS0 client, avoiding paho in the deployment image."""
    def __init__(self, cfg):
        self.cfg = cfg; self.sock = None; self.lock = threading.Lock()

    def connect(self):
        raw = socket.create_connection((self.cfg["host"], int(self.cfg.get("port", 1883))), 5)
        if self.cfg.get("tls"):
            ctx = ssl.create_default_context(cafile=self.cfg.get("ca_file") or None); raw = ctx.wrap_socket(raw, server_hostname=self.cfg["host"])
        client = self.cfg.get("client_id") or f"fall-rknn-{os.getpid()}"
        flags = 2; payload = _mqtt_string(client)
        if self.cfg.get("username"):
            flags |= 0x80; payload += _mqtt_string(self.cfg["username"])
        if self.cfg.get("password"):
            flags |= 0x40; payload += _mqtt_string(self.cfg["password"])
        variable = _mqtt_string("MQTT") + bytes((4, flags)) + struct.pack("!H", int(self.cfg.get("keepalive_sec", 30)))
        packet = variable + payload; raw.sendall(b"\x10" + _remaining(len(packet)) + packet)
        if raw.recv(4)[-1:] != b"\x00": raise ConnectionError("MQTT CONNACK rejected")
        self.sock = raw

    def publish(self, topic: str, payload: dict):
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        body = _mqtt_string(topic) + data; packet = b"\x30" + _remaining(len(body)) + body
        with self.lock:
            for attempt in range(2):
                try:
                    if self.sock is None: self.connect()
                    self.sock.sendall(packet); return
                except OSError:
                    if self.sock:
                        try: self.sock.close()
                        except OSError: pass
                    self.sock = None
                    if attempt: raise


def build_payload(stream_id, frame_id, now, inference_ms, pipeline_ms,
                  people, events, global_event_id, source_backend=None,
                  postprocess_backend=None):
    """Build the shared reCamera-compatible MQTT document."""
    visible_people = [x for x in people if x["person_detected"]]
    fallen = [x for x in people if x["fall_detected"]]
    order = {"normal":0,"suspected":1,"recovering":2,"fallen":3}
    state = max((x["state"] for x in people), key=lambda x:order[x], default="normal")
    primary = max(people, key=lambda x:(x["person_detected"],x["person_score"]), default=None)
    empty_features = {"valid":False,"hip_y":0.0,"person_score":0.0,"hip_drop_speed":0.0,
                      "hip_drop_distance":0.0,"torso_angle_deg":0.0,"bbox_aspect_ratio":0.0,
                      "lying_posture":False,"upright_posture":False,"in_cooldown":False}
    return {"type":"fall_detection","version":"0.3.0","stream_id":stream_id,
            "frame_id":frame_id,"timestamp":int(now*1000),
            "inference_time_ms":int(inference_ms+.5),"inference_ms":round(inference_ms,3),
            "pipeline_ms":round(pipeline_ms,3),"coordinate_space":"letterbox_model_input_normalized",
            "person_detected":bool(visible_people),"person_count":len(visible_people),
            "fallen_count":len(fallen),"tracking":bool(people),"persons":people,
            "state":state,"fall_detected":bool(fallen),"fall_event":bool(events),
            "event_id":global_event_id,"global_event_id":global_event_id,
            "event_id_scope":"stream_global_event_id",
            "features":primary["features"] if primary else empty_features,
            "keypoints":[],"pose17":primary["pose17"] if primary and primary["person_detected"] else [],
            "events":events,
            "source_backend":source_backend,"postprocess_backend":postprocess_backend}


class StreamWorker(threading.Thread):
    def __init__(self, cfg, stream, publisher):
        super().__init__(name=f"stream-{stream['id']}", daemon=True)
        self.cfg = cfg; self.stream = stream; self.publisher = publisher

    def run(self):
        model = RKNNPose(self.cfg["model_path"], self.stream.get("core_mask"))
        tracker = IoUTracker(float(self.cfg["tracker"].get("iou_threshold", .2)),
                             float(self.cfg["tracker"].get("max_lost_sec", .75)))
        source = create_video_source(self.stream, self.cfg, int(self.cfg.get("input_size", 640)))
        decoder = PoseDecoder(self.cfg.get("postprocess", {}))
        print(f"[{self.stream['id']}] video={source.active_backend} postprocess={decoder.active_backend}", flush=True)
        detectors = {}; temporal = {}; frame_id = 0; global_event_id = 0
        try:
            while RUNNING:
                frame = source.read()
                if frame is None:
                    time.sleep(float(self.stream.get("reconnect_delay_ms", 1000)) / 1000); continue
                start = time.perf_counter(); now = time.time()
                outputs, inference_ms = model.infer(frame)
                raw = decoder.decode(outputs, float(self.cfg.get("score_threshold", .35)),
                                     float(self.cfg.get("nms_threshold", .45)), frame.shape[0])
                detections = [Detection(x["box"], x["score"], x["keypoints"]) for x in raw]
                tracks = tracker.update(detections, now)
                for tid in tracker.expired_ids: detectors.pop(tid, None); temporal.pop(tid, None)
                people = []; events = []
                for tr in tracks:
                    if tr.track_id not in detectors:
                        detectors[tr.track_id] = FallDetector(FallConfig(**self.cfg.get("fall", {})))
                    obs = make_observation(tr.detection, now, frame.shape[1], frame.shape[0],
                                           float(self.cfg.get("keypoint_threshold", .25)))
                    if self.cfg.get("temporal_model"):
                        if tr.track_id not in temporal:
                            temporal[tr.track_id] = TemporalMLP(self.cfg["temporal_model"])
                        pose_frame = temporal[tr.track_id].pose_frame(tr.detection, obs, frame.shape[1], frame.shape[0])
                        positive, probability = temporal[tr.track_id].update(pose_frame, now)
                        obs.temporal_available = True; obs.temporal_positive = positive; obs.temporal_probability = probability
                    result = detectors[tr.track_id].update(obs)
                    visible = tr.detection is not None
                    x1,y1,x2,y2 = tr.box
                    pose17 = [[round(p[0]/frame.shape[1],4),round(p[1]/frame.shape[0],4),round(p[2],4)]
                              for p in tr.detection.keypoints] if visible else []
                    item = {"track_id": tr.track_id, "person_detected": visible,
                            "person_score": round(tr.detection.score,4) if visible else 0.0,
                            "tracking": visible, "missed_frames": tr.missed,
                            "bbox": [round((x1+x2)/2/frame.shape[1],4),round((y1+y2)/2/frame.shape[0],4),
                                     round((x2-x1)/frame.shape[1],4),round((y2-y1)/frame.shape[0],4)],
                            "pose17": pose17, "keypoints": [], **result}
                    people.append(item)
                    if result["fall_event"]:
                        global_event_id += 1
                        events.append({"kind": "fall", "track_id": tr.track_id, **result,
                                       "per_track_event_id": result["event_id"],
                                       "event_id": global_event_id,
                                       "global_event_id": global_event_id})
                frame_id += 1
                payload = build_payload(self.stream["id"], frame_id, now, inference_ms,
                                        (time.perf_counter()-start)*1000, people, events,
                                        global_event_id, source.active_backend,
                                        decoder.active_backend)
                topic = self.cfg["mqtt"]["topic"].replace("{stream_id}", self.stream["id"])
                try: self.publisher.publish(topic, payload)
                except OSError as exc: print(f"[{self.stream['id']}] MQTT: {exc}", flush=True)
        finally:
            source.close(); model.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="/config/config.json"); ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text())
    required = ("model_path", "mqtt", "streams")
    if any(k not in cfg for k in required): raise SystemExit(f"missing required config keys: {required}")
    validate_backend_config(cfg)
    if args.validate:
        print(json.dumps({"valid": True, "streams": len(cfg["streams"]),
                          "video": cfg.get("video", {}),
                          "postprocess": cfg.get("postprocess", {})})); return
    global RUNNING
    def stop(*_):
        global RUNNING; RUNNING = False
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    publisher = MqttPublisher(cfg["mqtt"])
    workers = [StreamWorker(cfg, x, publisher) for x in cfg["streams"] if x.get("enabled", True)]
    for worker in workers: worker.start()
    while RUNNING and any(x.is_alive() for x in workers): time.sleep(.5)
    for worker in workers: worker.join(timeout=5)


if __name__ == "__main__": main()
