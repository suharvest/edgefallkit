#!/usr/bin/env python3
"""Multi-RTSP RKNN fall detector with hardware video and postprocess fallbacks."""
from __future__ import annotations

import argparse, json, os, queue, signal, socket, ssl, struct, threading, time
from dataclasses import dataclass
from pathlib import Path

from fall_core import Detection, FallConfig, FallDetector, IoUTracker, TemporalMLP, make_observation
from rknn_pose import PoseDecoder, RKNNPose
from video_source import create_video_source, frame_canvas, validate_backend_config

RUNNING = True


@dataclass(frozen=True)
class CapturedFrame:
    """A frame owned by the inference queue; timestamp is from capture time."""
    frame: object
    timestamp: float
    read_ms: float
    sequence: int
    source_backend: str | None = None


class LatestFrameQueue:
    """Bounded depth-one queue. A slow inference context never builds latency."""
    def __init__(self):
        self._item = None
        self._cv = threading.Condition()
        self.dropped = 0
        self.closed = False

    def put(self, item):
        with self._cv:
            if self.closed:
                _close_frame(item)
                return False
            if self._item is not None:
                self.dropped += 1
                _close_frame(self._item)
            self._item = item
            self._cv.notify()
            return True

    def get(self, timeout=0.0):
        with self._cv:
            if self._item is None and timeout:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def close(self):
        with self._cv:
            self.closed = True
            item, self._item = self._item, None
            self._cv.notify_all()
        _close_frame(item)


def _close_frame(item):
    frame = getattr(item, "frame", item)
    close = getattr(frame, "close", None)
    if close is not None:
        try: close()
        except Exception: pass


def resolve_context_core_masks(config, context_count):
    """Resolve optional per-context masks without prescribing platform values."""
    context_count = int(context_count)
    if context_count <= 0:
        raise ValueError("context_pool_size must be a positive integer")
    masks = config.get("context_core_masks")
    scalar = config.get("core_mask", "auto")
    if masks is not None:
        if not isinstance(masks, list):
            raise ValueError("context_core_masks must be a JSON list")
        if scalar != "auto":
            raise ValueError("context_core_masks cannot be combined with scalar core_mask")
        if len(masks) != context_count or any(int(mask) <= 0 for mask in masks):
            raise ValueError("context_core_masks must contain one positive integer per context")
        return [int(mask) for mask in masks]
    if scalar == "auto":
        return [None] * context_count
    mask = int(scalar, 0) if isinstance(scalar, str) else int(scalar)
    if mask <= 0:
        raise ValueError("core_mask must be auto or a positive integer")
    return [mask] * context_count


def resolve_legacy_core_mask(config, stream, compatible_reader=None, legacy_single_stream=True):
    """Resolve legacy masks, with the measured RK3576 single-stream policy."""
    value = stream.get("core_mask", config.get("core_mask", "auto"))
    if value != "auto" and value is not None:
        mask = int(value, 0) if isinstance(value, str) else int(value)
        if mask <= 0:
            raise ValueError("core_mask must be auto or a positive integer")
        return mask
    if not legacy_single_stream or config.get("core_policy", "measured_auto") == "runtime_auto":
        return None
    reader = compatible_reader or (lambda: Path("/proc/device-tree/compatible").read_bytes())
    try:
        compatible = reader()
        if isinstance(compatible, bytes):
            compatible = compatible.decode("utf-8", errors="ignore")
    except (OSError, IOError):
        compatible = ""
    return 3 if "rk3576" in str(compatible).lower() else None


class ContextPoolRuntime:
    """Fair batch-1 context pool for multi-stream RKNN inference.

    Capture and inference are deliberately separate. Each stream has one latest
    frame, while each context owns exactly one RKNN runtime and processes one
    frame at a time. This does not claim RKNN batch or asynchronous support.
    """
    def __init__(self, streams, context_count, source_factory, model_factory,
                 frame_handler):
        enabled = [s for s in streams if s.get("enabled", True)]
        if not enabled:
            raise ValueError("at least one enabled stream is required")
        ids = [s.get("id") for s in enabled]
        if any(not stream_id for stream_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("enabled stream IDs must be present and unique")
        self.streams = enabled
        self.context_count = int(context_count)
        if self.context_count <= 0:
            raise ValueError("context_pool_size must be a positive integer")
        self.source_factory = source_factory
        self.model_factory = model_factory
        self.frame_handler = frame_handler
        self.queues = {s["id"]: LatestFrameQueue() for s in enabled}
        self.sources = {}; self.models = []
        self._stop = threading.Event(); self._threads = []
        self._model_threads = {}; self._started_model_ids = set()
        self._cursor = 0; self._lock = threading.Lock()
        self._busy = set(); self._closed_sources = set(); self.exception = None
        self._closed_models = set()

    def _next(self):
        # Round-robin over stream IDs, never waiting for a particular stream.
        for _ in range(len(self.streams)):
            with self._lock:
                stream = self.streams[self._cursor % len(self.streams)]
                self._cursor = (self._cursor + 1) % len(self.streams)
                stream_id = stream["id"]
                if stream_id in self._busy:
                    continue
                # Reserve before touching the queue: another context must not
                # claim a frame arriving in this small scheduling window.
                self._busy.add(stream_id)
            item = self.queues[stream_id].get(0)
            if item is not None:
                return stream, item
            with self._lock:
                self._busy.discard(stream_id)
        return None, None

    def _capture(self, stream):
        try:
            source = self.source_factory(stream); self.sources[stream["id"]] = source
            sequence = 0
            while not self._stop.is_set():
                started = time.monotonic(); frame = source.read()
                read_ms = (time.monotonic() - started) * 1000.0
                if frame is None:
                    delay = float(stream.get("reconnect_delay_ms", 1000)) / 1000.0
                    if self._stop.wait(max(0.0, delay)):
                        break
                    continue
                sequence += 1
                self.queues[stream["id"]].put(CapturedFrame(frame, time.time(), read_ms, sequence,
                                                             getattr(source, "active_backend", None)))
        except BaseException as exc:
            self._fail(exc)
        finally:
            source = self.sources.get(stream["id"])
            self._close_source(stream["id"], source)

    def _infer(self, model):
        try:
            while not self._stop.is_set():
                stream, item = self._next()
                if item is None:
                    time.sleep(0.001)
                    continue
                try:
                    self.frame_handler(stream, model, item)
                finally:
                    _close_frame(item)
                    with self._lock: self._busy.discard(stream["id"])
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._close_model(model)

    def _fail(self, exc):
        self._record_exception(exc)
        self._stop.set()
        for queue in self.queues.values(): queue.close()

    def _record_exception(self, exc):
        with self._lock:
            if self.exception is None: self.exception = exc

    def _close_source(self, stream_id, source):
        if source is None: return
        with self._lock:
            if stream_id in self._closed_sources: return
            self._closed_sources.add(stream_id)
        try:
            source.close()
        except BaseException as exc:
            self._record_exception(exc)

    def _close_model(self, model):
        key = id(model)
        with self._lock:
            if key in self._closed_models: return
            self._closed_models.add(key)
        try:
            model.close()
        except BaseException as exc:
            self._record_exception(exc)

    def start(self):
        try:
            for stream in self.streams:
                t = threading.Thread(target=self._capture, args=(stream,),
                                     name=f"capture-{stream['id']}", daemon=True)
                t.start(); self._threads.append(t)
            for index in range(self.context_count):
                model = self.model_factory(index); self.models.append(model)
                t = threading.Thread(target=self._infer, args=(model,),
                                     name=f"rknn-context-{index}", daemon=True)
                t.start()
                self._threads.append(t)
                self._model_threads[id(model)] = t
                self._started_model_ids.add(id(model))
        except BaseException:
            self.stop()
            raise

    def stop(self, timeout=5):
        self._stop.set()
        for queue in self.queues.values(): queue.close()
        for stream_id, source in list(self.sources.items()):
            self._close_source(stream_id, source)
        for thread in self._threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._record_exception(TimeoutError(f"context thread did not stop: {thread.name}"))
        # A model whose context thread never started has no owner that can
        # release it. Never release a model while its live native thread may
        # still be using it; that is a use-after-release hazard.
        for model in self.models:
            if id(model) not in self._started_model_ids:
                self._close_model(model)
        self._threads.clear()


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
        self.cfg = cfg; self.sock = None
        self.outbox = queue.Queue(maxsize=max(8, int(cfg.get("queue_size", 256))))
        self.stop_event = threading.Event(); self.exception = None
        self.state_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="mqtt-publisher", daemon=True)
        self.thread.start()

    def connect(self):
        raw = socket.create_connection((self.cfg["host"], int(self.cfg.get("port", 1883))), 5)
        try:
            # Each QoS0 publish is a small packet. Nagle plus delayed ACK can
            # cap a single connection well below the multi-stream frame rate.
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
            with self.state_lock:
                if self.stop_event.is_set():
                    raise RuntimeError("MQTT publisher closed during connect")
                self.sock = raw
        except BaseException:
            try: raw.close()
            except OSError: pass
            raise

    def publish(self, topic: str, payload: dict):
        with self.state_lock:
            if self.exception is not None:
                raise RuntimeError("MQTT publisher failed") from self.exception
            if self.stop_event.is_set():
                raise RuntimeError("MQTT publisher is closed")
            try:
                self.outbox.put_nowait((topic, payload))
            except queue.Full as exc:
                raise RuntimeError("MQTT publisher queue is full") from exc

    def _send(self, topic: str, payload: dict):
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        body = _mqtt_string(topic) + data; packet = b"\x30" + _remaining(len(body)) + body
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

    def _run(self):
        try:
            while not self.stop_event.is_set() or not self.outbox.empty():
                try: item = self.outbox.get(timeout=.1)
                except queue.Empty: continue
                try:
                    self._send(*item)
                except BaseException as exc:
                    with self.state_lock:
                        self.exception = exc
                        self.stop_event.set()
                    print(f"[mqtt] publisher: {exc}", flush=True)
                    while True:
                        try: self.outbox.get_nowait(); self.outbox.task_done()
                        except queue.Empty: break
                    return
                finally:
                    self.outbox.task_done()
        finally:
            self._close_socket()

    def _close_socket(self):
        with self.state_lock:
            sock, self.sock = self.sock, None
        if sock:
            try: sock.close()
            except OSError: pass

    def close(self, timeout=12.0):
        with self.state_lock:
            self.stop_event.set()
        self.thread.join(timeout)
        self._close_socket()
        if self.thread.is_alive():
            self.thread.join(1.0)
            if self.thread.is_alive():
                raise TimeoutError("MQTT publisher did not stop")
        if self.exception is not None:
            raise RuntimeError("MQTT publisher failed") from self.exception


def make_stream_publishers(cfg, streams):
    """Create one MQTT connection per stream to avoid a shared send bottleneck."""
    publishers = {}
    try:
        for stream in streams:
            stream_id = stream["id"]
            stream_cfg = dict(cfg)
            base_id = stream_cfg.get("client_id") or f"fall-rknn-{os.getpid()}"
            stream_cfg["client_id"] = f"{base_id}-{stream_id}"
            publishers[stream_id] = MqttPublisher(stream_cfg)
        return publishers
    except BaseException:
        for publisher in publishers.values():
            try: publisher.close()
            except BaseException: pass
        raise


def close_publishers(publishers):
    error = None
    for publisher in publishers.values():
        try: publisher.close()
        except BaseException as exc:
            if error is None: error = exc
    if error is not None: raise error


def build_payload(stream_id, frame_id, now, inference_ms, pipeline_ms, read_ms,
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
            "inference_time_ms":int(inference_ms+.5),"inference_ms":round(inference_ms,3),"read_ms":round(read_ms,3),
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
    def __init__(self, cfg, stream, publisher, stream_count=1):
        super().__init__(name=f"stream-{stream['id']}", daemon=True)
        self.cfg = cfg; self.stream = stream; self.publisher = publisher; self.stream_count = stream_count
        self.tracker = IoUTracker(float(cfg.get("tracker", {}).get("iou_threshold", .2)),
                                  float(cfg.get("tracker", {}).get("max_lost_sec", .75)))
        self.detectors = {}; self.temporal = {}; self.frame_id = 0; self.global_event_id = 0
        self.exception = None; self.source = None; self.source_lock = threading.Lock()

    def stop(self):
        with self.source_lock:
            source = self.source
        if source is not None:
            try: source.close()
            except BaseException as exc:
                if self.exception is None: self.exception = exc

    def process_frame(self, model, frame, now, read_ms, decoder=None, source_backend=None):
        """Run one captured frame, retaining all state on this stream."""
        decoder = decoder or PoseDecoder(self.cfg.get("postprocess", {}))
        start = time.perf_counter()
        try:
            outputs, inference_ms = model.infer(frame)
        finally:
            _close_frame(frame)
        canvas_w, canvas_h = frame_canvas(frame)
        raw = decoder.decode(outputs, float(self.cfg.get("score_threshold", .35)),
                             float(self.cfg.get("nms_threshold", .45)), canvas_h)
        detections = [Detection(x["box"], x["score"], x["keypoints"]) for x in raw]
        tracker = self.tracker
        tracks = tracker.update(detections, now)
        for tid in tracker.expired_ids:
            self.detectors.pop(tid, None); self.temporal.pop(tid, None)
        people = []; events = []
        for tr in tracks:
            if tr.track_id not in self.detectors:
                self.detectors[tr.track_id] = FallDetector(FallConfig(**self.cfg.get("fall", {})))
            obs = make_observation(tr.detection, now, canvas_w, canvas_h,
                                   float(self.cfg.get("keypoint_threshold", .25)))
            if self.cfg.get("temporal_model"):
                if tr.track_id not in self.temporal:
                    self.temporal[tr.track_id] = TemporalMLP(self.cfg["temporal_model"])
                pose_frame = self.temporal[tr.track_id].pose_frame(tr.detection, obs, canvas_w, canvas_h)
                positive, probability = self.temporal[tr.track_id].update(pose_frame, now)
                obs.temporal_available = True; obs.temporal_positive = positive; obs.temporal_probability = probability
            result = self.detectors[tr.track_id].update(obs)
            visible = tr.detection is not None; x1,y1,x2,y2 = tr.box
            pose17 = [[round(p[0]/canvas_w,4),round(p[1]/canvas_h,4),round(p[2],4)]
                      for p in tr.detection.keypoints] if visible else []
            item = {"track_id": tr.track_id, "person_detected": visible,
                    "person_score": round(tr.detection.score,4) if visible else 0.0,
                    "tracking": visible, "missed_frames": tr.missed,
                    "bbox": [round((x1+x2)/2/canvas_w,4),round((y1+y2)/2/canvas_h,4),
                             round((x2-x1)/canvas_w,4),round((y2-y1)/canvas_h,4)],
                    "pose17": pose17, "keypoints": [], **result}
            people.append(item)
            if result["fall_event"]:
                self.global_event_id += 1
                events.append({"kind": "fall", "track_id": tr.track_id, **result,
                               "per_track_event_id": result["event_id"], "event_id": self.global_event_id,
                               "global_event_id": self.global_event_id})
        self.frame_id += 1
        payload = build_payload(self.stream["id"], self.frame_id, now, inference_ms,
                                (time.perf_counter()-start)*1000, read_ms, people, events,
                                self.global_event_id, source_backend, decoder.active_backend)
        topic = self.cfg["mqtt"]["topic"].replace("{stream_id}", self.stream["id"])
        self.publisher.publish(topic, payload)

    def run(self):
        global RUNNING
        model = source = None
        try:
            model = RKNNPose(self.cfg["model_path"], resolve_legacy_core_mask(
                self.cfg, self.stream, legacy_single_stream=self.stream_count == 1),
                self.cfg.get("rknn", {}))
            self.tracker = IoUTracker(float(self.cfg["tracker"].get("iou_threshold", .2)),
                                 float(self.cfg["tracker"].get("max_lost_sec", .75)))
            source = create_video_source(self.stream, self.cfg, int(self.cfg.get("input_size", 640)))
            with self.source_lock: self.source = source
            decoder = PoseDecoder(self.cfg.get("postprocess", {}))
            print(f"[{self.stream['id']}] video={source.active_backend} postprocess={decoder.active_backend}", flush=True)
            while RUNNING:
                _t_read = time.monotonic()
                frame = source.read()
                # read() is the one stage no published field covered, which left
                # the decode backends impossible to compare in situ.
                read_ms = (time.monotonic() - _t_read) * 1000.0
                if frame is None:
                    time.sleep(float(self.stream.get("reconnect_delay_ms", 1000)) / 1000); continue
                self.process_frame(model, frame, time.time(), read_ms, decoder, source.active_backend)
        except BaseException as exc:
            self.exception = exc
        finally:
            for resource in (source, model):
                if resource is None: continue
                try: resource.close()
                except BaseException as exc:
                    if self.exception is None: self.exception = exc
            with self.source_lock: self.source = None
            if self.exception is not None: RUNNING = False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="/config/config.json"); ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text())
    required = ("model_path", "mqtt", "streams")
    if any(k not in cfg for k in required): raise SystemExit(f"missing required config keys: {required}")
    validate_backend_config(cfg)
    enabled = [x for x in cfg["streams"] if x.get("enabled", True)]
    if not enabled:
        raise SystemExit("at least one enabled stream is required")
    enabled_ids = [x.get("id") for x in enabled]
    if any(not stream_id for stream_id in enabled_ids) or len(enabled_ids) != len(set(enabled_ids)):
        raise SystemExit("enabled stream IDs must be present and unique")
    mode = str(cfg.get("inference_mode", "auto"))
    if mode not in ("auto", "legacy", "context_pool"):
        raise SystemExit("inference_mode must be auto, legacy, or context_pool")
    policy = cfg.get("core_policy", "measured_auto")
    if policy not in ("measured_auto", "runtime_auto"):
        raise SystemExit("core_policy must be measured_auto or runtime_auto")
    try:
        context_count = int(cfg.get("context_pool_size", min(len(enabled), 4)))
    except (TypeError, ValueError) as exc:
        raise SystemExit("context_pool_size must be a positive integer") from exc
    if context_count <= 0:
        raise SystemExit("context_pool_size must be a positive integer")
    stream_masks = any("core_mask" in stream for stream in enabled)
    if mode in ("legacy", "auto"):
        try:
            for stream in enabled:
                resolve_legacy_core_mask(cfg, stream)
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    if mode == "auto" and stream_masks and "context_core_masks" in cfg:
        raise SystemExit("auto cannot combine per-stream core_mask with context_core_masks")
    if mode == "legacy" and "context_core_masks" in cfg:
        raise SystemExit("legacy does not accept context_core_masks")
    if mode == "context_pool" and stream_masks:
        raise SystemExit("context_pool does not accept per-stream core_mask; use context_core_masks")
    if mode == "context_pool" and "context_core_masks" not in cfg:
        raise SystemExit("context_pool requires context_core_masks")
    if mode == "context_pool":
        paths = {str(stream.get("model_path", cfg["model_path"])) for stream in enabled}
        if len(paths) != 1:
            raise SystemExit("context_pool requires one model_path for all streams")
    if "context_core_masks" in cfg:
        try:
            resolve_context_core_masks(cfg, context_count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    if args.validate:
        print(json.dumps({"valid": True, "streams": len(cfg["streams"]),
                          "video": cfg.get("video", {}),
                          "postprocess": cfg.get("postprocess", {})})); return
    global RUNNING
    def stop(*_):
        global RUNNING; RUNNING = False
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    publishers = make_stream_publishers(cfg["mqtt"], enabled)
    # Auto keeps the historical one-stream execution path byte-for-byte in
    # scheduling terms, and only enables the pool when multiple streams exist.
    use_pool = mode == "context_pool" or (mode == "auto" and
                                            ((len(enabled) > 1 and not stream_masks) or
                                             "context_core_masks" in cfg))
    if not use_pool:
        workers = [StreamWorker(cfg, x, publishers[x["id"]], len(enabled)) for x in enabled]
        try:
            for worker in workers: worker.start()
            while RUNNING and any(x.is_alive() for x in workers): time.sleep(.5)
            for worker in workers: worker.stop()
            for worker in workers: worker.join(timeout=5)
            alive = [worker.name for worker in workers if worker.is_alive()]
            if alive: raise TimeoutError(f"stream workers did not stop: {alive}")
            errors = [worker.exception for worker in workers if worker.exception is not None]
            if errors: raise errors[0]
        finally:
            if not any(worker.is_alive() for worker in workers):
                close_publishers(publishers)
        return

    workers = {x["id"]: StreamWorker(cfg, x, publishers[x["id"]], len(enabled)) for x in enabled}
    decoders = {x["id"]: PoseDecoder(cfg.get("postprocess", {})) for x in enabled}
    def source_factory(stream):
        source = create_video_source(stream, cfg, int(cfg.get("input_size", 640)))
        print(f"[{stream['id']}] video={source.active_backend} postprocess={decoders[stream['id']].active_backend}", flush=True)
        return source
    def model_factory(_index):
        # Contexts are fixed batch-1 runtimes. A stream-specific model override
        # is supported only when all streams use the same model, as RKNN cannot
        # share a runtime across differing loaded graphs.
        paths = {str(x.get("model_path", cfg["model_path"])) for x in enabled}
        if len(paths) != 1:
            raise ValueError("context_pool requires one model_path for all streams")
        return RKNNPose(next(iter(paths)), context_masks[_index], cfg.get("rknn", {}))
    def frame_handler(stream, model, item):
        worker = workers[stream["id"]]
        worker.process_frame(model, item.frame, item.timestamp, item.read_ms,
                             decoders[stream["id"]], item.source_backend)
    context_masks = resolve_context_core_masks(cfg, context_count)
    pool = ContextPoolRuntime(enabled, context_count,
                              source_factory, model_factory, frame_handler)
    try:
        pool.start()
        while RUNNING and not pool._stop.is_set() and any(t.is_alive() for t in pool._threads):
            time.sleep(.5)
    finally:
        pool.stop()
        close_publishers(publishers)
    if pool.exception is not None:
        raise pool.exception


if __name__ == "__main__": main()
