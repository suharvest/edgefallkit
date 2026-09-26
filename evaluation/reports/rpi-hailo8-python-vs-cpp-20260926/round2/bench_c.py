#!/usr/bin/env python3
"""Variant B: Python Hailo yolov8s-pose multi-stream pipeline (benchmark only).

Skeleton follows edge-security-kit esk_hailo/esk_core: one capture thread per
stream (OpenCV FFmpeg software decode), one shared InferModel, numpy head decode
in the quantized domain, paho publish. Batching mirrors fall-hailo C++
(FrameBatcher: per-stream queue depth 2, drop-oldest, batch 1/4/8 auto, 20 ms
wait, partial batches padded; run_async). Pose decode follows
fall-detection/platforms/rpi-hailo/src/hailo_pose_decoder.cpp semantics.
"""
import argparse, collections, functools, json, math, mmap, os, subprocess, sys, threading, time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

cv2.setNumThreads(1)


def batch_for(n):
    return 1 if n <= 3 else (4 if n == 4 else 8)


# ---------------------------------------------------------------- decode
def iou_one(box, boxes):
    # boxes are (cx, cy, w, h) normalized
    ax1, ay1, ax2, ay2 = box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2
    bx1, by1 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
    bx2, by2 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


ARANGE16 = np.arange(16, dtype=np.float32)


def decode_pose(outs, groups, thr, nms_thr):
    """outs: dict name->array (H,W,C) quantized. groups: list of (side, stride, (bname,bq),(sname,sq),(kname,kq))."""
    all_boxes, all_scores, all_kpts = [], [], []
    for side, stride, (bn, (bs, bz)), (sn, (ss, sz)), (kn, (ks, kz)) in groups:
        s = outs[sn].reshape(-1)
        level = math.ceil(sz + thr / ss - 1e-6)
        if level > np.iinfo(s.dtype).max:
            continue
        idx = np.flatnonzero(s >= level)
        if idx.size == 0:
            continue
        score = (s[idx].astype(np.float32) - sz) * ss
        b = (outs[bn].reshape(-1, 64)[idx].astype(np.float32) - bz) * bs
        b = b.reshape(-1, 4, 16)
        b = np.exp(b - b.max(axis=2, keepdims=True))
        dist = (b * ARANGE16).sum(axis=2) / b.sum(axis=2) * stride
        ys, xs = np.divmod(idx, side)
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        l, t, r, bo = cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]
        boxes = np.stack([(l + r) / 1280.0, (t + bo) / 1280.0, (r - l) / 640.0, (bo - t) / 640.0], axis=1)
        k = ((outs[kn].reshape(-1, 51)[idx].astype(np.float32) - kz) * ks).reshape(-1, 17, 3)
        kp = np.empty_like(k)
        kp[..., 0] = (stride * (2 * k[..., 0] - 0.5) + cx[:, None]) / 640.0
        kp[..., 1] = (stride * (2 * k[..., 1] - 0.5) + cy[:, None]) / 640.0
        kp[..., 2] = 1.0 / (1.0 + np.exp(-k[..., 2]))
        all_boxes.append(boxes); all_scores.append(score); all_kpts.append(kp)
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32)
    boxes = np.concatenate(all_boxes); scores = np.concatenate(all_scores); kpts = np.concatenate(all_kpts)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        ious = iou_one(boxes[i], boxes[order[1:]])
        order = order[1:][ious < nms_thr]
    keep = np.asarray(keep)
    return boxes[keep], scores[keep], kpts[keep]


class Tracker:
    """Greedy IoU tracker + aspect-ratio fall flag (stand-in for the C++ tracker/temporal)."""

    def __init__(self):
        self.tracks = {}  # id -> [box, lost, hist]
        self.next_id = 1

    def update(self, boxes):
        ids = [-1] * len(boxes)
        free = set(self.tracks)
        for i, b in enumerate(boxes):
            best, bi = 0.3, None
            for tid in free:
                v = iou_one(b, self.tracks[tid][0][None, :])[0]
                if v > best:
                    best, bi = v, tid
            if bi is None:
                bi = self.next_id; self.next_id += 1
                self.tracks[bi] = [b, 0, collections.deque(maxlen=15)]
            else:
                free.discard(bi)
                self.tracks[bi][0] = b; self.tracks[bi][1] = 0
            self.tracks[bi][2].append(float(b[2] / max(b[3], 1e-6)))
            ids[i] = bi
        for tid in list(free):
            self.tracks[tid][1] += 1
            if self.tracks[tid][1] > 15:
                del self.tracks[tid]
        return ids

    def fallen(self, tid):
        h = self.tracks[tid][2]
        return len(h) >= 5 and sum(h) / len(h) > 1.2


# ---------------------------------------------------------------- stream
class Stream:
    def __init__(self, index, sid, url):
        self.index, self.id, self.url = index, sid, url
        self.q = collections.deque()
        self.seq = 0
        self.decoded = 0; self.dropped = 0
        self.frames = 0; self.pipe_ms = []; self.full_ms = []
        self.tracker = Tracker()
        self.last_err = ""


class Pipeline:
    def __init__(self, a, streams):
        self.a = a
        self.streams = streams
        self.cv = threading.Condition()
        self.stop = False
        self.measuring = False
        self.infer_ms = []
        self.async_ms = []
        self.batch_hist = collections.Counter()
        self.post_q = collections.deque()
        self.post_cv = threading.Condition()
        self.payload_errors = 0
        self.published = 0
        self.rr = 0
        self._setup_hailo()
        self.client = mqtt.Client(client_id=f"bench-b-{os.getpid()}")
        self.client.connect(a.mqtt_host, a.mqtt_port, 30)
        self.client.loop_start()

    def _setup_hailo(self):
        from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice
        a = self.a
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE if a.sched == 'none' else HailoSchedulingAlgorithm.ROUND_ROBIN
        if a.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"
        self.vdevice = VDevice(params)
        self.im = self.vdevice.create_infer_model(a.hef)
        self.B = a.batch or batch_for(len(self.streams))
        self.im.set_batch_size(self.B)
        native = {i.name: i.format.type for i in HEF(a.hef).get_output_vstream_infos()}
        self.out_names = [o.name for o in self.im.outputs]
        dt = {}
        for n in self.out_names:
            ft = native[n]
            if ft not in (FormatType.UINT8, FormatType.UINT16):
                ft = FormatType.UINT8
            self.im.output(n).set_format_type(ft)
            dt[n] = np.uint8 if ft == FormatType.UINT8 else np.uint16
        self.shapes = {n: tuple(self.im.output(n).shape) for n in self.out_names}
        quants = {n: (float(self.im.output(n).quant_infos[0].qp_scale), float(self.im.output(n).quant_infos[0].qp_zp)) for n in self.out_names}
        self.groups = []
        for side in (80, 40, 20):
            g = {}
            for n, sh in self.shapes.items():
                if sh[0] == side and sh[1] == side:
                    g[sh[2]] = (n, quants[n])
            if 64 in g and 1 in g and 51 in g:
                self.groups.append((side, 640 // side, g[64], g[1], g[51]))
        if len(self.groups) != 3:
            raise RuntimeError(f"unexpected output shapes {self.shapes}")
        self.dtypes = dt
        self.configured = self.im.configure()
        if a.sched == 'none':
            self.configured.activate()
        self._maps = []

        def alloc(shape, dtype):
            if not a.aligned:
                return np.empty(shape, dtype=dtype)
            n = int(np.prod(shape)) * np.dtype(dtype).itemsize
            m = mmap.mmap(-1, (n + 4095) // 4096 * 4096)
            self._maps.append(m)
            return np.frombuffer(m, dtype=np.uint8, count=n).view(dtype).reshape(shape)
        # binding pool: a.inflight batches in flight
        self.pool = collections.deque()
        for _ in range(a.inflight):
            batch = []
            for _ in range(self.B):
                bufs = {n: alloc(self.shapes[n], dt[n]) for n in self.out_names}
                bnd = self.configured.create_bindings(output_buffers=bufs)
                if a.aligned:
                    inp = alloc((640, 640, 3), np.uint8)
                    bnd.input().set_buffer(inp)
                    bufs = dict(bufs); bufs['__in__'] = inp
                batch.append((bnd, bufs))
            self.pool.append(batch)
        self.pool_cv = threading.Condition()
        self.info = {"batch": self.B, "outputs": {n: [list(self.shapes[n]), str(dt[n].__name__), quants[n]] for n in self.out_names}}

    # capture thread: esk_core StreamWorker shape (open/read/reopen)
    def capture(self, s):
        while not self.stop:
            cap = cv2.VideoCapture(s.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                s.last_err = "open failed"; time.sleep(1); continue
            while not self.stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    s.last_err = "read failed"; break
                t = time.monotonic()
                if frame.shape[0] != 640 or frame.shape[1] != 640:
                    frame = cv2.resize(frame, (640, 640))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # contiguous, no negative-stride copy
                with self.cv:
                    if len(s.q) >= 2:
                        s.q.popleft()
                        if self.measuring:
                            s.dropped += 1
                    s.q.append((rgb, t, s.seq))
                    s.seq += 1
                    if self.measuring:
                        s.decoded += 1
                    self.cv.notify()
            cap.release()

    def queued(self):
        return sum(len(s.q) for s in self.streams)

    def batcher(self):
        n = len(self.streams)
        while not self.stop:
            with self.cv:
                while not self.stop and self.queued() == 0:
                    self.cv.wait(0.2)
                deadline = time.monotonic() + self.a.wait_ms / 1000.0
                while not self.stop and self.queued() < self.B:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        break
                    self.cv.wait(rem)
                frames = []
                start = self.rr
                for k in range(n):
                    if len(frames) >= self.B:
                        break
                    st = self.streams[(start + k) % n]
                    if st.q:
                        frames.append((st, *st.q.popleft()))
                self.rr = (start + 1) % n
            if not frames:
                continue
            with self.pool_cv:
                while not self.pool:
                    self.pool_cv.wait()
                batch = self.pool.popleft()
            bl = []
            for i in range(self.B):
                if self.a.aligned:
                    if i < len(frames):  # pad slots keep stale data (C++ memsets zeros)
                        np.copyto(batch[i][1]['__in__'], frames[i][1])
                else:
                    f = frames[min(i, len(frames) - 1)]
                    batch[i][0].input().set_buffer(f[1])
                bl.append(batch[i][0])
            self.configured.wait_for_async_ready(30000, self.B)
            t0 = time.monotonic()
            self.configured.run_async(bl, functools.partial(self.done, batch, frames, t0))
            if self.measuring:
                self.async_ms.append((time.monotonic() - t0) * 1000)
            if self.measuring:
                self.batch_hist[len(frames)] += 1

    def done(self, batch, frames, t0, completion_info):
        t1 = time.monotonic()
        if completion_info.exception:
            print("async error", completion_info.exception, file=sys.stderr)
        with self.post_cv:
            self.post_q.append((batch, frames, t0, t1))
            self.post_cv.notify()

    def post(self):
        a = self.a
        while not self.stop:
            with self.post_cv:
                while not self.stop and not self.post_q:
                    self.post_cv.wait(0.2)
                if self.stop:
                    return
                batch, frames, t0, t1 = self.post_q.popleft()
            if self.measuring:
                self.infer_ms.append((t1 - t0) * 1000)
            results = []
            for i, (st, rgb, tdec, seq) in enumerate(frames):
                boxes, scores, kpts = decode_pose(batch[i][1], self.groups, a.score, 0.7)
                results.append((st, tdec, seq, boxes, scores, kpts))
            with self.pool_cv:
                self.pool.append(batch)
                self.pool_cv.notify()
            for st, tdec, seq, boxes, scores, kpts in results:
                keep = [j for j in range(len(boxes))]
                ids = st.tracker.update(boxes)
                t_tr = time.monotonic()
                persons = []
                fall = False
                for j in keep:
                    fl = st.tracker.fallen(ids[j]); fall = fall or fl
                    k = kpts[j]
                    persons.append({"track_id": int(ids[j]), "score": round(float(scores[j]), 4),
                                    "bbox": [round(float(v), 4) for v in boxes[j]],
                                    "keypoints": [[round(float(x), 4), round(float(y), 4), round(float(c), 3)] for x, y, c in k],
                                    "fallen": fl})
                pipe = (t_tr - tdec) * 1000
                payload = json.dumps({"schema": "bench.fall/1", "stream_id": st.id, "frame_seq": seq,
                                      "timestamp_ms": int(time.time() * 1000), "person_count": len(persons),
                                      "fall_detected": fall, "persons": persons,
                                      "pipeline_ms": round(pipe, 2), "backend": "python-hailort-4.21"},
                                     separators=(",", ":"))
                info = self.client.publish(f"bench/fall/{st.id}", payload, qos=0)
                t_pub = time.monotonic()
                if info.rc != 0:
                    self.payload_errors += 1
                if self.measuring:
                    st.frames += 1
                    st.pipe_ms.append(pipe)
                    st.full_ms.append((t_pub - tdec) * 1000)
                    self.published += 1


def pct(v, p):
    if not v:
        return None
    return round(float(np.percentile(np.asarray(v), p)), 2)


def run_single(a, stream_list):
    streams = [Stream(i, sid, url) for i, (sid, url) in enumerate(stream_list)]
    p = Pipeline(a, streams)
    threads = [threading.Thread(target=p.capture, args=(s,), daemon=True) for s in streams]
    threads.append(threading.Thread(target=p.batcher, daemon=True))
    threads += [threading.Thread(target=p.post, daemon=True) for _ in range(a.post_threads)]
    for t in threads:
        t.start()
    now = time.time()
    t_start = a.measure_start if a.measure_start else now + a.warmup
    time.sleep(max(0, t_start - time.time()))
    with p.cv:
        p.measuring = True
    m0 = time.monotonic()
    time.sleep(a.seconds)
    with p.cv:
        p.measuring = False
    sec = time.monotonic() - m0
    res = {"pid": os.getpid(), "hailo": p.info, "seconds": sec, "batch_hist": dict(p.batch_hist),
           "infer_ms_p50": pct(p.infer_ms, 50), "run_async_ms_p50": pct(p.async_ms, 50), "run_async_ms_p95": pct(p.async_ms, 95), "infer_ms_p95": pct(p.infer_ms, 95), "infer_batches": len(p.infer_ms),
           "published": p.published, "publish_errors": p.payload_errors, "streams": []}
    for s in streams:
        res["streams"].append({"id": s.id, "frames": s.frames, "fps": round(s.frames / sec, 4),
                               "decoded": s.decoded, "dropped_queue": s.dropped,
                               "pipeline_ms_p50": pct(s.pipe_ms, 50), "pipeline_ms_p95": pct(s.pipe_ms, 95),
                               "full_ms_p50": pct(s.full_ms, 50), "full_ms_p95": pct(s.full_ms, 95),
                               "last_err": s.last_err})
    # HailoRT python teardown segfaults with async jobs in flight (see esk README
    # infer-wrapper teardown); results are written and the process hard-exits.
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--streams", required=True, help="id|url;id|url")
    ap.add_argument("--warmup", type=float, default=30)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--measure-start", type=float, default=0)
    ap.add_argument("--procs", type=int, default=1, help="shard streams over N processes (needs hailort service)")
    ap.add_argument("--multi-process-service", action="store_true")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--wait-ms", type=int, default=20)
    ap.add_argument("--sched", default="rr")
    ap.add_argument("--inflight", type=int, default=3)
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--post-threads", type=int, default=1)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=18830)
    ap.add_argument("--window-file", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stream_list = [tuple(x.split("|", 1)) for x in a.streams.split(";") if x]
    if a.procs <= 1:
        if not a.measure_start:
            a.measure_start = time.time() + a.warmup
        if a.window_file:
            json.dump({"start": a.measure_start, "end": a.measure_start + a.seconds}, open(a.window_file, "w"))
        res = run_single(a, stream_list)
        res["mode"] = "single-process" if not a.multi_process_service else "shard"
        json.dump(res, open(a.out, "w"), indent=1)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    # sharded: children with <= ceil(N/procs) streams each, shared device via hailort service
    shards = [stream_list[i::a.procs] for i in range(a.procs)]
    shards = [s for s in shards if s]
    ms = time.time() + a.warmup
    if a.window_file:
        json.dump({"start": ms, "end": ms + a.seconds}, open(a.window_file, "w"))
    kids = []
    for i, sh in enumerate(shards):
        out = f"{a.out}.shard{i}"
        cmd = [sys.executable, os.path.abspath(__file__), "--hef", a.hef,
               "--streams", ";".join(f"{x}|{y}" for x, y in sh), "--seconds", str(a.seconds),
               "--measure-start", str(ms), "--multi-process-service", "--wait-ms", str(a.wait_ms),
               "--mqtt-host", a.mqtt_host, "--mqtt-port", str(a.mqtt_port), "--out", out]
        kids.append((subprocess.Popen(cmd), out))
    agg = {"mode": f"multi-process x{len(shards)}", "shards": [], "streams": [], "published": 0}
    for pr, out in kids:
        rc = pr.wait()
        try:
            r = json.load(open(out))
        except Exception as e:  # noqa: BLE001
            r = {"error": f"rc={rc} {e}", "streams": [], "published": 0}
        r["rc"] = rc
        agg["shards"].append({k: v for k, v in r.items() if k != "streams"})
        agg["streams"] += r["streams"]
        agg["published"] += r.get("published", 0)
    json.dump(agg, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
