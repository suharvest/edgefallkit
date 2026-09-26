#!/usr/bin/env python3
"""Synthetic Hailo transport benchmark (no video decode).

Feeds pre-decoded 640x640 RGB frames into one InferModel as fast as the device
accepts them and varies only the host-side data movement:
  --sched rr|none        scheduler ROUND_ROBIN (bench_b.py) vs NONE+activate (C++ BatchedHailoRunner)
  --batch B              InferModel batch size
  --inflight K           number of binding slots (= max batches in flight)
  --buffers unaligned|aligned   np.empty (malloc, 16 B aligned) vs mmap page-aligned
  --input fresh|pool     fresh: new ndarray per frame + set_buffer every batch (bench_b.py)
                         pool:  memcpy into the slot's own input buffer, set_buffer only once (C++)
  --decode post|callback|none   pose decode in a post thread / inside the HailoRT callback / skipped
"""
import argparse, collections, json, mmap, os, sys, threading, time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_b import decode_pose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hef", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--sched", default="rr")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--inflight", type=int, default=3)
ap.add_argument("--buffers", default="unaligned")
ap.add_argument("--input", default="fresh")
ap.add_argument("--decode", default="post")
ap.add_argument("--post-threads", type=int, default=1)
ap.add_argument("--warmup", type=float, default=5)
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice  # noqa: E402

raw = np.fromfile(a.frames, dtype=np.uint8)
FR = raw.reshape(-1, 640, 640, 3)
NF = FR.shape[0]
_maps = []


def alloc(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if a.buffers == "aligned":
        m = mmap.mmap(-1, (n + 4095) // 4096 * 4096)
        _maps.append(m)
        arr = np.frombuffer(m, dtype=np.uint8, count=n).view(dtype).reshape(shape)
        assert arr.ctypes.data % 4096 == 0
        return arr
    return np.empty(shape, dtype=dtype)


params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN if a.sched == "rr" else HailoSchedulingAlgorithm.NONE
vd = VDevice(params)
im = vd.create_infer_model(a.hef)
B = a.batch
im.set_batch_size(B)
native = {i.name: i.format.type for i in HEF(a.hef).get_output_vstream_infos()}
out_names = [o.name for o in im.outputs]
dt, shapes, quants = {}, {}, {}
for n in out_names:
    ft = native[n] if native[n] in (FormatType.UINT8, FormatType.UINT16) else FormatType.UINT8
    im.output(n).set_format_type(ft)
    dt[n] = np.uint8 if ft == FormatType.UINT8 else np.uint16
    shapes[n] = tuple(im.output(n).shape)
    q = im.output(n).quant_infos[0]
    quants[n] = (float(q.qp_scale), float(q.qp_zp))
groups = []
for side in (80, 40, 20):
    g = {sh[2]: (n, quants[n]) for n, sh in shapes.items() if sh[0] == side and sh[1] == side}
    groups.append((side, 640 // side, g[64], g[1], g[51]))
cfg = im.configure()
if a.sched == "none":
    cfg.activate()

free = collections.deque()
for _ in range(a.inflight):
    slot = []
    for _ in range(B):
        outs = {n: alloc(shapes[n], dt[n]) for n in out_names}
        inp = alloc((640, 640, 3), np.uint8)
        bnd = cfg.create_bindings(output_buffers=outs)
        bnd.input().set_buffer(inp)
        slot.append((bnd, outs, inp))
    free.append(slot)
free_cv = threading.Condition()
post_q = collections.deque()
post_cv = threading.Condition()
stop = False
measuring = False
st = collections.Counter()
t_async, t_ready, t_fill, t_cb = [], [], [], []
persons = [0]


def release(slot):
    with free_cv:
        free.append(slot)
        free_cv.notify()


def do_decode(slot):
    for bnd, outs, _ in slot:
        b, s, k = decode_pose(outs, groups, 0.35, 0.7)
        persons[0] += len(b)


def done(slot, t0, completion_info):
    t1 = time.perf_counter()
    if completion_info.exception:
        st["err"] += 1
    if measuring:
        st["frames"] += B
        st["batches"] += 1
    if a.decode == "callback":
        do_decode(slot)
        if measuring:
            t_cb.append(time.perf_counter() - t1)
        release(slot)
    elif a.decode == "post":
        with post_cv:
            post_q.append(slot)
            post_cv.notify()
    else:
        release(slot)


def post():
    while not stop:
        with post_cv:
            while not stop and not post_q:
                post_cv.wait(0.2)
            if stop:
                return
            slot = post_q.popleft()
        do_decode(slot)
        release(slot)


def feeder():
    i = 0
    while not stop:
        with free_cv:
            while not free and not stop:
                free_cv.wait(0.2)
            if stop:
                return
            slot = free.popleft()
        tf = time.perf_counter()
        for bnd, _, inp in slot:
            src = FR[i % NF]
            i += 1
            if a.input == "fresh":
                x = src.copy()  # new ndarray per frame, as cvtColor returns in bench_b.py
                bnd.input().set_buffer(x)
            else:
                np.copyto(inp, src)  # memcpy into a persistent buffer, as C++ does
        t0 = time.perf_counter()
        cfg.wait_for_async_ready(30000, B)
        t1 = time.perf_counter()
        cfg.run_async([s[0] for s in slot], lambda completion_info, slot=slot, t1=t1: done(slot, t1, completion_info))
        t2 = time.perf_counter()
        if measuring:
            t_fill.append(t0 - tf); t_ready.append(t1 - t0); t_async.append(t2 - t1)


th = [threading.Thread(target=feeder, daemon=True)] + [threading.Thread(target=post, daemon=True) for _ in range(a.post_threads)]
for t in th:
    t.start()
time.sleep(a.warmup)
proc = psutil.Process()
c0 = proc.cpu_times(); w0 = time.monotonic(); measuring = True
time.sleep(a.seconds)
measuring = False
c1 = proc.cpu_times(); w1 = time.monotonic()
sec = w1 - w0


def pct(v, p, scale=1000):
    return round(float(np.percentile(np.asarray(v), p)) * scale, 3) if v else None


res = {"args": vars(a), "seconds": round(sec, 2), "frames": st["frames"], "fps": round(st["frames"] / sec, 1),
       "errors": st["err"], "persons_per_frame": round(persons[0] / max(1, st["frames"]), 2),
       "cpu_percent": round(((c1.user - c0.user) + (c1.system - c0.system)) / sec * 100, 1),
       "cpu_user_pct": round((c1.user - c0.user) / sec * 100, 1), "cpu_sys_pct": round((c1.system - c0.system) / sec * 100, 1),
       "run_async_ms_p50": pct(t_async, 50), "run_async_ms_p95": pct(t_async, 95),
       "run_async_us_per_frame_p50": pct(t_async, 50, 1e6 / B),
       "wait_ready_ms_p50": pct(t_ready, 50), "wait_ready_ms_p95": pct(t_ready, 95),
       "fill_ms_p50": pct(t_fill, 50), "callback_decode_ms_p50": pct(t_cb, 50)}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res))
sys.stdout.flush()
os._exit(0)
