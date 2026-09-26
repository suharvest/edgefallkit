#!/usr/bin/env python3
"""Sample CPU% (100 = one core) and RSS of a process tree inside the measure window.
Also MQTT subscriber mode: count/validate payloads inside the window."""
import argparse, json, os, sys, time

import psutil


def wait_window(path, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        try:
            w = json.load(open(path))
            return w["start"], w["end"]
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise SystemExit("no window file")


def tree(pid):
    try:
        p = psutil.Process(pid)
        return [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def cpu_total(procs):
    tot = 0.0
    for p in procs:
        try:
            c = p.cpu_times()
            tot += c.user + c.system + getattr(c, "children_user", 0) * 0
        except psutil.NoSuchProcess:
            pass
    return tot


def monitor(a):
    s, e = wait_window(a.window_file)
    pid = a.pid
    if a.pid_file:
        while not os.path.exists(a.pid_file):
            time.sleep(0.2)
        pid = int(open(a.pid_file).read().strip())
    time.sleep(max(0, s - time.time()))
    rss_max = 0; samples = []
    last = {}
    def acc():
        tot = 0.0; rss = 0; n = 0
        for p in tree(pid):
            try:
                c = p.cpu_times(); v = c.user + c.system
                rss += p.memory_info().rss; n += 1
            except psutil.NoSuchProcess:
                continue
            prev = last.get(p.pid)
            if prev is not None:
                tot += max(0.0, v - prev)
            last[p.pid] = v
        return tot, rss, n
    acc(); t0 = time.time(); used = 0.0; procs = []
    psutil.cpu_percent(None)
    sys_samples = []
    while time.time() < e:
        time.sleep(1)
        d, rss, n = acc(); used += d; procs = [None] * max(len(procs), n)
        rss_max = max(rss_max, rss)
        samples.append(rss)
        sys_samples.append(psutil.cpu_percent(None))
    t1 = time.time(); c0 = 0.0; c1 = used
    out = {"pid": pid, "nprocs": len(procs), "cpu_percent": round((c1 - c0) / (t1 - t0) * 100, 1),
           "rss_mib_max": round(rss_max / 2**20, 1),
           "rss_mib_mean": round(sum(samples) / max(1, len(samples)) / 2**20, 1),
           "system_cpu_percent_mean_of_400": round(sum(sys_samples) / max(1, len(sys_samples)) * 4, 1),
           "seconds": round(t1 - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)


def subscribe(a):
    import paho.mqtt.client as mqtt
    s, e = wait_window(a.window_file)
    st = {"msgs": 0, "invalid": 0, "per_topic": {}, "sample": None, "missing_keys": 0}
    need = [k for k in a.keys.split(",") if k]

    def on_msg(c, u, m):
        now = time.time()
        if not (s <= now <= e):
            return
        st["msgs"] += 1
        st["per_topic"][m.topic] = st["per_topic"].get(m.topic, 0) + 1
        try:
            d = json.loads(m.payload)
            if any(k not in d for k in need):
                st["missing_keys"] += 1
            if st["sample"] is None:
                st["sample"] = m.payload.decode()[:3000]
        except Exception:  # noqa: BLE001
            st["invalid"] += 1
    c = mqtt.Client(client_id=f"bench-sub-{os.getpid()}")
    c.on_message = on_msg
    c.connect(a.mqtt_host, a.mqtt_port, 30)
    c.subscribe("#", 0)
    c.loop_start()
    while time.time() < e + 2:
        time.sleep(0.5)
    c.loop_stop()
    st["topics"] = len(st["per_topic"])
    st["per_topic_rate_min"] = round(min(st["per_topic"].values()) / (e - s), 3) if st["per_topic"] else 0
    json.dump(st, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cpu", "sub"])
    ap.add_argument("--window-file", required=True)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--pid-file", default="")
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=18830)
    ap.add_argument("--keys", default="stream_id")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (monitor if a.mode == "cpu" else subscribe)(a)
