#!/usr/bin/env python3
import glob, json, os, re, sys, statistics as st

R = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{R}/*.window")):
    tag = os.path.basename(f)[:-7]
    m = re.match(r"(A|Bs|Bm)-n(\d+)-r(\d+)", tag)
    if not m:
        continue
    v, n, rep = m.group(1), int(m.group(2)), int(m.group(3))
    row = {"tag": tag, "v": v, "n": n, "rep": rep}
    def load(sfx):
        try:
            return json.load(open(f"{R}/{tag}.{sfx}"))
        except Exception:
            return {}
    cpu, sub = load("cpu.json"), load("sub.json")
    if v == "A":
        log = open(f"{R}/{tag}.log").read() if os.path.exists(f"{R}/{tag}.log") else ""
        b = re.findall(r"BENCHMARK stream=(\S+) frames=(\d+) seconds=(\S+) fps=(\S+) mean_pipeline_ms=(\S+)", log)
        fps = [float(x[3]) for x in b]; pm = [float(x[4]) for x in b]
        hb = re.search(r"HAILO_BATCH .*backend=(\S+) batch=(\d+)", log)
        row.update(fps_min=min(fps) if fps else None, fps_max=max(fps) if fps else None,
                   pipe=f"mean {min(pm):.1f}-{max(pm):.1f}" if pm else "", infer="n/a",
                   batch=f"{hb.group(1)}/{hb.group(2)}" if hb else "")
        frames = [int(x[1]) for x in b]
    else:
        d = load("json")
        ss = d.get("streams", [])
        fps = [s["fps"] for s in ss]
        p50 = [s["pipeline_ms_p50"] for s in ss if s["pipeline_ms_p50"] is not None]
        p95 = [s["pipeline_ms_p95"] for s in ss if s["pipeline_ms_p95"] is not None]
        if v == "Bs":
            inf = f"{d.get('infer_ms_p50')}/{d.get('infer_ms_p95')}"; batch = str(d.get("hailo", {}).get("batch"))
        else:
            sh = d.get("shards", [])
            inf = ";".join(f"{s.get('infer_ms_p50')}/{s.get('infer_ms_p95')}" for s in sh)
            batch = f"{len(sh)}proc x b{sh[0].get('hailo', {}).get('batch') if sh else '?'}"
        row.update(fps_min=min(fps) if fps else None, fps_max=max(fps) if fps else None,
                   pipe=f"p50 {st.median(p50):.1f} / p95 max {max(p95):.1f}" if p50 else "", infer=inf, batch=batch,
                   qdrop=sum(s["dropped_queue"] for s in ss))
        frames = [s["frames"] for s in ss]
    row["pass"] = bool(row.get("fps_min") and row["fps_min"] >= 14.5 and len(fps) == n)
    row["cpu"] = cpu.get("cpu_percent"); row["rss"] = cpu.get("rss_mib_max"); row["sys_cpu"] = cpu.get("system_cpu_percent_mean_of_400")
    row["mqtt"] = f"{sub.get('msgs')} msgs, invalid {sub.get('invalid')}, topics {sub.get('topics')}, min {sub.get('per_topic_rate_min')}/s"
    row["missing_frames"] = sum(max(0, 1800 - x) for x in frames) if frames else None
    try:
        row["env"] = open(f"{R}/{tag}.env").read().replace("\n", " ").strip()
    except Exception:
        row["env"] = ""
    rows.append(row)
rows.sort(key=lambda r: (r["n"], r["v"], r["rep"]))
print("| tag | fps_min | fps_max | pass | batch | infer p50/p95 ms | pipeline ms | CPU% proc | sysCPU%/400 | RSS MiB | missing frames | MQTT | temp |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    fm = f"{r['fps_min']:.2f}" if r.get("fps_min") is not None else "-"
    fx = f"{r['fps_max']:.2f}" if r.get("fps_max") is not None else "-"
    print(f"| {r['tag']} | {fm} | {fx} | {'Y' if r['pass'] else 'N'} | {r.get('batch')} | {r.get('infer')} | {r.get('pipe')} | {r['cpu']} | {r['sys_cpu']} | {r['rss']} | {r['missing_frames']} | {r['mqtt']} | {r['env']} |")
json.dump(rows, open(f"{R}/summary.json", "w"), indent=1)
