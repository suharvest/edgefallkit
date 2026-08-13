#!/usr/bin/env python3
"""RKNN runtime benchmark: native inference plus selected postprocess backend."""
import argparse, json, resource, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rknn_pose import PoseDecoder, RKNNPose


def pct(values, p): return sorted(values)[min(len(values)-1, int(len(values) * p))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--contexts", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=200); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--input-npy"); ap.add_argument("--image"); ap.add_argument("--json-out")
    ap.add_argument("--postprocess-backend", choices=("cpp", "numpy", "auto"), default="cpp")
    args = ap.parse_args()
    if args.input_npy:
        frame=np.load(args.input_npy)
    elif args.image:
        import cv2
        bgr=cv2.imread(args.image); h,w=bgr.shape[:2]; scale=min(640/w,640/h)
        nw,nh=int(round(w*scale)),int(round(h*scale)); resized=cv2.resize(bgr,(nw,nh))
        frame=np.full((640,640,3),114,np.uint8); left=(640-nw)//2; top=(640-nh)//2
        frame[top:top+nh,left:left+nw]=resized; frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    else:
        frame=np.full((640,640,3),114,np.uint8)
    runners = [RKNNPose(args.model) for _ in range(args.contexts)]; locks = [threading.Lock() for _ in runners]
    decoders = [PoseDecoder({"backend": args.postprocess_backend, "strict": True, "fallback": "none"})
                for _ in range(args.contexts)]
    for i in range(args.warmup): runners[i % len(runners)].infer(frame)
    infer_ms=[]; total_ms=[]; detection_counts=[]
    def one(i):
        start=time.perf_counter()
        with locks[i % len(runners)]: outputs, ms = runners[i % len(runners)].infer(frame)
        detections=decoders[i % len(decoders)].decode(outputs)
        return ms, (time.perf_counter()-start)*1000, len(detections)
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.contexts) as pool:
        for im, tm, dc in pool.map(one, range(args.iterations)):
            infer_ms.append(im); total_ms.append(tm); detection_counts.append(dc)
    wall=time.perf_counter()-started
    result={"contexts":args.contexts,"iterations":args.iterations,"throughput_fps":args.iterations/wall,
            "postprocess_backend": decoders[0].active_backend,
            "inference_ms":{"mean":statistics.mean(infer_ms),"p50":pct(infer_ms,.5),"p95":pct(infer_ms,.95),"max":max(infer_ms)},
            "pipeline_ms":{"mean":statistics.mean(total_ms),"p50":pct(total_ms,.5),"p95":pct(total_ms,.95),"max":max(total_ms)},
            "detections_per_frame":{"mean":statistics.mean(detection_counts),"min":min(detection_counts),"max":max(detection_counts)},
            "max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for r in runners:r.close()
    print(json.dumps(result,indent=2));
    if args.json_out: open(args.json_out,"w").write(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__": main()
