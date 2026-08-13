#!/usr/bin/env python3
"""Convert the fixed 640x640 raw-head YOLO11n-Pose ONNX to native RKNN2."""
import argparse
from rknn.api import RKNN


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--onnx",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--platform",choices=("rk3576","rk3588"),required=True); ap.add_argument("--dataset")
    args=ap.parse_args(); r=RKNN(verbose=True)
    r.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform=args.platform,
             quantized_dtype="w8a8",quantized_algorithm="normal",optimization_level=3)
    if r.load_onnx(model=args.onnx): raise SystemExit("load_onnx failed")
    if r.build(do_quantization=bool(args.dataset),dataset=args.dataset): raise SystemExit("build failed")
    if r.export_rknn(args.out): raise SystemExit("export failed")
    r.release()


if __name__ == "__main__": main()
