#!/usr/bin/env python3
"""Export a compiled Jetson temporal-weight header to a dependency-light NPZ."""
import argparse, re
from pathlib import Path
import numpy as np


def scalar(text, name, cast=float):
    match = re.search(rf"\b{name}\s*=\s*([-+0-9.eE]+)f?\s*;", text)
    if not match: raise ValueError(f"missing {name}")
    return cast(float(match.group(1)))


def array(text, name):
    match = re.search(rf"\b{name}\s*\[[^]]*\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match: raise ValueError(f"missing {name}")
    return np.asarray([float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))], np.float32)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("header"); ap.add_argument("output"); args=ap.parse_args()
    text=Path(args.header).read_text(); hidden=scalar(text,"kHiddenDim",int); feature=scalar(text,"kFeatureDim",int)
    w1=array(text,"kW1").reshape(feature,hidden)
    np.savez_compressed(args.output, window=scalar(text,"kWindow",int), frame_mask=array(text,"kFrameMask"),
        mean=array(text,"kMean"), scale=array(text,"kScale"), w1=w1, b1=array(text,"kB1"), w2=array(text,"kW2"),
        b2=scalar(text,"kB2"), threshold=scalar(text,"kThreshold"), consecutive=scalar(text,"kConsecutive",int))
    print(args.output)


if __name__ == "__main__": main()
