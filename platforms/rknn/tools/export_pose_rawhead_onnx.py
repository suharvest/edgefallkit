#!/usr/bin/env python3
"""Export official Ultralytics YOLO11n-Pose weights as nine raw head tensors."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


class RawPoseHeads(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head = model.model[-1]
        self.features = None
        self.head.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        # The Ultralytics head replaces entries in its input list; retain the
        # original backbone tensors rather than that mutable list.
        self.features = tuple(inputs[0])

    def forward(self, image):
        self.model(image)
        outputs = []
        for i, feature in enumerate(self.features):
            outputs.extend((self.head.cv2[i](feature), self.head.cv3[i](feature),
                            self.head.cv4[i](feature)))
        return tuple(outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=640)
    args = ap.parse_args()
    pose = YOLO(args.weights).model.eval()
    wrapper = RawPoseHeads(pose).eval()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    names = [f"{kind}_{level}" for level in range(3)
             for kind in ("box", "score", "keypoints")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, dummy, args.out, input_names=["images"],
                      output_names=names, opset_version=12)


if __name__ == "__main__":
    main()
