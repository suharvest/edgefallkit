#!/usr/bin/env python3
"""Build a fixed 640x640 TensorRT INT8 engine with an image calibrator."""
import argparse
import array
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart
from PIL import Image


W = H = 640


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images, cache, batch=1):
        super().__init__()
        self.images = images
        self.cache = cache
        self.i = 0
        self.batch = batch
        self.bytes = W * H * 3 * 4 * batch
        err, self.dev = cudart.cudaMalloc(self.bytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed: {err}")

    def __del__(self):
        dev = getattr(self, "dev", 0)
        if dev:
            cudart.cudaFree(dev)

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.i >= len(self.images):
            return None
        if self.batch != 1:
            raise RuntimeError("calibration only supports batch=1")
        vals = array.array("f", [0.0]) * (W * H * 3)
        for b in range(self.batch):
            if self.i + b >= len(self.images):
                break
            image = Image.open(self.images[self.i + b]).convert("RGB")
            scale = min(W / image.width, H / image.height)
            nw, nh = round(image.width * scale), round(image.height * scale)
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize((nw, nh), resampling)
            canvas = Image.new("RGB", (W, H), (114, 114, 114))
            canvas.paste(image, ((W - nw) // 2, (H - nh) // 2))
            sample = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)
            sample = np.ascontiguousarray(sample / 255.0)
            vals = array.array("f", sample.ravel())
        self.i += self.batch
        err, = cudart.cudaMemcpy(self.dev, vals.buffer_info()[0], self.bytes,
                                 cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy failed: {err}")
        return [int(self.dev)]

    def read_calibration_cache(self):
        if self.cache.exists():
            return self.cache.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="newline-delimited calibration image paths")
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument(
        "--fp16-layer-prefix",
        action="append",
        default=[],
        help=(
            "force matching TensorRT layer names to FP16; repeat for multiple "
            "prefixes (for example /model.22/cv4 for the YOLO pose branch)"
        ),
    )
    args = ap.parse_args()
    images = []
    for line in args.manifest.read_text().splitlines():
        if line.strip():
            path = Path(line.strip())
            images.append(path if path.is_absolute() else args.manifest.parent / path)
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"manifest has missing image: {missing[0]}")
    if len(images) < 64:
        raise SystemExit(f"need >=64 calibration images, found {len(images)}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")
    inp = network.get_input(0)
    print(f"input_name={inp.name} input_shape={inp.shape}")
    inp.shape = (1, 3, H, W)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, 3, H, W), (1, 3, H, W), (1, 3, H, W))
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    if args.fp16_layer_prefix:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        matched = []
        skipped_non_float = []
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if not any(layer.name.startswith(prefix) for prefix in args.fp16_layer_prefix):
                continue
            outputs = [
                layer.get_output(output_index)
                for output_index in range(layer.num_outputs)
                if layer.get_output(output_index) is not None
            ]
            if not outputs or any(
                output.dtype not in (trt.float16, trt.float32) for output in outputs
            ):
                skipped_non_float.append(layer.name)
                continue
            layer.precision = trt.float16
            for output_index in range(layer.num_outputs):
                layer.set_output_type(output_index, trt.float16)
            matched.append(layer.name)
        if not matched:
            raise SystemExit(
                "no TensorRT layers matched --fp16-layer-prefix="
                + ",".join(args.fp16_layer_prefix)
            )
        print(
            "fp16_layer_prefixes=" + ",".join(args.fp16_layer_prefix)
            + f" matched_layers={len(matched)}"
        )
        for name in matched:
            print(f"fp16_layer={name}")
        for name in skipped_non_float:
            print(f"fp16_skipped_non_float_layer={name}")
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.int8_calibrator = Calibrator(images, args.cache)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("INT8 build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(
        f"engine={args.engine} bytes={args.engine.stat().st_size} "
        f"images={len(images)} cache={args.cache}"
    )


if __name__ == "__main__":
    main()
