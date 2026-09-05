#!/usr/bin/env python3
"""Extract the nine fixed YOLOv8-Pose raw heads without changing logits."""
import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

HEAD_RE = re.compile(r"^/model\.22/cv([234])\.([0-2])/cv\1\.\2\.2/Conv_output_0$")
EXPECTED = {"heads": 9, "channels": [64, 1, 51],
            "spatial": [[80, 80], [40, 40], [20, 20]]}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--input", default=None)
    return ap.parse_args()


def main():
    args = parse()
    source, target = Path(args.onnx).resolve(), Path(args.out).resolve()
    sidecar = Path(str(target) + ".json")
    if not source.is_file():
        raise SystemExit(f"ONNX not found: {source}")
    existing = [path for path in (target, sidecar) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("refusing to overwrite existing artifact(s): " +
                         ", ".join(map(str, existing)) + " (use --overwrite)")
    import onnx
    from onnx import checker, shape_inference
    model = onnx.load(str(source), load_external_data=False)
    checker.check_model(model)
    inputs = list(model.graph.input)
    if len(inputs) != 1:
        raise SystemExit(f"expected one input, found {len(inputs)}")
    inp = inputs[0]
    input_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    # The Jetson S export uses symbolic batch/height/width.  Freeze only the
    # declared input contract; no weights or operator graph are optimized.
    if len(inp.type.tensor_type.shape.dim) != 4:
        raise SystemExit(f"input must have rank 4, got {input_shape}")
    for dim, value in zip(inp.type.tensor_type.shape.dim, (1, 3, 640, 640)):
        dim.ClearField("dim_param")
        dim.dim_value = value
    input_name = args.input or inp.name
    if input_name != inp.name:
        raise SystemExit(f"--input must name the graph input ({inp.name})")

    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if dims:
            shapes[value.name] = dims
    tensors = {out for node in model.graph.node for out in node.output}
    selected = {}
    for name in tensors:
        match = HEAD_RE.fullmatch(name)
        if match:
            branch, level = int(match.group(1)), int(match.group(2))
            selected[(level, branch)] = name
    missing = [(level, branch) for level in range(3) for branch in (2, 3, 4)
               if (level, branch) not in selected]
    if missing:
        raise SystemExit("missing anchored raw-head endpoints: " + ", ".join(map(str, missing)))

    # The RKNN decoder accepts score probabilities.  Make this explicit in the
    # extracted graph instead of relying on its legacy value-range heuristic.
    # Box and keypoint heads remain raw logits; only cv3 scores get Sigmoid.
    output_names = []
    for level in range(3):
        for branch in (2, 3, 4):
            name = selected[(level, branch)]
            if branch == 3:
                sigmoid_name = name.replace("/Conv_output_0", "/Sigmoid_output_0")
                model.graph.node.append(onnx.helper.make_node(
                    "Sigmoid", [name], [sigmoid_name],
                    name=name.replace("/Conv_output_0", "/Sigmoid")))
                output_names.append(sigmoid_name)
            else:
                output_names.append(name)
    expected_shapes = [[1, c, s, s] for s in (80, 40, 20) for c in (64, 1, 51)]
    actual = [shapes.get(selected[(level, branch)])
              for level in range(3) for branch in (2, 3, 4)]
    if actual != expected_shapes:
        raise SystemExit(f"raw-head shape contract mismatch: expected {expected_shapes}, got {actual}")

    target.parent.mkdir(parents=True, exist_ok=True)
    import onnx.utils
    # extract_model reloads the source path, so persist the one-node score
    # semantic change in a short-lived intermediate file first.
    with tempfile.TemporaryDirectory(prefix="rknn-aligned-") as td:
        intermediate = Path(td) / "with-score-sigmoid.onnx"
        onnx.save(model, str(intermediate))
        onnx.utils.extract_model(str(intermediate), str(target), [input_name], output_names,
                                 check_model=True)
    metadata = {
        "source": str(source), "source_sha256": sha256(source),
        "artifact": str(target), "raw_head_sha256": sha256(target),
        "raw_head_bytes": target.stat().st_size, "endpoint_names": output_names,
        "input_contract": {"name": input_name, "shape": [1, 3, 640, 640], "layout": "NCHW"},
        "output_contract": EXPECTED,
        "score_semantics": "probabilities/Sigmoid (cv3); box and keypoint heads remain raw logits",
        "semantics": "raw box/keypoint logits plus explicit score Sigmoid; no graph optimization or weight change",
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
