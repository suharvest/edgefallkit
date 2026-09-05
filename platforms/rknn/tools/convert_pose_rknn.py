#!/usr/bin/env python3
"""Convert fixed raw-head pose ONNX to RKNN, with host-safe validation."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SUBJECT_RE = re.compile(r"(?:^|[/_.-])subject[-_]?([1-4])(?:$|[/_.-])", re.I)
EXPECTED_SHAPES = [[1, c, s, s] for s in (80, 40, 20) for c in (64, 1, 51)]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    ap.add_argument("--dataset", help="RKNN calibration list; required for explicit INT8")
    ap.add_argument("--precision", choices=("fp16", "int8"),
                    help="omitted preserves legacy dataset inference")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict-calibration", action="store_true")
    ap.add_argument("--strict-9-head", action="store_true")
    return ap


def _contract(path):
    sidecar = Path(str(path) + ".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        if data.get("raw_head_sha256") != sha256_file(path):
            raise SystemExit("aligned ONNX sidecar raw_head_sha256 mismatch")
        input_contract = data.get("input_contract", {})
        if input_contract.get("shape") != [1, 3, 640, 640] or input_contract.get("layout") != "NCHW":
            raise SystemExit("aligned ONNX sidecar has invalid input contract")
        expected = {"heads": 9, "channels": [64, 1, 51],
                    "spatial": [[80, 80], [40, 40], [20, 20]]}
        if data.get("output_contract") != expected:
            raise SystemExit("aligned ONNX sidecar has invalid 9-head contract")
        _graph_contract(path)
        return data
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        ok = len(model.graph.output) == 9
        shapes = [[d.dim_value for d in o.type.tensor_type.shape.dim]
                  for o in model.graph.output]
        ok = ok and shapes == EXPECTED_SHAPES
        return {"source": str(path), "raw_head": True} if ok else None
    except Exception:
        return None


def _graph_contract(path):
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        inputs = list(model.graph.input)
        shapes = [[d.dim_value for d in o.type.tensor_type.shape.dim]
                  for o in model.graph.output]
        if len(inputs) != 1 or [d.dim_value for d in inputs[0].type.tensor_type.shape.dim] != [1, 3, 640, 640]:
            raise SystemExit("aligned ONNX graph input contract mismatch")
        if shapes != EXPECTED_SHAPES:
            raise SystemExit("aligned ONNX graph output shape contract mismatch")
        producers = {output: node for node in model.graph.node for output in node.output}
        if any(producers.get(output.name) is None or producers[output.name].op_type != "Sigmoid"
               for output in model.graph.output[1::3]):
            raise SystemExit("aligned ONNX graph score outputs are not explicit Sigmoid probabilities")
    except ImportError as exc:
        raise SystemExit("strict aligned validation requires onnx") from exc


def _validate_calibration(path):
    sidecar = Path(str(path) + ".json")
    data = json.loads(sidecar.read_text())
    if data.get("calibration_manifest") != str(Path(path).resolve()):
        raise SystemExit("calibration sidecar manifest path mismatch")
    if data.get("calibration_manifest_sha256") != sha256_file(path):
        raise SystemExit("calibration manifest hash mismatch")
    source_list = Path(data.get("source_list", ""))
    if not source_list.is_file() or data.get("source_list_sha256") != sha256_file(source_list):
        raise SystemExit("calibration source-list hash mismatch")
    listed_sources = []
    for line in source_list.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source = Path(line.strip()).expanduser()
        listed_sources.append(str((source if source.is_absolute() else source_list.parent / source).resolve()))
    listed_derived = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    expected_sources = [item.get("source") for item in data.get("images", [])]
    expected_derived = [item.get("derived") for item in data.get("images", [])]
    if listed_sources != expected_sources or listed_derived != expected_derived:
        raise SystemExit("calibration manifest rows do not match sidecar")
    if len(set(listed_sources)) != len(listed_sources) or len(set(listed_derived)) != len(listed_derived):
        raise SystemExit("calibration manifest contains duplicate rows")
    if not data.get("images") or set(data.get("subjects", [])) - {1, 2, 3}:
        raise SystemExit("calibration sidecar contains invalid subjects")
    if len(data["images"]) != data.get("count"):
        raise SystemExit("calibration sidecar image count mismatch")
    for item in data["images"]:
        match = SUBJECT_RE.search(item.get("source", ""))
        if not match or int(match.group(1)) not in (1, 2, 3) or item.get("subject") != int(match.group(1)):
            raise SystemExit("calibration sidecar contains Subject 4 or unknown subject")
        for field in ("source", "derived"):
            image = Path(item[field])
            if not image.is_file():
                raise SystemExit(f"calibration image missing: {image}")
            if item.get(field + "_sha256") != sha256_file(image):
                raise SystemExit(f"calibration image hash mismatch: {image}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    source, target = Path(args.onnx).resolve(), Path(args.out).resolve()
    if not source.is_file():
        raise SystemExit(f"ONNX not found: {source}")
    if target.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing artifact: {target} (use --overwrite)")
    if Path(str(target) + ".json").exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing sidecar: {target}.json (use --overwrite)")
    if args.precision == "int8" and not args.dataset:
        raise SystemExit("--precision int8 requires --dataset")
    if args.precision == "fp16" and args.dataset:
        raise SystemExit("--precision fp16 cannot be combined with --dataset")
    if args.dataset and not Path(args.dataset).is_file():
        raise SystemExit(f"calibration dataset not found: {args.dataset}")
    if args.strict_9_head and not Path(str(source) + ".json").is_file():
        raise SystemExit("strict 9-head validation requires prepare_aligned_onnx sidecar")
    contract = _contract(source)
    if args.strict_9_head and contract is None:
        raise SystemExit("strict 9-head validation requires prepare_aligned_onnx sidecar")
    if args.strict_calibration and (not args.dataset or
                                    not Path(str(args.dataset) + ".json").is_file()):
        raise SystemExit("--strict-calibration requires calibration.txt.json sidecar")
    if args.strict_calibration:
        _validate_calibration(args.dataset)

    from rknn.api import RKNN  # proprietary dependency, intentionally lazy
    try:
        from importlib.metadata import version as package_version
        toolkit_version = package_version("rknn-toolkit2")
    except Exception:
        toolkit_version = "unknown"
    precision = args.precision or ("int8" if args.dataset else "fp16")
    r = RKNN(verbose=True)
    try:
        r.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                 target_platform=args.platform, quantized_dtype="w8a8",
                 quantized_algorithm="normal", optimization_level=3)
        if r.load_onnx(model=str(source)):
            raise SystemExit("load_onnx failed")
        kwargs = {"do_quantization": bool(args.dataset)}
        if args.dataset:
            kwargs["dataset"] = args.dataset
        if r.build(**kwargs):
            raise SystemExit("build failed")
        target.parent.mkdir(parents=True, exist_ok=True)
        if r.export_rknn(str(target)):
            raise SystemExit("export failed")
    finally:
        r.release()

    metadata = {
        "artifact": str(target), "bytes": target.stat().st_size,
        "sha256": sha256_file(target), "target": args.platform,
        "precision_request": precision,
        "toolkit": {"package": "rknn-toolkit2", "version": toolkit_version},
        "onnx": {"path": str(source), "sha256": sha256_file(source)},
        "input_contract": {"shape": [1, 3, 640, 640], "layout": "NCHW"},
        "output_contract": contract or {"declared": "legacy input; not independently verified"},
        "calibration": ({"path": str(Path(args.dataset).resolve()),
                         "sha256": sha256_file(args.dataset),
                         "sidecar_sha256": sha256_file(str(args.dataset) + ".json")
                         if Path(str(args.dataset) + ".json").is_file() else None}
                        if args.dataset else None),
        "note": "precision_request is not a claim that every layer uses INT8",
    }
    Path(str(target) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
