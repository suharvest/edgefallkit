#!/usr/bin/env python3
"""Create a deterministic, letterboxed RKNN calibration image set."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SUBJECT_RE = re.compile(r"(?:^|[/_.-])subject[-_]?([1-4])(?:$|[/_.-])", re.I)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    source_list = Path(args.source_list).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not source_list.is_file():
        raise SystemExit(f"source list not found: {source_list}")
    lines = [line.strip() for line in source_list.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise SystemExit("source list is empty")
    sources = [Path(line).expanduser() for line in lines]
    sources = [(path if path.is_absolute() else source_list.parent / path).resolve() for path in sources]
    if len(set(sources)) != len(sources):
        raise SystemExit("source list contains duplicate paths")
    subjects = []
    for source in sources:
        match = SUBJECT_RE.search(str(source))
        if not match:
            raise SystemExit(f"cannot identify subject-1..3 from filename: {source}")
        subject = int(match.group(1))
        if subject == 4:
            raise SystemExit(f"Subject 4 is forbidden in calibration set: {source}")
        if not source.is_file():
            raise SystemExit(f"calibration image not found: {source}")
        subjects.append(subject)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to mix calibration runs: {out_dir} (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    records = []
    manifest = out_dir / "calibration.txt"
    with manifest.open("w") as listing:
        for index, (source, subject) in enumerate(zip(sources, subjects), 1):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise SystemExit(f"unable to read image: {source}")
            height, width = image.shape[:2]
            if height <= 0 or width <= 0:
                raise SystemExit(f"invalid image dimensions: {source}")
            scale = min(640 / width, 640 / height)
            resized_w, resized_h = round(width * scale), round(height * scale)
            resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            dw, dh = 640 - resized_w, 640 - resized_h
            left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
            top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            if padded.shape[:2] != (640, 640):
                raise SystemExit(f"letterbox produced invalid shape for {source}: {padded.shape}")
            derived = out_dir / f"calibration-{index:05d}.png"
            if not cv2.imwrite(str(derived), padded):
                raise SystemExit(f"failed to write derived image: {derived}")
            derived = derived.resolve()
            listing.write(str(derived) + "\n")
            records.append({"source": str(source), "source_sha256": sha256(source),
                            "derived": str(derived), "derived_sha256": sha256(derived),
                            "subject": subject, "source_color": "BGR",
                            "derived_color": "BGR", "size": [640, 640]})
    metadata = {
        "source_list": str(source_list), "source_list_sha256": sha256(source_list),
        "calibration_manifest": str(manifest), "calibration_manifest_sha256": sha256(manifest),
        "count": len(records), "subjects": sorted(set(subjects)),
        "subject_note": "Subjects 1-3 only; Subject 3 is present for quant calibration, not temporal validation.",
        "preprocess": {"letterbox": [640, 640], "padding": 114, "interpolation": "INTER_LINEAR",
                       "saved_format": "lossless PNG", "saved_color": "BGR for RKNN toolkit RGB reader"},
        "images": records,
    }
    Path(str(manifest) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
