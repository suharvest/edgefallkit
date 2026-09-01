#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
from pathlib import Path


path = Path(__file__).parents[1] / "tools" / "extract_gmdcsa_calibration_frames.py"
spec = importlib.util.spec_from_file_location("calibration_sampling", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as temporary:
    csv_path = Path(temporary) / "Fall.csv"
    csv_path.write_text(
        "File Name,Length (seconds),Description,Classes\n"
        "04.mp4,04,Right side fall,Falling (SW)[2.3 to 4]; Sitting[0 to 2.2]\n",
        encoding="utf-8",
    )
    assert module.fall_metadata(csv_path) == {"04.mp4": (4.0, 2.3)}

assert module.sample_times("ADL", 6.0) == [1.98, 4.0200000000000005]
fall = module.sample_times("Fall", 4.0, 2.3)
assert len(fall) == 6
assert fall[0] == 2.05
assert fall[-1] < 3.9
assert fall == sorted(fall)
print("calibration_sampling_test passed")
