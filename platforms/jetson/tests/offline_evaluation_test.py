#!/usr/bin/env python3
import importlib.util
import math
import sys
from pathlib import Path


path = Path(__file__).parents[1] / "tools" / "evaluate_videos.py"
spec = importlib.util.spec_from_file_location("offline_eval", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

clips = [
    module.Clip(Path("adl"), "adl", 0, math.inf),
    module.Clip(Path("fall-ok"), "fall-ok", 1, 1.0),
    module.Clip(Path("fall-early"), "fall-early", 1, 2.0),
]
rows = [
    {"clip_id": "adl", "trigger": None},
    {"clip_id": "fall-ok", "trigger": 1.2},
    {"clip_id": "fall-early", "trigger": 1.0},
]
summary = module.metric_summary(clips, "trigger", rows)
assert summary["tp"] == 1 and summary["fn"] == 1
assert summary["tn"] == 1 and summary["fp"] == 0
assert summary["early_fall_alerts"] == 1
assert abs(summary["mean_detection_latency_sec"] - 0.2) < 1e-9
assert module.first_trigger([False, False, True]) == 2 / 15.0
fall_only = module.metric_summary(
    [module.Clip(Path("fall"), "fall", 1, 1.0)], "trigger",
    [{"clip_id": "fall", "trigger": 1.1}],
)
assert fall_only["recall"] == 1.0
assert fall_only["accuracy"] is None and fall_only["specificity"] is None
assert fall_only["precision"] is None and fall_only["f1"] is None
print("offline_evaluation_test passed")
