import json
import importlib.util
from pathlib import Path

p = json.loads((Path(__file__).parent / "fixtures" / "recamera_payload.json").read_text())
top = {"timestamp","frame_id","inference_time_ms","stream_id","fall_detected","fall_event",
       "event_id","global_event_id","event_id_scope","state","person_detected","person_count",
       "fallen_count","tracking","features","keypoints","pose17","persons"}
person = {"track_id","state","fall_detected","fall_event","event_id","person_detected",
          "person_score","tracking","missed_frames","bbox","features","keypoints","pose17"}
assert top <= p.keys()
assert p["event_id_scope"] == "stream_global_event_id"
assert person <= p["persons"][0].keys()
assert isinstance(p["keypoints"], list)
assert isinstance(p["persons"][0]["keypoints"], list)
feature_required = {"valid", "hip_drop_speed", "hip_drop_distance",
                    "torso_angle_deg", "bbox_aspect_ratio"}
assert feature_required <= p["features"].keys()
assert feature_required <= p["persons"][0]["features"].keys()
assert len(p["pose17"]) == len(p["persons"][0]["pose17"]) == 17
validator_path = Path(__file__).resolve().parents[3] / "contracts" / "validate_payload.py"
spec = importlib.util.spec_from_file_location("mqtt_contract", validator_path)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)
validator.validate(p)
print("payload contract passed")
