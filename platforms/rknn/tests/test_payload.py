import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from app import build_payload


class PayloadContractTest(unittest.TestCase):
    def test_empty_contract(self):
        payload=build_payload("cam-01",7,1723456789.125,12.6,16.2,[],[],3)
        required={"timestamp","frame_id","inference_time_ms","stream_id","fall_detected",
          "fall_event","event_id","global_event_id","event_id_scope","state","person_detected",
          "person_count","fallen_count","tracking","features","keypoints","pose17","persons"}
        self.assertFalse(required-payload.keys())
        self.assertEqual(payload["timestamp"],1723456789125)
        self.assertEqual(payload["event_id_scope"],"stream_global_event_id")
        self.assertEqual(payload["event_id"],payload["global_event_id"])


if __name__=="__main__": unittest.main()
