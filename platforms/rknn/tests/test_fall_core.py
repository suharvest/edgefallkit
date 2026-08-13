import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from fall_core import FallConfig,FallDetector,Observation


class FallCoreTest(unittest.TestCase):
    def test_invalid_temporal_cannot_alarm(self):
        d=FallDetector(FallConfig())
        out=d.update(Observation(False,1.0,temporal_available=True,temporal_positive=True,temporal_probability=.99))
        self.assertEqual(out["state"], "normal"); self.assertFalse(out["fall_event"])

    def test_valid_temporal_can_alarm(self):
        d=FallDetector(FallConfig())
        out=d.update(Observation(True,1.0,.5,70,1.5,.9,True,True,.99))
        self.assertEqual(out["state"], "fallen"); self.assertTrue(out["fall_event"])

    def test_default_geometry_only_cannot_confirm(self):
        d=FallDetector(FallConfig())
        d.update(Observation(True,0.0,.30,5,.5,.9))
        d.update(Observation(True,.1,.50,70,1.5,.9))
        out=d.update(Observation(True,1.0,.52,70,1.5,.9))
        self.assertEqual(out["state"], "suspected"); self.assertFalse(out["fall_event"])

    def test_explicit_legacy_geometry_can_confirm(self):
        d=FallDetector(FallConfig(temporal_confirmation_required=False))
        d.update(Observation(True,0.0,.30,5,.5,.9))
        d.update(Observation(True,.1,.50,70,1.5,.9))
        out=d.update(Observation(True,1.0,.52,70,1.5,.9))
        self.assertEqual(out["state"], "fallen"); self.assertTrue(out["fall_event"])


if __name__ == "__main__": unittest.main()
