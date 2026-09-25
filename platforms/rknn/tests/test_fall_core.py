import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from fall_core import FallConfig,FallDetector,Observation


def obs(ts,hip,torso,aspect,valid=True,positive=False):
    return Observation(valid,ts,hip,torso,aspect,.9,True,positive,.99 if positive else .1)

def replay(d,frames,positive_from=0.0):
    outs=[]
    for f in frames:
        ts,hip,torso,aspect,*rest=f
        valid=rest[0] if rest else True
        outs.append(d.update(obs(ts,hip,torso,aspect,valid,ts>=positive_from)))
    return outs

STAND=(8.0,.45); LIE=(80.0,2.45)

def fall_frames(t_fall,n,fall_sec=.4,hip0=.45,hip1=.75,stand_up_at=None,gap=None):
    for i in range(n):
        ts=i/15.0
        if stand_up_at is not None and ts>=stand_up_at:
            hip,(torso,aspect)=hip0,STAND
        elif ts<t_fall:
            hip,(torso,aspect)=hip0,STAND
        elif fall_sec>0 and ts<t_fall+fall_sec:
            k=(ts-t_fall)/fall_sec
            hip=hip0+(hip1-hip0)*k
            torso=STAND[0]+(LIE[0]-STAND[0])*k
            aspect=STAND[1]+(LIE[1]-STAND[1])*k
        else:
            hip,(torso,aspect)=hip1,LIE
        yield ts,hip,torso,aspect,not(gap and gap[0]<=ts<gap[1])


class FallCoreTest(unittest.TestCase):
    def test_invalid_temporal_cannot_alarm(self):
        d=FallDetector(FallConfig())
        out=d.update(Observation(False,1.0,temporal_available=True,temporal_positive=True,temporal_probability=.99))
        self.assertEqual(out["state"], "normal"); self.assertFalse(out["fall_event"])

    def test_temporal_only_confirms_armed_lying_candidate(self):
        # Replaces the old NORMAL->FALLEN assertion: a temporal positive in
        # NORMAL must do nothing; it confirms only a suspected lying candidate.
        d=FallDetector(FallConfig())
        out=d.update(obs(0.0,.5,70,1.5,positive=True))
        self.assertEqual(out["state"], "normal"); self.assertFalse(out["fall_event"])
        d.update(obs(0.0,.30,5,.5))
        armed=d.update(obs(.1,.50,70,1.5))
        self.assertEqual(armed["state"], "suspected")
        out=d.update(obs(1.0,.52,70,1.5,positive=True))
        self.assertEqual(out["state"], "fallen"); self.assertTrue(out["fall_event"])

    def test_default_geometry_only_cannot_confirm(self):
        d=FallDetector(FallConfig())
        d.update(obs(0.0,.30,5,.5))
        d.update(obs(.1,.50,70,1.5))
        out=d.update(obs(1.0,.52,70,1.5))
        self.assertEqual(out["state"], "suspected"); self.assertFalse(out["fall_event"])

    def test_explicit_legacy_geometry_can_confirm(self):
        d=FallDetector(FallConfig(temporal_confirmation_required=False))
        d.update(obs(0.0,.30,5,.5))
        d.update(obs(.1,.50,70,1.5))
        out=d.update(obs(1.0,.52,70,1.5))
        self.assertEqual(out["state"], "fallen"); self.assertTrue(out["fall_event"])

    def test_legacy_invalid_frames_never_originate_event(self):
        # Legacy geometry-only mode: stand, fall, then the pose goes invalid.
        # The hard invariant holds in every mode: an invalid/missing
        # observation can never originate a fall event.
        d=FallDetector(FallConfig(temporal_confirmation_required=False))
        frames=list(fall_frames(1.0,150,gap=(1.5,10.0)))
        outs=replay(d,frames,positive_from=1e9)  # temporal never positive
        invalid=[o for o,(ts,hip,torso,aspect,valid) in zip(outs,frames) if not valid]
        self.assertTrue(invalid)
        self.assertFalse(any(o["fall_event"] for o in invalid))
        self.assertFalse(any(o["fall_event"] for o in outs))

    def test_upright_person_with_temporal_positive_never_alarms(self):
        d=FallDetector(FallConfig())
        outs=replay(d,[(i/15.0,.45,8.0,.45) for i in range(150)])
        self.assertFalse(any(o["fall_event"] for o in outs))
        self.assertEqual({o["state"] for o in outs},{"normal"})

    def test_sit_down_with_temporal_positive_never_alarms(self):
        d=FallDetector(FallConfig())
        frames=[]
        for i in range(150):
            ts=i/15.0
            if ts<1.0: hip,torso,aspect=.45,8.0,.45
            elif ts<1.8:
                k=(ts-1.0)/.8; hip,torso,aspect=.45+.2*k,8.0+22.0*k,.45+.9*k
            else: hip,torso,aspect=.65,30.0,1.35
            frames.append((ts,hip,torso,aspect))
        outs=replay(d,frames,positive_from=1.2)
        self.assertFalse(any(o["fall_event"] for o in outs))
        self.assertNotIn("fallen",{o["state"] for o in outs})

    def test_real_fall_with_temporal_positive_alarms_once(self):
        d=FallDetector(FallConfig())
        outs=replay(d,list(fall_frames(1.0,150)),positive_from=1.6)
        edges=[i for i,o in enumerate(outs) if o["fall_event"]]
        self.assertEqual(len(edges),1); self.assertEqual(outs[edges[0]]["event_id"],1)
        self.assertEqual(outs[edges[0]-1]["state"],"suspected")

    def test_occluded_fall_arms_by_displacement_and_alarms_once(self):
        d=FallDetector(FallConfig())
        frames=list(fall_frames(1.0,150,fall_sec=0.0,hip0=.40,hip1=.60,gap=(1.0,2.0)))
        outs=replay(d,frames,positive_from=2.4)
        speeds=[o["features"]["hip_drop_speed"] for o in outs]
        self.assertLess(max(speeds),.25)
        self.assertEqual(sum(o["fall_event"] for o in outs),1)

    def test_late_temporal_positive_while_still_lying_confirms_once(self):
        d=FallDetector(FallConfig())
        outs=replay(d,list(fall_frames(1.0,150)),positive_from=4.0)
        self.assertEqual(sum(o["fall_event"] for o in outs),1)

    def test_late_temporal_positive_after_standing_up_does_not_alarm(self):
        d=FallDetector(FallConfig())
        outs=replay(d,list(fall_frames(1.0,150,stand_up_at=3.0)),positive_from=4.0)
        self.assertFalse(any(o["fall_event"] for o in outs))
        self.assertEqual(outs[-1]["state"],"normal")

    def test_min_features_three_confirms_stationary_lying_victim(self):
        d=FallDetector(FallConfig(min_suspected_features=3))
        outs=replay(d,list(fall_frames(1.0,150)),positive_from=1.9)
        edges=[o for o in outs if o["fall_event"]]
        self.assertEqual(len(edges),1)
        self.assertEqual(edges[0]["features"]["hip_drop_speed"],0.0)


if __name__ == "__main__": unittest.main()
