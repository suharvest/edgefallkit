#!/usr/bin/env python3
"""One-frame native RKNN -> pose -> tracker -> temporal/state-machine smoke."""
import argparse,json,time
import numpy as np
from fall_core import Detection,FallConfig,FallDetector,IoUTracker,TemporalMLP,make_observation
from rknn_pose import RKNNPose,decode_pose


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--input-npy',required=True);ap.add_argument('--temporal-model',required=True);args=ap.parse_args()
 frame=np.load(args.input_npy);model=RKNNPose(args.model);outputs,ms=model.infer(frame);raw=decode_pose(outputs)
 detections=[Detection(x['box'],x['score'],x['keypoints']) for x in raw];tracks=IoUTracker().update(detections,time.time());people=[]
 for tr in tracks:
  obs=make_observation(tr.detection,time.time(),frame.shape[1],frame.shape[0]);temporal=TemporalMLP(args.temporal_model)
  positive,prob=temporal.update(temporal.pose_frame(tr.detection,obs,frame.shape[1],frame.shape[0]),obs.timestamp)
  obs.temporal_available=True;obs.temporal_positive=positive;obs.temporal_probability=prob
  people.append({'track_id':tr.track_id,'observation_valid':obs.valid,'state':FallDetector(FallConfig()).update(obs)['state']})
 model.close();print(json.dumps({'inference_ms':ms,'detections':len(detections),'people':people},indent=2))


if __name__=='__main__':main()
