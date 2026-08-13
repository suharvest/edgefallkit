#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "fall_detector.h"
#include "pose.h"
#include "temporal_classifier.h"

namespace rpi_hailo {
using jetson_fall::FallDetector;
using jetson_fall::FallObservation;
using jetson_fall::FallOutput;
using jetson_fall::Pose;
using jetson_fall::Keypoint;
struct Box { float x=0, y=0, w=0, h=0, score=0; };
struct Detection { Box box; std::vector<Keypoint> keypoints; };
struct Track {
  uint64_t id=0; Box box; Pose pose; float score=0; int age=0, missed=0;
  FallDetector fall;
  jetson_fall::TemporalClassifier temporal{jetson_fall::TemporalProfile::Hailo8YoloV8sPose};
  FallObservation observation; FallOutput output;
};
class Tracker {
 public:
  explicit Tracker(float keypoint_threshold=0.25f): keypoint_threshold_(keypoint_threshold) {}
  std::vector<Track*> update(const std::vector<Detection>&, double timestamp_sec);
  const std::vector<std::unique_ptr<Track>>& tracks() const { return tracks_; }
 private:
  float keypoint_threshold_; uint64_t next_id_=1;
  std::vector<std::unique_ptr<Track>> tracks_;
};
float iou(const Box&, const Box&);
FallObservation observationFrom(const Track&, double);
}
