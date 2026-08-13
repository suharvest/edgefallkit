#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "fall_detector.h"
#include "temporal_classifier.h"
#include "yolo_pose.h"

namespace jetson_fall {

struct TrackedPerson {
    std::uint64_t track_id = 0;
    DetectionBox box;
    Pose pose;
    float score = 0.0f;
    int age = 0;
    int missed = 0;
    FallDetector fall;
    TemporalClassifier temporal;
    FallOutput output;
    FallObservation observation;
};

struct TrackerConfig {
    float iou_threshold = 0.20f;
    float center_distance_threshold = 0.25f;
    int max_missed_frames = 8;
    float keypoint_threshold = 0.25f;
    FallConfig fall;
};

// Lightweight greedy tracker.  Each track owns its temporal window and fall
// state machine, so a second person can never splice into the first person's
// history.  This intentionally avoids a heavyweight dependency such as
// DeepSORT; YOLO pose boxes at 15-30 FPS are sufficient for the edge case.
class MultiPersonTracker {
public:
    explicit MultiPersonTracker(TrackerConfig config = {});

    void reset();
    std::vector<TrackedPerson*> update(const std::vector<PoseDetection>& detections,
                                       double timestamp_sec, int frame_width,
                                       int frame_height);
    const std::vector<std::unique_ptr<TrackedPerson>>& tracks() const { return tracks_; }
    const TrackerConfig& config() const { return config_; }

private:
    TrackerConfig config_;
    std::uint64_t next_id_ = 1;
    std::vector<std::unique_ptr<TrackedPerson>> tracks_;
};

FallObservation observationFromPose(const TrackedPerson& track, double timestamp_sec);

}  // namespace jetson_fall
