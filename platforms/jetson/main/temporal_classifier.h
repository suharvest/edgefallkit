#ifndef _FALL_TEMPORAL_CLASSIFIER_H_
#define _FALL_TEMPORAL_CLASSIFIER_H_

#include <array>
#include <deque>

#include "fall_detector.h"
#include "pose.h"
#include "temporal_model_weights.h"
#include "temporal_model_weights_hailo.h"
#include "temporal_model_weights_yolo11m.h"

namespace jetson_fall {

using TemporalFrame = std::array<float, temporal_weights::kFrameDim>;

enum class TemporalProfile {
    Yolo11sPose,
    Yolo11mPose,
    Hailo8YoloV8sPose,
};

// Build the exact 56-value, pelvis-centred COCO-17 representation used by the
// training script. Missing poses deliberately become all-zero frames.
TemporalFrame makeTemporalFrame(const Pose* pose, const FallObservation& observation,
                                int inference_width, int inference_height);

struct TemporalPrediction {
    bool evaluated = false;
    bool positive = false;
    float probability = 0.0f;
};

// Tiny CPU classifier over a 3.2 second pose window. Pose inference remains in
// TensorRT; this layer is only a few thousand multiply-adds every third frame.
class TemporalClassifier {
public:
    explicit TemporalClassifier(TemporalProfile profile = TemporalProfile::Yolo11sPose)
        : profile_(profile) {}

    void reset();
    TemporalPrediction update(const TemporalFrame& frame, double timestamp_sec);

private:
    struct TimedFrame {
        double timestamp_sec = 0.0;
        TemporalFrame values{};
    };

    float evaluate(double timestamp_sec) const;

    TemporalProfile profile_;
    std::deque<TimedFrame> frames_;
    double last_timestamp_sec_ = -1.0;
    double last_evaluation_sec_ = -1.0;
    int positive_run_ = 0;
    float last_probability_ = 0.0f;
    bool last_positive_ = false;
};

}  // namespace jetson_fall

#endif  // _FALL_TEMPORAL_CLASSIFIER_H_
