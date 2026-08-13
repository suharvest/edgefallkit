#pragma once

#include <cstdint>
#include <vector>

#include "pose.h"

namespace jetson_fall {

struct DetectionBox {
    float x = 0.0f;  // normalized to the original frame
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;
    float left() const { return x - w * 0.5f; }
    float right() const { return x + w * 0.5f; }
    float top() const { return y - h * 0.5f; }
    float bottom() const { return y + h * 0.5f; }
};

struct PoseDetection {
    DetectionBox box;
    std::vector<Keypoint> keypoints;
};

struct LetterboxInfo {
    int input_width = 640;
    int input_height = 640;
    int source_width = 640;
    int source_height = 640;
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

// Decode the post-processed YOLO11-Pose tensor.  Ultralytics exports most
// commonly appear as [1,56,8400] or [1,8400,56].  The parser also accepts a
// flattened [56,8400] / [8400,56] shape, making it useful with trtexec dumps.
class YoloPoseParser {
public:
    explicit YoloPoseParser(float score_threshold = 0.35f,
                            float keypoint_threshold = 0.25f,
                            float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          keypoint_threshold_(keypoint_threshold),
          nms_threshold_(nms_threshold) {}

    std::vector<PoseDetection> parse(const float* values, std::size_t value_count,
                                     const std::vector<int64_t>& shape,
                                     const LetterboxInfo& letterbox) const;

    void setScoreThreshold(float value) { score_threshold_ = value; }
    void setKeypointThreshold(float value) { keypoint_threshold_ = value; }
    void setNmsThreshold(float value) { nms_threshold_ = value; }

private:
    float score_threshold_;
    float keypoint_threshold_;
    float nms_threshold_;
};

float boxIou(const DetectionBox& a, const DetectionBox& b);

}  // namespace jetson_fall
