#include "yolo_pose.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace jetson_fall;

static void putFeatureMajor(std::vector<float>& tensor, int anchors, int feature, int anchor, float value) {
    tensor[static_cast<std::size_t>(feature) * anchors + anchor] = value;
}

int main() {
    constexpr int anchors = 3;
    constexpr int features = 56;
    std::vector<float> tensor(static_cast<std::size_t>(anchors) * features, 0.0f);
    putFeatureMajor(tensor, anchors, 0, 0, 320.0f);
    putFeatureMajor(tensor, anchors, 1, 0, 320.0f);
    putFeatureMajor(tensor, anchors, 2, 0, 200.0f);
    putFeatureMajor(tensor, anchors, 3, 0, 300.0f);
    putFeatureMajor(tensor, anchors, 4, 0, 0.95f);
    for (int keypoint = 0; keypoint < 17; ++keypoint) {
        putFeatureMajor(tensor, anchors, 5 + keypoint * 3, 0, 320.0f);
        putFeatureMajor(tensor, anchors, 6 + keypoint * 3, 0, 320.0f);
        putFeatureMajor(tensor, anchors, 7 + keypoint * 3, 0, 0.9f);
    }
    // A second, distinct person validates multi-detection parsing and NMS.
    putFeatureMajor(tensor, anchors, 0, 1, 500.0f);
    putFeatureMajor(tensor, anchors, 1, 1, 300.0f);
    putFeatureMajor(tensor, anchors, 2, 1, 100.0f);
    putFeatureMajor(tensor, anchors, 3, 1, 200.0f);
    putFeatureMajor(tensor, anchors, 4, 1, 0.88f);
    for (int keypoint = 0; keypoint < 17; ++keypoint) {
        putFeatureMajor(tensor, anchors, 5 + keypoint * 3, 1, 500.0f);
        putFeatureMajor(tensor, anchors, 6 + keypoint * 3, 1, 300.0f);
        putFeatureMajor(tensor, anchors, 7 + keypoint * 3, 1, 0.85f);
    }
    LetterboxInfo info;
    info.input_width = 640;
    info.input_height = 640;
    info.source_width = 640;
    info.source_height = 640;
    info.scale = 1.0f;
    YoloPoseParser parser;
    const auto detections = parser.parse(tensor.data(), tensor.size(), {1, 56, anchors}, info);
    assert(detections.size() == 2);
    assert(std::abs(detections[0].box.x - 0.5f) < 0.01f);
    assert(detections[0].keypoints.size() == 17);
    std::vector<float> row_major(static_cast<std::size_t>(anchors) * features, 0.0f);
    for (int feature = 0; feature < features; ++feature) {
        for (int anchor = 0; anchor < anchors; ++anchor) {
            row_major[static_cast<std::size_t>(anchor) * features + feature] =
                tensor[static_cast<std::size_t>(feature) * anchors + anchor];
        }
    }
    const auto row_detections = parser.parse(row_major.data(), row_major.size(), {1, anchors, 56}, info);
    assert(row_detections.size() == 2);
    std::cout << "yolo_pose_parser_test passed\n";
}
