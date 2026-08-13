#pragma once

// Platform-neutral COCO-17 pose representation.  TensorRT/YOLO code converts
// normalized keypoints to this type once; the fall state machine only sees
// pixels and confidence values and is therefore easy to replay in tests.

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace jetson_fall {

enum class Joint : int {
    Nose = 0,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    Count,
};

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

struct Keypoint {
    float x = 0.0f;       // normalized to the original frame
    float y = 0.0f;
    float confidence = 0.0f;
};

class Pose {
public:
    Pose() = default;
    Pose(const std::vector<Keypoint>& points, int frame_width, int frame_height,
         float keypoint_threshold = 0.50f);

    bool visible(Joint joint) const;
    Point2f at(Joint joint) const;
    float confidence(Joint joint) const;
    float sideScore(std::initializer_list<Joint> joints) const;
    bool allVisible(std::initializer_list<Joint> joints) const;
    bool empty() const { return points_.empty(); }
    std::size_t size() const { return points_.size(); }

private:
    std::vector<Point2f> points_;
    std::vector<float> confidence_;
    float threshold_ = 0.50f;
};

float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c);
inline bool isReading(float value) { return !std::isnan(value); }

}  // namespace jetson_fall
