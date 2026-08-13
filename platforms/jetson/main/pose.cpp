#include "pose.h"

#include <algorithm>

namespace jetson_fall {

Pose::Pose(const std::vector<Keypoint>& points, int frame_width, int frame_height,
           float keypoint_threshold)
    : threshold_(std::clamp(keypoint_threshold, 0.0f, 1.0f)) {
    const float width = static_cast<float>(std::max(1, frame_width));
    const float height = static_cast<float>(std::max(1, frame_height));
    points_.reserve(points.size());
    confidence_.reserve(points.size());
    for (const auto& point : points) {
        // Parsers normally hand us normalized coordinates.  Clamping here
        // keeps malformed output from poisoning geometry or JSON payloads.
        points_.push_back({std::clamp(point.x, 0.0f, 1.0f) * width,
                           std::clamp(point.y, 0.0f, 1.0f) * height});
        confidence_.push_back(std::clamp(point.confidence, 0.0f, 1.0f));
    }
}

bool Pose::visible(Joint joint) const {
    const auto index = static_cast<std::size_t>(joint);
    return index < confidence_.size() && confidence_[index] >= threshold_;
}

Point2f Pose::at(Joint joint) const {
    const auto index = static_cast<std::size_t>(joint);
    return index < points_.size() ? points_[index] : Point2f{};
}

float Pose::confidence(Joint joint) const {
    const auto index = static_cast<std::size_t>(joint);
    return index < confidence_.size() ? confidence_[index] : 0.0f;
}

float Pose::sideScore(std::initializer_list<Joint> joints) const {
    float sum = 0.0f;
    int count = 0;
    for (const auto joint : joints) {
        if (!visible(joint)) return 0.0f;
        sum += confidence(joint);
        ++count;
    }
    return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

bool Pose::allVisible(std::initializer_list<Joint> joints) const {
    for (const auto joint : joints) {
        if (!visible(joint)) return false;
    }
    return true;
}

float jointAngle(const Point2f& a, const Point2f& b, const Point2f& c) {
    const float bax = a.x - b.x;
    const float bay = a.y - b.y;
    const float bcx = c.x - b.x;
    const float bcy = c.y - b.y;
    const float norm_a = std::hypot(bax, bay);
    const float norm_c = std::hypot(bcx, bcy);
    if (norm_a < 1e-6f || norm_c < 1e-6f) return std::nanf("");
    const float cosine = std::clamp((bax * bcx + bay * bcy) / (norm_a * norm_c), -1.0f, 1.0f);
    return std::acos(cosine) * 180.0f / 3.14159265358979323846f;
}

}  // namespace jetson_fall
