#include "yolo_pose.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace jetson_fall {
namespace {

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float e = std::exp(-value);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(value);
    return e / (1.0f + e);
}

float probability(float value) {
    // Exported YOLO graphs normally contain sigmoid outputs.  Keeping this
    // branch makes the parser tolerant of raw logits from custom exports.
    return (value >= 0.0f && value <= 1.0f) ? value : sigmoid(value);
}

float clamp01(float value) { return std::clamp(value, 0.0f, 1.0f); }

float coordinateToInput(float value, float extent) {
    // Values in [0,1] are normalized; values outside that range are pixels.
    return std::abs(value) <= 2.0f ? value * extent : value;
}

struct TensorLayout {
    std::size_t anchors = 0;
    std::size_t features = 0;
    bool feature_major = false;
};

bool getLayout(const std::vector<int64_t>& shape, std::size_t value_count,
               TensorLayout& out) {
    if (shape.empty()) return false;
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (const auto dim : shape) {
        if (dim > 0) dims.push_back(dim);
    }
    if (dims.size() >= 3 && dims.front() == 1) dims.erase(dims.begin());
    if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) return false;
    if (dims[0] == 56) {
        out.features = static_cast<std::size_t>(dims[0]);
        out.anchors = static_cast<std::size_t>(dims[1]);
        out.feature_major = true;
    } else if (dims[1] == 56) {
        out.anchors = static_cast<std::size_t>(dims[0]);
        out.features = static_cast<std::size_t>(dims[1]);
        out.feature_major = false;
    } else {
        // A few exports omit the class count and use 5+3*K.  We still accept
        // any feature width >= 6 with a keypoint triplet payload.
        if (dims[0] >= 6 && (dims[0] - 5) % 3 == 0) {
            out.features = static_cast<std::size_t>(dims[0]);
            out.anchors = static_cast<std::size_t>(dims[1]);
            out.feature_major = true;
        } else if (dims[1] >= 6 && (dims[1] - 5) % 3 == 0) {
            out.anchors = static_cast<std::size_t>(dims[0]);
            out.features = static_cast<std::size_t>(dims[1]);
            out.feature_major = false;
        } else {
            return false;
        }
    }
    return out.anchors > 0 && out.features <= value_count &&
           out.anchors * out.features <= value_count;
}

}  // namespace

float boxIou(const DetectionBox& a, const DetectionBox& b) {
    const float left = std::max(a.left(), b.left());
    const float top = std::max(a.top(), b.top());
    const float right = std::min(a.right(), b.right());
    const float bottom = std::min(a.bottom(), b.bottom());
    const float intersection = std::max(0.0f, right - left) *
                               std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
    const float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
    const float union_area = area_a + area_b - intersection;
    return union_area > 1e-8f ? intersection / union_area : 0.0f;
}

std::vector<PoseDetection> YoloPoseParser::parse(const float* values, std::size_t value_count,
                                                 const std::vector<int64_t>& shape,
                                                 const LetterboxInfo& letterbox) const {
    std::vector<PoseDetection> candidates;
    TensorLayout layout;
    if (values == nullptr || !getLayout(shape, value_count, layout) ||
        layout.features < 8 || letterbox.scale <= 0.0f ||
        letterbox.source_width <= 0 || letterbox.source_height <= 0) {
        return candidates;
    }
    const std::size_t keypoint_count = (layout.features - 5) / 3;
    if (keypoint_count == 0) return candidates;

    auto at = [&](std::size_t anchor, std::size_t feature) {
        return layout.feature_major ? values[feature * layout.anchors + anchor]
                                    : values[anchor * layout.features + feature];
    };
    const float input_w = static_cast<float>(std::max(1, letterbox.input_width));
    const float input_h = static_cast<float>(std::max(1, letterbox.input_height));
    const float source_w = static_cast<float>(letterbox.source_width);
    const float source_h = static_cast<float>(letterbox.source_height);

    for (std::size_t anchor = 0; anchor < layout.anchors; ++anchor) {
        const float score = probability(at(anchor, 4));
        if (score < score_threshold_) continue;

        const float cx_input = coordinateToInput(at(anchor, 0), input_w);
        const float cy_input = coordinateToInput(at(anchor, 1), input_h);
        const float w_input = std::abs(coordinateToInput(at(anchor, 2), input_w));
        const float h_input = std::abs(coordinateToInput(at(anchor, 3), input_h));
        if (!(w_input > 1e-3f && h_input > 1e-3f)) continue;

        auto undo_x = [&](float input_x) {
            return clamp01((input_x - letterbox.pad_x) / letterbox.scale / source_w);
        };
        auto undo_y = [&](float input_y) {
            return clamp01((input_y - letterbox.pad_y) / letterbox.scale / source_h);
        };
        PoseDetection detection;
        const float left_px = (cx_input - w_input * 0.5f - letterbox.pad_x) / letterbox.scale;
        const float top_px = (cy_input - h_input * 0.5f - letterbox.pad_y) / letterbox.scale;
        const float right_px = (cx_input + w_input * 0.5f - letterbox.pad_x) / letterbox.scale;
        const float bottom_px = (cy_input + h_input * 0.5f - letterbox.pad_y) / letterbox.scale;
        detection.box.x = clamp01((left_px + right_px) * 0.5f / source_w);
        detection.box.y = clamp01((top_px + bottom_px) * 0.5f / source_h);
        detection.box.w = clamp01((right_px - left_px) / source_w);
        detection.box.h = clamp01((bottom_px - top_px) / source_h);
        detection.box.score = score;
        detection.keypoints.reserve(keypoint_count);
        for (std::size_t k = 0; k < keypoint_count; ++k) {
            const std::size_t offset = 5 + k * 3;
            const float x = coordinateToInput(at(anchor, offset), input_w);
            const float y = coordinateToInput(at(anchor, offset + 1), input_h);
            const float confidence = probability(at(anchor, offset + 2));
            detection.keypoints.push_back({undo_x(x), undo_y(y), confidence});
        }
        candidates.push_back(std::move(detection));
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.box.score > b.box.score;
    });
    std::vector<PoseDetection> kept;
    kept.reserve(candidates.size());
    for (auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (boxIou(candidate.box, selected.box) > nms_threshold_) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(std::move(candidate));
    }
    return kept;
}

}  // namespace jetson_fall
