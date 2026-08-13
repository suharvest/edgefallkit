#include "c_api.h"

#include "trt_runner.h"
#include "temporal_classifier.h"
#include "yolo_pose.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct jf_trt_handle {
    jetson_fall::TrtRunner runner;
    jetson_fall::YoloPoseParser parser;
    std::string last_error;
};

struct jf_temporal_handle {
    explicit jf_temporal_handle(jetson_fall::TemporalProfile profile) : classifier(profile) {}
    jetson_fall::TemporalClassifier classifier;
};

namespace {

void setError(jf_trt_handle* handle, const char* message) {
    if (handle != nullptr) handle->last_error = message == nullptr ? "unknown error" : message;
}

}  // namespace

extern "C" jf_trt_handle* jf_trt_create(const char* engine_path, int input_width,
                                          int input_height, float score_threshold,
                                          float keypoint_threshold, float nms_threshold) {
    if (engine_path == nullptr || engine_path[0] == '\0') return nullptr;
    auto handle = std::make_unique<jf_trt_handle>();
    handle->parser = jetson_fall::YoloPoseParser(score_threshold, keypoint_threshold, nms_threshold);
    if (!handle->runner.load(engine_path, {input_width, input_height, 114.0f})) return nullptr;
    return handle.release();
}

extern "C" void jf_trt_destroy(jf_trt_handle* handle) { delete handle; }

extern "C" int jf_trt_infer(jf_trt_handle* handle, const uint8_t* bgr, int width,
                             int height, size_t stride_bytes, jf_detection* detections,
                             size_t detection_capacity, jf_keypoint* keypoints,
                             size_t keypoint_capacity, jf_frame_meta* meta) {
    if (handle == nullptr || bgr == nullptr || width <= 0 || height <= 0 ||
        stride_bytes < static_cast<size_t>(width) * 3 || detections == nullptr ||
        keypoints == nullptr || meta == nullptr) {
        setError(handle, "invalid infer arguments");
        return -1;
    }
    cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(bgr), stride_bytes);
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> output;
    std::vector<int64_t> shape;
    jetson_fall::LetterboxInfo letterbox;
    if (!handle->runner.infer(frame, output, shape, letterbox)) {
        setError(handle, "TensorRT inference failed");
        return -2;
    }
    const auto parsed = handle->parser.parse(output.data(), output.size(), shape, letterbox);
    size_t total_keypoints = 0;
    for (const auto& item : parsed) total_keypoints += item.keypoints.size();
    if (parsed.size() > detection_capacity || total_keypoints > keypoint_capacity) {
        setError(handle, "output capacity too small");
        return -3;
    }
    size_t keypoint_offset = 0;
    for (size_t index = 0; index < parsed.size(); ++index) {
        const auto& item = parsed[index];
        detections[index] = {item.box.x, item.box.y, item.box.w, item.box.h,
                             item.box.score, static_cast<uint32_t>(keypoint_offset),
                             static_cast<uint32_t>(item.keypoints.size())};
        for (const auto& point : item.keypoints) {
            keypoints[keypoint_offset++] = {point.x, point.y, point.confidence};
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    meta->detection_count = static_cast<uint32_t>(parsed.size());
    meta->keypoint_count = static_cast<uint32_t>(keypoint_offset);
    meta->inference_ms = std::chrono::duration<float, std::milli>(finished - started).count();
    meta->width = width;
    meta->height = height;
    handle->last_error.clear();
    return 0;
}

extern "C" const char* jf_trt_last_error(jf_trt_handle* handle) {
    return handle == nullptr ? "null TensorRT handle" : handle->last_error.c_str();
}

extern "C" jf_temporal_handle* jf_temporal_create(void) {
    return new jf_temporal_handle(jetson_fall::TemporalProfile::Yolo11sPose);
}

extern "C" jf_temporal_handle* jf_temporal_create_profile(const char* profile) {
    if (profile == nullptr || std::strcmp(profile, "yolo11s-pose") == 0) {
        return new jf_temporal_handle(jetson_fall::TemporalProfile::Yolo11sPose);
    }
    if (std::strcmp(profile, "yolo11m-pose") == 0) {
        return new jf_temporal_handle(jetson_fall::TemporalProfile::Yolo11mPose);
    }
    return nullptr;
}

extern "C" void jf_temporal_destroy(jf_temporal_handle* handle) { delete handle; }

extern "C" int jf_temporal_update(jf_temporal_handle* handle, const jf_keypoint* keypoints,
                                    size_t keypoint_count, int frame_width, int frame_height,
                                    float hip_y, float torso_angle_deg,
                                    float bbox_aspect_ratio, float person_score,
                                    int32_t valid, double timestamp_sec,
                                    jf_temporal_result* result) {
    if (handle == nullptr || result == nullptr || keypoint_count > 17 ||
        (keypoint_count > 0 && keypoints == nullptr) || frame_width <= 0 || frame_height <= 0) {
        return -1;
    }
    std::vector<jetson_fall::Keypoint> points;
    points.reserve(keypoint_count);
    for (size_t index = 0; index < keypoint_count; ++index) {
        points.push_back({keypoints[index].x, keypoints[index].y, keypoints[index].confidence});
    }
    jetson_fall::Pose pose(points, frame_width, frame_height, 0.0f);
    jetson_fall::FallObservation observation;
    observation.valid = valid != 0;
    observation.timestamp_sec = timestamp_sec;
    observation.hip_y = hip_y;
    observation.torso_angle_deg = torso_angle_deg;
    observation.bbox_aspect_ratio = bbox_aspect_ratio;
    observation.person_score = person_score;
    const auto frame = jetson_fall::makeTemporalFrame(
        observation.valid ? &pose : nullptr, observation, frame_width, frame_height);
    const auto prediction = handle->classifier.update(frame, timestamp_sec);
    result->evaluated = prediction.evaluated ? 1 : 0;
    result->positive = prediction.positive ? 1 : 0;
    result->probability = prediction.probability;
    return 0;
}
