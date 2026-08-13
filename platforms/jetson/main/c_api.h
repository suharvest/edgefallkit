#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jf_trt_handle jf_trt_handle;
typedef struct jf_temporal_handle jf_temporal_handle;

typedef struct jf_detection {
    float x;
    float y;
    float w;
    float h;
    float score;
    uint32_t keypoint_offset;
    uint32_t keypoint_count;
} jf_detection;

typedef struct jf_keypoint {
    float x;
    float y;
    float confidence;
} jf_keypoint;

typedef struct jf_frame_meta {
    uint32_t detection_count;
    uint32_t keypoint_count;
    float inference_ms;
    int32_t width;
    int32_t height;
} jf_frame_meta;

typedef struct jf_temporal_result {
    int32_t evaluated;
    int32_t positive;
    float probability;
} jf_temporal_result;

// Create a TensorRT runner.  The model is loaded once and reused by all
// infer calls on this handle.  The handle is intentionally not thread-safe;
// Python owns one handle per stream (or serializes calls).
jf_trt_handle* jf_trt_create(const char* engine_path, int input_width,
                              int input_height, float score_threshold,
                              float keypoint_threshold, float nms_threshold);
void jf_trt_destroy(jf_trt_handle* handle);

// Infer one BGR frame supplied by a numpy/OpenCV buffer.  No tensor-sized
// buffer crosses the Python boundary: only compact detections and keypoints
// are copied into caller-owned arrays.
int jf_trt_infer(jf_trt_handle* handle, const uint8_t* bgr, int width, int height,
                 size_t stride_bytes, jf_detection* detections,
                 size_t detection_capacity, jf_keypoint* keypoints,
                 size_t keypoint_capacity, jf_frame_meta* meta);

const char* jf_trt_last_error(jf_trt_handle* handle);

// Optional learned temporal gate.  It owns the same compact 48-frame window
// used by the reCamera host implementation; Python passes only 17 keypoints
// and scalar geometry, never a video/tensor-sized object.
jf_temporal_handle* jf_temporal_create(void);
jf_temporal_handle* jf_temporal_create_profile(const char* profile);
void jf_temporal_destroy(jf_temporal_handle* handle);
int jf_temporal_update(jf_temporal_handle* handle, const jf_keypoint* keypoints,
                       size_t keypoint_count, int frame_width, int frame_height,
                       float hip_y, float torso_angle_deg, float bbox_aspect_ratio,
                       float person_score, int32_t valid, double timestamp_sec,
                       jf_temporal_result* result);

#ifdef __cplusplus
}
#endif
