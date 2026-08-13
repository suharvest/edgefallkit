#pragma once
#include <cstdint>
#include <vector>
#include <hailo/hailort.h>
#include "tracker_hailo.h"
namespace rpi_hailo {
struct RawTensor { const uint8_t* data=nullptr; hailo_vstream_info_t info{}; };
std::vector<Detection> decodeYoloV8Pose(const std::vector<RawTensor>& tensors,
                                       float score_threshold, float nms_threshold);
}
