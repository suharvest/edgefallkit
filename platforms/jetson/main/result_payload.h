#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tracker.h"

namespace jetson_fall {

struct StreamPayload {
    std::string stream_id;
    std::uint64_t timestamp_ms = 0;
    std::uint64_t frame_id = 0;
    float inference_time_ms = 0.0f;
    int frame_width = 0;
    int frame_height = 0;
    std::vector<const TrackedPerson*> persons;
};

// The first-level fields intentionally retain the reCamera fall-detection
// contract.  `stream_id` and `persons[]` are additive and make the same topic
// usable by a multi-camera Jetson deployment.
std::string buildResultJson(const StreamPayload& payload);

}  // namespace jetson_fall
