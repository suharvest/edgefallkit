#pragma once

#include <charconv>
#include <stdexcept>
#include <string>

namespace rpi_hailo_config {

inline int parseNonNegativeInt(const char* name, const std::string& value, int max_value = -1) {
  if (value.empty()) throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  int parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0 ||
      (max_value >= 0 && parsed > max_value)) {
    throw std::invalid_argument(std::string(name) + " must be a non-negative integer" +
                                (max_value >= 0 ? " within range" : ""));
  }
  return parsed;
}

inline int parseQueueDepth(const std::string& value) {
  const int parsed = parseNonNegativeInt("INFERENCE_QUEUE_DEPTH", value, 8);
  if (parsed < 1) throw std::invalid_argument("INFERENCE_QUEUE_DEPTH must be in range 1..8");
  return parsed;
}

inline bool parseDropOnLatency(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("RTSP_DROP_ON_LATENCY must be true, false, 1, or 0");
}

}  // namespace rpi_hailo_config
