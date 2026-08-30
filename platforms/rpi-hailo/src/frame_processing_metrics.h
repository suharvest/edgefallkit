#pragma once

namespace rpi_hailo {

enum class ProcessingBackend { Legacy, Shared };

constexpr const char *pipelineMetric(ProcessingBackend backend) {
  return backend == ProcessingBackend::Legacy
             ? "pre_hailonet_to_hailonet_src"
             : "appsink_enqueue_to_hailort_completion";
}

constexpr const char *pipelineFullMetric(ProcessingBackend backend) {
  return backend == ProcessingBackend::Legacy
             ? "pre_hailonet_to_post_tracker"
             : "appsink_enqueue_to_post_tracker";
}

constexpr double trackerTimestampSeconds(ProcessingBackend backend,
                                         double output_probe_seconds,
                                         double frame_seconds) {
  return backend == ProcessingBackend::Legacy ? output_probe_seconds
                                               : frame_seconds;
}

}  // namespace rpi_hailo
