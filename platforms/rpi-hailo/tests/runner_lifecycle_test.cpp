#include "runner_lifecycle.h"
#include "frame_processing_metrics.h"

#include <stdexcept>
#include <string>

namespace {
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  rpi_hailo::RunnerLifecycle lifecycle;
  check(lifecycle.shouldRunLoop(), "healthy runner should enter loop");
  check(lifecycle.exitCode(0) == 0, "healthy runner exit code");

  int posted_sources = 0;
  lifecycle.fail([&] { ++posted_sources; });
  check(posted_sources == 1, "failure must post a quit source");
  check(!lifecycle.shouldRunLoop(), "pre-loop failure must gate loop entry");
  check(lifecycle.exitCode(0) == 4, "runner failure must make cleanup fail");
  check(lifecycle.exitCode(3) == 3, "existing startup error must be preserved");

  lifecycle.fail([&] { ++posted_sources; });
  check(posted_sources == 1, "repeated errors must not queue repeated sources");

  using rpi_hailo::ProcessingBackend;
  check(std::string(rpi_hailo::pipelineMetric(ProcessingBackend::Legacy)) ==
            "pre_hailonet_to_hailonet_src",
        "legacy pipeline metric");
  check(std::string(rpi_hailo::pipelineFullMetric(ProcessingBackend::Legacy)) ==
            "pre_hailonet_to_post_tracker",
        "legacy full metric");
  check(std::string(rpi_hailo::pipelineMetric(ProcessingBackend::Shared)) ==
            "appsink_enqueue_to_hailort_completion",
        "shared pipeline metric");
  check(std::string(rpi_hailo::pipelineFullMetric(ProcessingBackend::Shared)) ==
            "appsink_enqueue_to_post_tracker",
        "shared full metric");
  check(rpi_hailo::trackerTimestampSeconds(ProcessingBackend::Legacy, 20.0,
                                           10.0) == 20.0,
        "legacy tracker must use output probe time");
  check(rpi_hailo::trackerTimestampSeconds(ProcessingBackend::Shared, 20.0,
                                           10.0) == 10.0,
        "shared tracker must use frame enqueue time");
  return 0;
}
