#include "runtime_config.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void expectInvalid(const std::function<void()>& fn) {
  bool invalid = false;
  try { fn(); } catch (const std::invalid_argument&) { invalid = true; }
  require(invalid, "expected std::invalid_argument");
}
}

int main() {
  require(rpi_hailo_config::parseQueueDepth("1") == 1, "queue depth 1");
  require(rpi_hailo_config::parseQueueDepth("8") == 8, "queue depth 8");
  for (const std::string value : {"0", "9", "-1", "abc", "1x", ""})
    expectInvalid([&] { rpi_hailo_config::parseQueueDepth(value); });

  require(rpi_hailo_config::parseDropOnLatency("true"), "drop true");
  require(!rpi_hailo_config::parseDropOnLatency("false"), "drop false");
  require(rpi_hailo_config::parseDropOnLatency("1"), "drop 1");
  require(!rpi_hailo_config::parseDropOnLatency("0"), "drop 0");
  for (const std::string value : {"TRUE", "yes", "2", ""})
    expectInvalid([&] { rpi_hailo_config::parseDropOnLatency(value); });

  for (const std::string value : {"0", "1", "600"})
    require(rpi_hailo_config::parseNonNegativeInt("BENCHMARK_WARMUP_SECONDS", value, 600) >= 0, "warmup nonnegative");
  for (const std::string value : {"-1", "601", "abc", "1x"})
    expectInvalid([&] { rpi_hailo_config::parseNonNegativeInt("BENCHMARK_WARMUP_SECONDS", value, 600); });
  require(rpi_hailo_config::parseNonNegativeInt("BENCHMARK_SECONDS", "0") == 0, "benchmark zero");
  require(rpi_hailo_config::parseNonNegativeInt("BENCHMARK_SECONDS", "3600") == 3600, "benchmark positive");
  for (const std::string value : {"-1", "abc", "1x"})
    expectInvalid([&] { rpi_hailo_config::parseNonNegativeInt("BENCHMARK_SECONDS", value); });

  std::cout << "runtime config test passed\n";
}
