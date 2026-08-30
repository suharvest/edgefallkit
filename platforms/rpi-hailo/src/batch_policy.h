#pragma once

#include <string>

namespace rpi_hailo {
enum class BatchMode { Auto, Off, Fixed };
struct BatchConfig { BatchMode mode; int fixed_size = 1; };
struct BatchDecision { bool multi_context = false; bool shared = false; int batch_size = 1; };

BatchConfig parseBatchMode(const std::string &value);
int parseBatchWaitMs(const std::string &value);
BatchDecision chooseBatch(const BatchConfig &config, int network_group_count,
                          bool network_group_multi_context, int streams);
const char *batchModeName(const BatchConfig &config);
}
