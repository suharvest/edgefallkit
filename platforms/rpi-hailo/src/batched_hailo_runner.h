#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "frame_batcher.h"
#include "hailo_pose_decoder.h"

namespace rpi_hailo {

struct HefContextInfo {
  int network_groups = 0;
  bool multi_context = false;
};

class BatchedHailoRunner {
 public:
  using ResultHandler =
      std::function<void(BatchFrame &&, const std::vector<RawTensor> &)>;
  using ErrorHandler = std::function<void(const std::string &)>;

  static HefContextInfo inspectHef(const std::string &hef_path);

  BatchedHailoRunner(const std::string &hef_path, int batch_size,
                     FrameBatcher &batcher, ResultHandler on_result,
                     ErrorHandler on_error);
  ~BatchedHailoRunner();
  BatchedHailoRunner(const BatchedHailoRunner &) = delete;
  BatchedHailoRunner &operator=(const BatchedHailoRunner &) = delete;

  void start();
  void stop();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rpi_hailo
