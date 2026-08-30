#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace rpi_hailo {
struct BatchFrame { int stream = 0; uint64_t seq = 0; std::chrono::steady_clock::time_point timestamp{}; std::vector<uint8_t> rgb; };
struct BatchStats { std::vector<uint64_t> drops; std::vector<uint64_t> histogram; };
class FrameBatcher {
 public:
  FrameBatcher(int streams, int batch_size, int wait_ms, size_t depth = 2);
  bool enqueue(BatchFrame frame);
  bool take(std::vector<BatchFrame> &out);
  void stop(bool discard = true);
  BatchStats stats() const;
 private:
  mutable std::mutex mutex_; std::condition_variable cv_; std::vector<std::deque<BatchFrame>> queues_;
  int batch_size_, wait_ms_; size_t depth_; size_t rr_ = 0; bool stopped_ = false; std::vector<uint64_t> drops_, hist_;
};
}
