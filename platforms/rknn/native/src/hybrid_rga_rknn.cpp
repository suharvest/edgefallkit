#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

namespace {
using Clock = std::chrono::steady_clock;
std::mutex g_rga_mutex;
thread_local std::string g_error;

struct HybridContext {
  rknn_context ctx{};
  rknn_tensor_attr input_native{};
  rknn_tensor_attr input_logical{};
  rknn_tensor_mem* input_mem{};
  std::vector<rknn_tensor_attr> output_native;
  std::vector<rknn_tensor_attr> output_logical;
  std::vector<rknn_tensor_mem*> output_mems;
  int width{};
  int height{};
  std::mutex run_mutex;
};

void set_error(const std::string& value) { g_error = value; }

std::vector<uint8_t> read_file(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return {};
  const auto size = file.tellg();
  if (size <= 0) return {};
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return file ? data : std::vector<uint8_t>{};
}

void destroy_context(HybridContext* state) {
  if (!state) return;
  for (auto* mem : state->output_mems) {
    if (mem) rknn_destroy_mem(state->ctx, mem);
  }
  if (state->input_mem) rknn_destroy_mem(state->ctx, state->input_mem);
  if (state->ctx) rknn_destroy(state->ctx);
  delete state;
}

bool dimensions(const rknn_tensor_attr& attr, int* width, int* height) {
  if (attr.n_dims != 4) return false;
  if (attr.fmt == RKNN_TENSOR_NCHW) {
    *height = static_cast<int>(attr.dims[2]);
    *width = static_cast<int>(attr.dims[3]);
  } else {
    *height = static_cast<int>(attr.dims[1]);
    *width = static_cast<int>(attr.dims[2]);
  }
  return *width > 0 && *height > 0;
}
}  // namespace

extern "C" {

const char* hybrid_last_error() { return g_error.c_str(); }

void* hybrid_create(const char* model_path, uint32_t core_mask) {
  g_error.clear();
  auto model = read_file(model_path);
  if (model.empty()) {
    set_error(std::string("cannot read model: ") + model_path);
    return nullptr;
  }
  auto* state = new HybridContext();
  int ret = rknn_init(&state->ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
  if (ret != RKNN_SUCC) {
    set_error("rknn_init=" + std::to_string(ret));
    destroy_context(state);
    return nullptr;
  }
  ret = rknn_set_core_mask(state->ctx, static_cast<rknn_core_mask>(core_mask));
  if (ret != RKNN_SUCC) {
    set_error("rknn_set_core_mask=" + std::to_string(ret));
    destroy_context(state);
    return nullptr;
  }

  rknn_input_output_num io{};
  ret = rknn_query(state->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (ret != RKNN_SUCC || io.n_input != 1 || io.n_output == 0) {
    set_error("unexpected model IO: ret=" + std::to_string(ret) +
              " inputs=" + std::to_string(io.n_input) +
              " outputs=" + std::to_string(io.n_output));
    destroy_context(state);
    return nullptr;
  }
  state->input_logical.index = 0;
  ret = rknn_query(state->ctx, RKNN_QUERY_INPUT_ATTR, &state->input_logical,
                   sizeof(state->input_logical));
  if (ret != RKNN_SUCC || !dimensions(state->input_logical, &state->width, &state->height)) {
    set_error("RKNN_QUERY_INPUT_ATTR=" + std::to_string(ret));
    destroy_context(state);
    return nullptr;
  }
  state->input_native.index = 0;
  ret = rknn_query(state->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &state->input_native,
                   sizeof(state->input_native));
  if (ret != RKNN_SUCC) {
    set_error("RKNN_QUERY_NATIVE_INPUT_ATTR=" + std::to_string(ret));
    destroy_context(state);
    return nullptr;
  }
  // Rockchip's official zero-copy example keeps the native layout but changes
  // the external element type to UINT8, allowing normalization/quantization to
  // remain fused in the NPU graph.
  state->input_native.type = RKNN_TENSOR_UINT8;
  state->input_mem = rknn_create_mem(state->ctx, state->input_native.size_with_stride);
  if (!state->input_mem) {
    set_error("rknn_create_mem(input) returned null");
    destroy_context(state);
    return nullptr;
  }
  ret = rknn_set_io_mem(state->ctx, state->input_mem, &state->input_native);
  if (ret != RKNN_SUCC) {
    set_error("rknn_set_io_mem(input)=" + std::to_string(ret));
    destroy_context(state);
    return nullptr;
  }

  state->output_native.resize(io.n_output);
  state->output_logical.resize(io.n_output);
  state->output_mems.resize(io.n_output, nullptr);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    auto& logical = state->output_logical[i];
    logical.index = i;
    ret = rknn_query(state->ctx, RKNN_QUERY_OUTPUT_ATTR, &logical, sizeof(logical));
    if (ret != RKNN_SUCC) {
      set_error("RKNN_QUERY_OUTPUT_ATTR[" + std::to_string(i) + "]=" + std::to_string(ret));
      destroy_context(state);
      return nullptr;
    }
    auto& attr = state->output_native[i];
    attr.index = i;
    ret = rknn_query(state->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN_SUCC) {
      set_error("RKNN_QUERY_NATIVE_OUTPUT_ATTR[" + std::to_string(i) + "]=" + std::to_string(ret));
      destroy_context(state);
      return nullptr;
    }
    state->output_mems[i] = rknn_create_mem(state->ctx, attr.size_with_stride);
    if (!state->output_mems[i]) {
      set_error("rknn_create_mem(output) returned null");
      destroy_context(state);
      return nullptr;
    }
    ret = rknn_set_io_mem(state->ctx, state->output_mems[i], &attr);
    if (ret != RKNN_SUCC) {
      set_error("rknn_set_io_mem(output)=" + std::to_string(ret));
      destroy_context(state);
      return nullptr;
    }
  }
  return state;
}

int hybrid_model_width(void* handle) {
  return handle ? static_cast<HybridContext*>(handle)->width : 0;
}

int hybrid_model_height(void* handle) {
  return handle ? static_cast<HybridContext*>(handle)->height : 0;
}

uint32_t hybrid_output_count(void* handle) {
  return handle ? static_cast<uint32_t>(static_cast<HybridContext*>(handle)->output_logical.size()) : 0;
}

uint32_t hybrid_output_elems(void* handle, uint32_t index) {
  if (!handle) return 0;
  auto* state = static_cast<HybridContext*>(handle);
  if (index >= state->output_logical.size()) return 0;
  return state->output_logical[index].n_elems;
}

uint32_t hybrid_output_ndims(void* handle, uint32_t index) {
  if (!handle) return 0;
  auto* state = static_cast<HybridContext*>(handle);
  if (index >= state->output_logical.size()) return 0;
  return state->output_logical[index].n_dims;
}

uint32_t hybrid_output_dim(void* handle, uint32_t index, uint32_t dim) {
  if (!handle) return 0;
  auto* state = static_cast<HybridContext*>(handle);
  if (index >= state->output_logical.size() || dim >= state->output_logical[index].n_dims) return 0;
  return state->output_logical[index].dims[dim];
}

int hybrid_infer_pose_nv12_fd(void* handle, int source_fd, int width, int height,
                              int y_stride, double* rga_ms, double* rknn_ms,
                              float* flat_outputs, uint32_t flat_capacity) {
  g_error.clear();
  if (!handle || !flat_outputs) { set_error("invalid pose infer arguments"); return -1; }
  auto* state = static_cast<HybridContext*>(handle);
  uint64_t required = 0;
  for (const auto& attr : state->output_logical) required += attr.n_elems;
  if (required > flat_capacity) { set_error("output buffer too small"); return -2; }
  if (width != state->width || height != state->height || y_stride < width) {
    set_error("pose probe requires source dimensions equal to model input"); return -3;
  }
  std::lock_guard<std::mutex> run_guard(state->run_mutex);
  const int destination_stride = state->input_native.w_stride > 0
      ? static_cast<int>(state->input_native.w_stride) : state->width;
  rga_buffer_t src = wrapbuffer_fd(source_fd, width, height,
                                   RK_FORMAT_YCbCr_420_SP, y_stride, height);
  rga_buffer_t dst = wrapbuffer_fd(state->input_mem->fd, state->width, state->height,
                                   RK_FORMAT_RGB_888, destination_stride, state->height);
  const auto before_rga = Clock::now();
  IM_STATUS rga_status;
  { std::lock_guard<std::mutex> rga_guard(g_rga_mutex);
    rga_status = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP,
                            RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1); }
  const auto after_rga = Clock::now();
  if (rga_status != IM_STATUS_SUCCESS) { set_error("imcvtcolor failed"); return -4; }
  int ret = rknn_run(state->ctx, nullptr);
  if (ret != RKNN_SUCC) { set_error("rknn_run=" + std::to_string(ret)); return -5; }
  std::vector<rknn_output> outputs(state->output_logical.size());
  uint64_t offset = 0;
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    outputs[i].index = i; outputs[i].want_float = 1; outputs[i].is_prealloc = 1;
    outputs[i].buf = flat_outputs + offset;
    outputs[i].size = state->output_logical[i].n_elems * sizeof(float);
    offset += state->output_logical[i].n_elems;
  }
  ret = rknn_outputs_get(state->ctx, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr);
  const auto after_rknn = Clock::now();
  if (ret != RKNN_SUCC) { set_error("rknn_outputs_get=" + std::to_string(ret)); return -6; }
  const int release_ret = rknn_outputs_release(state->ctx, static_cast<uint32_t>(outputs.size()), outputs.data());
  if (release_ret != RKNN_SUCC) { set_error("rknn_outputs_release=" + std::to_string(release_ret)); return -7; }
  if (rga_ms) *rga_ms = std::chrono::duration<double, std::milli>(after_rga - before_rga).count();
  if (rknn_ms) *rknn_ms = std::chrono::duration<double, std::milli>(after_rknn - after_rga).count();
  return 0;
}

int hybrid_infer_nv12_fd(void* handle, int source_fd, int width, int height,
                         int y_stride, double* rga_ms, double* rknn_ms,
                         uint64_t* output_checksum) {
  g_error.clear();
  if (!handle || source_fd < 0 || width <= 0 || height <= 0 || y_stride < width) {
    set_error("invalid infer arguments");
    return -1;
  }
  auto* state = static_cast<HybridContext*>(handle);
  if (width != state->width || height != state->height) {
    set_error("prototype requires source dimensions equal to model input");
    return -2;
  }
  std::lock_guard<std::mutex> run_guard(state->run_mutex);
  const int destination_stride = state->input_native.w_stride > 0
      ? static_cast<int>(state->input_native.w_stride) : state->width;
  rga_buffer_t src = wrapbuffer_fd(source_fd, width, height,
                                   RK_FORMAT_YCbCr_420_SP, y_stride, height);
  rga_buffer_t dst = wrapbuffer_fd(state->input_mem->fd, state->width, state->height,
                                   RK_FORMAT_RGB_888, destination_stride, state->height);
  const auto before_rga = Clock::now();
  IM_STATUS rga_status;
  {
    // Concurrent BLITs from independent MPP pipelines fail on the measured
    // RK3588 driver. A process-wide queue preserves hardware preprocessing
    // while avoiding that driver race.
    std::lock_guard<std::mutex> rga_guard(g_rga_mutex);
    rga_status = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP,
                            RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1);
  }
  const auto after_rga = Clock::now();
  if (rga_status != IM_STATUS_SUCCESS) {
    set_error(std::string("imcvtcolor=") + std::to_string(rga_status) +
              " " + imStrError(rga_status));
    return -3;
  }
  const int run_status = rknn_run(state->ctx, nullptr);
  const auto after_rknn = Clock::now();
  if (run_status != RKNN_SUCC) {
    set_error("rknn_run=" + std::to_string(run_status));
    return -4;
  }
  uint64_t checksum = 0;
  for (auto* mem : state->output_mems) {
    const int sync_status = rknn_mem_sync(state->ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (sync_status != RKNN_SUCC) {
      set_error("rknn_mem_sync(output)=" + std::to_string(sync_status));
      return -5;
    }
    if (mem->virt_addr && mem->size) {
      const auto* bytes = static_cast<const uint8_t*>(mem->virt_addr);
      checksum = checksum * 1315423911u + bytes[0] + bytes[mem->size - 1];
    }
  }
  if (rga_ms) *rga_ms = std::chrono::duration<double, std::milli>(after_rga - before_rga).count();
  if (rknn_ms) *rknn_ms = std::chrono::duration<double, std::milli>(after_rknn - after_rga).count();
  if (output_checksum) *output_checksum = checksum;
  return 0;
}

void hybrid_destroy(void* handle) { destroy_context(static_cast<HybridContext*>(handle)); }

}  // extern "C"
