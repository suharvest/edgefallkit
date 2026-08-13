#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace jetson_fall {

// CUDA data path used between the host-side capture and TensorRT.  The source
// is compact BGR bytes in device staging memory (one row after another); the
// destination is the TensorRT NCHW input allocation.  The kernel performs the
// bilinear resize, constant letterbox, BGR->RGB channel swap, normalization,
// and FP32/FP16 conversion in one pass.
enum class PreprocessOutputType {
    Float32,
    Float16,
};

bool launchCudaPreprocess(const unsigned char* source_device,
                          int source_width, int source_height,
                          int source_stride_bytes,
                          void* destination_device,
                          int destination_width, int destination_height,
                          float scale, float pad_x, float pad_y,
                          float pad_value, PreprocessOutputType output_type,
                          cudaStream_t stream);

}  // namespace jetson_fall
