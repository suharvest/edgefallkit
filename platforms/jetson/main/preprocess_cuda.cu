#include "preprocess_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace jetson_fall {
namespace {

template <typename T>
__device__ inline void storeNormalized(T* destination, std::size_t index, float value);

template <>
__device__ inline void storeNormalized<float>(float* destination, std::size_t index, float value) {
    destination[index] = value;
}

template <>
__device__ inline void storeNormalized<__half>(__half* destination, std::size_t index, float value) {
    destination[index] = __float2half(value);
}

__device__ inline float sourcePixel(const unsigned char* source, int width, int height,
                                    int stride, int x, int y, int channel,
                                    float pad_value) {
    (void)pad_value;
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return static_cast<float>(source[static_cast<std::size_t>(y) * stride +
                                     static_cast<std::size_t>(x) * 3 + channel]);
}

template <typename T>
__global__ void preprocessKernel(const unsigned char* source,
                                 int source_width, int source_height,
                                 int source_stride_bytes, T* destination,
                                 int destination_width, int destination_height,
                                 float scale, float pad_x, float pad_y,
                                 float pad_value) {
    const std::size_t pixel_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixel_count = static_cast<std::size_t>(destination_width) * destination_height;
    if (pixel_index >= pixel_count) return;
    const int x = static_cast<int>(pixel_index % static_cast<std::size_t>(destination_width));
    const int y = static_cast<int>(pixel_index / static_cast<std::size_t>(destination_width));
    const float resized_width = static_cast<float>(source_width) * scale;
    const float resized_height = static_cast<float>(source_height) * scale;
    const float letterbox_pad = pad_value / 255.0f;
    const bool inside = x >= pad_x && y >= pad_y &&
                        static_cast<float>(x) < pad_x + resized_width &&
                        static_cast<float>(y) < pad_y + resized_height;

    const std::size_t plane = static_cast<std::size_t>(destination_width) * destination_height;
    for (int output_channel = 0; output_channel < 3; ++output_channel) {
        float normalized = letterbox_pad;
        if (inside) {
            // Pixel-center mapping matches cv::resize's half-pixel convention
            // closely while keeping the entire preprocessing pass on CUDA.
            const float source_x = ((static_cast<float>(x) - pad_x + 0.5f) / scale) - 0.5f;
            const float source_y = ((static_cast<float>(y) - pad_y + 0.5f) / scale) - 0.5f;
            const int x0 = static_cast<int>(floorf(source_x));
            const int y0 = static_cast<int>(floorf(source_y));
            const float x_weight = source_x - static_cast<float>(x0);
            const float y_weight = source_y - static_cast<float>(y0);
            // TensorRT models conventionally consume RGB.  The capture buffer
            // is BGR, so read the opposite source channel here.
            const int source_channel = 2 - output_channel;
            const float top_left = sourcePixel(source, source_width, source_height,
                                               source_stride_bytes, x0, y0,
                                               source_channel, pad_value);
            const float top_right = sourcePixel(source, source_width, source_height,
                                                source_stride_bytes, x0 + 1, y0,
                                                source_channel, pad_value);
            const float bottom_left = sourcePixel(source, source_width, source_height,
                                                  source_stride_bytes, x0, y0 + 1,
                                                  source_channel, pad_value);
            const float bottom_right = sourcePixel(source, source_width, source_height,
                                                   source_stride_bytes, x0 + 1, y0 + 1,
                                                   source_channel, pad_value);
            const float top = top_left + (top_right - top_left) * x_weight;
            const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
            normalized = (top + (bottom - top) * y_weight) / 255.0f;
        }
        storeNormalized(destination, static_cast<std::size_t>(output_channel) * plane + pixel_index,
                        normalized);
    }
}

}  // namespace

bool launchCudaPreprocess(const unsigned char* source_device,
                          int source_width, int source_height,
                          int source_stride_bytes, void* destination_device,
                          int destination_width, int destination_height,
                          float scale, float pad_x, float pad_y,
                          float pad_value, PreprocessOutputType output_type,
                          cudaStream_t stream) {
    if (source_device == nullptr || destination_device == nullptr ||
        source_width <= 0 || source_height <= 0 || destination_width <= 0 ||
        destination_height <= 0 || source_stride_bytes < source_width * 3 ||
        !(scale > 0.0f) || stream == nullptr) {
        return false;
    }
    constexpr int kThreads = 256;
    const std::size_t pixels = static_cast<std::size_t>(destination_width) * destination_height;
    const int blocks = static_cast<int>((pixels + kThreads - 1) / kThreads);
    if (output_type == PreprocessOutputType::Float32) {
        preprocessKernel<<<blocks, kThreads, 0, stream>>>(
            source_device, source_width, source_height, source_stride_bytes,
            static_cast<float*>(destination_device), destination_width, destination_height,
            scale, pad_x, pad_y, pad_value);
    } else {
        preprocessKernel<<<blocks, kThreads, 0, stream>>>(
            source_device, source_width, source_height, source_stride_bytes,
            static_cast<__half*>(destination_device), destination_width, destination_height,
            scale, pad_x, pad_y, pad_value);
    }
    // Peek so the caller can report/clear the launch error with its normal
    // CUDA diagnostic helper.
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace jetson_fall
