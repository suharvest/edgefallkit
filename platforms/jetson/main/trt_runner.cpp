#include "trt_runner.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace jetson_fall {

void TrtRunner::Logger::log(Severity severity, const char* message) noexcept {
    if (message == nullptr) return;
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
}

TrtRunner::~TrtRunner() {
    if (input_device_ != nullptr) cudaFree(input_device_);
    if (output_device_ != nullptr) cudaFree(output_device_);
    if (bgr_staging_device_ != nullptr) cudaFree(bgr_staging_device_);
    if (bgr_staging_host_ != nullptr) cudaFreeHost(bgr_staging_host_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

std::size_t TrtRunner::volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return 0;
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

std::size_t TrtRunner::elementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return sizeof(__half);
        case nvinfer1::DataType::kINT8: return sizeof(std::int8_t);
        case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
        case nvinfer1::DataType::kBOOL: return sizeof(bool);
        default: return 0;
    }
}

bool TrtRunner::checkCuda(cudaError_t status, const char* operation) const {
    if (status == cudaSuccess) return true;
    std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool TrtRunner::load(const std::string& engine_path, TrtRunnerConfig config) {
    config_ = config;
    config_.input_width = std::max(1, config_.input_width);
    config_.input_height = std::max(1, config_.input_height);
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open TensorRT engine: " << engine_path << '\n';
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) return false;
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!file.read(bytes.data(), size)) return false;

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) {
        std::cerr << "TensorRT failed to deserialize engine: " << engine_path << '\n';
        return false;
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;
    if (!checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate")) return false;

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (name == nullptr) continue;
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (input_name_.empty()) input_name_ = name;
        } else if (output_name_.empty()) {
            output_name_ = name;
        }
    }
    if (input_name_.empty() || output_name_.empty()) {
        std::cerr << "TensorRT engine needs one input and one output tensor\n";
        return false;
    }

    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (input_dims.nbDims == 4) {
        input_dims.d[0] = 1;
        input_dims.d[1] = 3;
        input_dims.d[2] = config_.input_height;
        input_dims.d[3] = config_.input_width;
    }
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        std::cerr << "TensorRT rejected input shape\n";
        return false;
    }
    input_type_ = engine_->getTensorDataType(input_name_.c_str());
    output_type_ = engine_->getTensorDataType(output_name_.c_str());
    if (!allocateForInput(input_dims)) return false;
    std::cerr << "Loaded TensorRT engine " << engine_path << " (input "
              << config_.input_width << 'x' << config_.input_height << ")\n";
    return true;
}

bool TrtRunner::allocateForInput(const nvinfer1::Dims& input_dims) {
    const auto output_dims = context_->getTensorShape(output_name_.c_str());
    const auto input_type = input_type_;
    const std::size_t input_count = volume(input_dims);
    const std::size_t output_count = volume(output_dims);
    const std::size_t input_element_size = elementSize(input_type);
    const std::size_t output_element_size = elementSize(output_type_);
    if (input_count == 0 || output_count == 0 || input_element_size == 0 || output_element_size == 0) {
        std::cerr << "TensorRT dynamic shape was not resolved\n";
        return false;
    }
    input_bytes_ = input_count * input_element_size;
    output_bytes_ = output_count * output_element_size;
    if (!checkCuda(cudaMalloc(&input_device_, input_bytes_), "cudaMalloc(input)")) return false;
    if (!checkCuda(cudaMalloc(&output_device_, output_bytes_), "cudaMalloc(output)")) return false;
    if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
        !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
        std::cerr << "TensorRT failed to bind tensor addresses\n";
        return false;
    }
    return true;
}

bool TrtRunner::ensureBgrStaging(std::size_t required_bytes) {
    if (required_bytes == 0) return false;
    if (required_bytes <= bgr_staging_capacity_) return true;
    // Grow geometrically so a camera renegotiating resolution does not cause
    // a cudaMalloc on every frame. Existing capacity is retained across all
    // normal frames and only grows when a larger source arrives.
    std::size_t capacity = std::max<std::size_t>(required_bytes, 1U << 20);
    if (bgr_staging_capacity_ > 0) {
        capacity = std::max(capacity, bgr_staging_capacity_);
        while (capacity < required_bytes) capacity *= 2;
    }
    unsigned char* replacement_device = nullptr;
    unsigned char* replacement_host = nullptr;
    if (!checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&replacement_host), capacity,
                                 cudaHostAllocPortable), "cudaHostAlloc(BGR staging)")) {
        return false;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&replacement_device), capacity),
                   "cudaMalloc(BGR staging)")) {
        cudaFreeHost(replacement_host);
        return false;
    }
    if (bgr_staging_device_ != nullptr) cudaFree(bgr_staging_device_);
    if (bgr_staging_host_ != nullptr) cudaFreeHost(bgr_staging_host_);
    bgr_staging_device_ = replacement_device;
    bgr_staging_host_ = replacement_host;
    bgr_staging_capacity_ = capacity;
    return true;
}

bool TrtRunner::infer(const cv::Mat& bgr, std::vector<float>& output,
                      std::vector<int64_t>& output_shape, LetterboxInfo& letterbox) {
    output.clear();
    output_shape.clear();
    if (!initialized() || bgr.empty() || bgr.channels() != 3) return false;

    const float source_w = static_cast<float>(bgr.cols);
    const float source_h = static_cast<float>(bgr.rows);
    letterbox.input_width = config_.input_width;
    letterbox.input_height = config_.input_height;
    letterbox.source_width = bgr.cols;
    letterbox.source_height = bgr.rows;
    letterbox.scale = std::min(config_.input_width / source_w, config_.input_height / source_h);
    const int resized_w = std::max(1, static_cast<int>(std::round(source_w * letterbox.scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(source_h * letterbox.scale)));
    letterbox.pad_x = (config_.input_width - resized_w) * 0.5f;
    letterbox.pad_y = (config_.input_height - resized_h) * 0.5f;
    if (input_type_ != nvinfer1::DataType::kFLOAT &&
        input_type_ != nvinfer1::DataType::kHALF) {
        std::cerr << "Unsupported TensorRT input type (expected FP32/FP16)\n";
        return false;
    }
    const std::size_t source_bytes = static_cast<std::size_t>(bgr.cols) * bgr.rows * 3;
    if (!ensureBgrStaging(source_bytes)) return false;
    // cv::Mat may include row padding and Python's numpy buffer is usually
    // pageable. Compact into reusable pinned host staging first, then the
    // actual H2D transfer is genuinely asynchronous with respect to stream_.
    const std::size_t row_bytes = static_cast<std::size_t>(bgr.cols) * 3;
    for (int row = 0; row < bgr.rows; ++row) {
        std::memcpy(bgr_staging_host_ + static_cast<std::size_t>(row) * row_bytes,
                    bgr.ptr(row), row_bytes);
    }
    if (!checkCuda(cudaMemcpy2DAsync(bgr_staging_device_, row_bytes,
                                     bgr_staging_host_, row_bytes, row_bytes,
                                     bgr.rows, cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpy2DAsync(BGR)")) {
        return false;
    }
    const auto preprocess_type = input_type_ == nvinfer1::DataType::kFLOAT
        ? PreprocessOutputType::Float32 : PreprocessOutputType::Float16;
    if (!launchCudaPreprocess(bgr_staging_device_, bgr.cols, bgr.rows,
                              bgr.cols * 3, input_device_, config_.input_width,
                              config_.input_height, letterbox.scale,
                              letterbox.pad_x, letterbox.pad_y, config_.pad_value,
                              preprocess_type, stream_)) {
        return checkCuda(cudaGetLastError(), "CUDA preprocess launch");
    }
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "TensorRT enqueueV3 failed\n";
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) return false;

    const auto output_dims = context_->getTensorShape(output_name_.c_str());
    const std::size_t count = volume(output_dims);
    output.resize(count);
    if (output_type_ == nvinfer1::DataType::kFLOAT) {
        if (!checkCuda(cudaMemcpy(output.data(), output_device_, output_bytes_,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy(output)")) return false;
    } else if (output_type_ == nvinfer1::DataType::kHALF) {
        std::vector<__half> half_output(count);
        if (!checkCuda(cudaMemcpy(half_output.data(), output_device_, output_bytes_,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy(output half)")) return false;
        for (std::size_t i = 0; i < count; ++i) output[i] = __half2float(half_output[i]);
    } else {
        std::cerr << "Unsupported TensorRT output type (expected FP32/FP16)\n";
        return false;
    }
    output_shape.reserve(output_dims.nbDims);
    for (int i = 0; i < output_dims.nbDims; ++i) output_shape.push_back(output_dims.d[i]);
    return true;
}

}  // namespace jetson_fall
