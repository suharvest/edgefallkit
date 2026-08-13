#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "yolo_pose.h"
#include "preprocess_cuda.h"

namespace jetson_fall {

template <typename T>
struct TrtDeleter {
    void operator()(T* object) const {
        // TensorRT 10.x removed the old destroy() methods.  The public
        // interfaces are released with C++ delete (TRT 8 compatibility is
        // intentionally not targeted by this Jetson/TRT 10.3 solution).
        delete object;
    }
};

struct TrtRunnerConfig {
    int input_width = 640;
    int input_height = 640;
    float pad_value = 114.0f;
};

class TrtRunner {
public:
    TrtRunner() = default;
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    bool load(const std::string& engine_path, TrtRunnerConfig config = {});
    bool infer(const cv::Mat& bgr, std::vector<float>& output,
               std::vector<int64_t>& output_shape, LetterboxInfo& letterbox);
    bool initialized() const { return context_ != nullptr; }
    const std::string& inputName() const { return input_name_; }
    const std::string& outputName() const { return output_name_; }

private:
    bool allocateForInput(const nvinfer1::Dims& input_dims);
    bool ensureBgrStaging(std::size_t required_bytes);
    static std::size_t volume(const nvinfer1::Dims& dims);
    static std::size_t elementSize(nvinfer1::DataType type);
    bool checkCuda(cudaError_t status, const char* operation) const;

    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    };

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;
    cudaStream_t stream_ = nullptr;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    unsigned char* bgr_staging_device_ = nullptr;
    unsigned char* bgr_staging_host_ = nullptr;
    std::size_t bgr_staging_capacity_ = 0;
    std::size_t input_bytes_ = 0;
    std::size_t output_bytes_ = 0;
    nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
    std::string input_name_;
    std::string output_name_;
    TrtRunnerConfig config_;
};

}  // namespace jetson_fall
