#include "app_config.h"
#include "mqtt_publisher.h"
#include "result_payload.h"
#include "trt_runner.h"
#include "tracker.h"
#include "yolo_pose.h"

#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace jetson_fall {
namespace {

std::atomic<bool> running{true};

void signalHandler(int) { running.store(false); }

std::string streamTopic(const std::string& pattern, const std::string& stream_id) {
    std::string topic = pattern;
    const std::string marker = "{stream_id}";
    std::size_t position = 0;
    while ((position = topic.find(marker, position)) != std::string::npos) {
        topic.replace(position, marker.size(), stream_id);
        position += stream_id.size();
    }
    return topic;
}

class StreamWorker {
public:
    StreamWorker(StreamConfig config, const AppConfig& app, MqttPublisher* mqtt)
        : config_(std::move(config)), app_(app), mqtt_(mqtt),
          parser_(app.score_threshold, app.keypoint_threshold, app.nms_threshold),
          tracker_([&] {
              TrackerConfig tracker = app.tracker;
              tracker.keypoint_threshold = app.keypoint_threshold;
              return tracker;
          }()) {}

    void run() {
        TrtRunner runner;
        if (!runner.load(app_.engine_path, {app_.input_width, app_.input_height, 114.0f})) {
            std::cerr << '[' << config_.id << "] TensorRT engine unavailable\n";
            return;
        }
        std::uint64_t frame_id = 0;
        int reconnect_delay = std::max(100, config_.reconnect_delay_ms);
        while (running.load()) {
            cv::VideoCapture capture;
            if (!openCapture(capture)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay));
                continue;
            }
            std::cerr << '[' << config_.id << "] RTSP connected: " << config_.rtsp_url << '\n';
            cv::Mat frame;
            while (running.load()) {
                if (!capture.read(frame) || frame.empty()) {
                    std::cerr << '[' << config_.id << "] RTSP read failed; reconnecting\n";
                    capture.release();
                    tracker_.reset();
                    break;
                }
                const auto started = std::chrono::steady_clock::now();
                std::vector<float> raw;
                std::vector<int64_t> shape;
                LetterboxInfo letterbox;
                if (!runner.infer(frame, raw, shape, letterbox)) continue;
                const auto detections = parser_.parse(raw.data(), raw.size(), shape, letterbox);
                const double temporal_timestamp = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                const auto persons = tracker_.update(detections, temporal_timestamp, frame.cols, frame.rows);
                const auto finished = std::chrono::steady_clock::now();
                const float inference_ms = std::chrono::duration<float, std::milli>(finished - started).count();
                if (mqtt_ != nullptr && (app_.publish_empty_frames || !persons.empty())) {
                    StreamPayload payload;
                    payload.stream_id = config_.id;
                    payload.timestamp_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                    payload.frame_id = ++frame_id;
                    payload.inference_time_ms = inference_ms;
                    payload.frame_width = frame.cols;
                    payload.frame_height = frame.rows;
                    for (auto* person : persons) payload.persons.push_back(person);
                    const std::string json = buildResultJson(payload);
                    mqtt_->publish(streamTopic(app_.mqtt.topic, config_.id), json, app_.mqtt.retain);
                }
                if (app_.max_fps > 0) {
                    const auto budget = std::chrono::microseconds(1000000 / app_.max_fps);
                    const auto elapsed = std::chrono::steady_clock::now() - started;
                    if (elapsed < budget) std::this_thread::sleep_for(budget - elapsed);
                }
            }
        }
    }

private:
    bool openCapture(cv::VideoCapture& capture) const {
        // CAP_GSTREAMER is intentional: Jetson's accelerated RTSP depay,
        // decode and NVMM path is selected by the OpenCV build on the target.
        if (capture.open(config_.rtsp_url, cv::CAP_GSTREAMER)) return true;
        std::cerr << '[' << config_.id << "] OpenCV GStreamer could not open RTSP\n";
        return false;
    }

    StreamConfig config_;
    const AppConfig& app_;
    MqttPublisher* mqtt_ = nullptr;
    YoloPoseParser parser_;
    MultiPersonTracker tracker_;
};

}  // namespace
}  // namespace jetson_fall

int main(int argc, char** argv) {
    using namespace jetson_fall;
    std::string config_path = "/app/config/config.json";
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "-c" || argument == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (argument == "-h" || argument == "--help") {
            std::cout << "Usage: jetson-fall-detection [--config PATH]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            return 2;
        }
    }

    AppConfig config;
    std::string error;
    if (!loadConfigFile(config_path, config, error)) {
        std::cerr << "Configuration error: " << error << '\n';
        return 2;
    }
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    MqttPublisher mqtt;
    MqttPublisher* mqtt_ptr = nullptr;
    if (config.mqtt.enabled) {
        if (!mqtt.start(config.mqtt, "jetson-fall-detection")) {
            std::cerr << "MQTT disabled after connection setup failure\n";
        } else {
            mqtt_ptr = &mqtt;
        }
    }

    std::vector<std::unique_ptr<StreamWorker>> workers;
    std::vector<std::thread> threads;
    for (const auto& stream : config.streams) {
        if (!stream.enabled) continue;
        workers.push_back(std::make_unique<StreamWorker>(stream, config, mqtt_ptr));
    }
    if (workers.empty()) {
        std::cerr << "No enabled RTSP streams\n";
        return 2;
    }
    for (auto& worker : workers) threads.emplace_back(&StreamWorker::run, worker.get());
    for (auto& thread : threads) thread.join();
    mqtt.stop();
    return 0;
}
