#pragma once

#include <string>
#include <vector>

#include "fall_detector.h"
#include "tracker.h"

namespace jetson_fall {

struct StreamConfig {
    std::string id;
    std::string rtsp_url;
    bool enabled = true;
    int reconnect_delay_ms = 1000;
};

struct MqttConfig {
    bool enabled = true;
    std::string host = "127.0.0.1";
    int port = 1883;
    std::string username;
    std::string password;
    std::string topic = "recamera/fall-detection/results";
    int keepalive_sec = 30;
    bool retain = false;
    bool tls = false;
    std::string ca_file;
    std::string cert_file;
    std::string key_file;
};

struct AppConfig {
    std::string engine_path = "/models/yolo11n-pose.fp16.engine";
    int input_width = 640;
    int input_height = 640;
    float score_threshold = 0.35f;
    float keypoint_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_fps = 0;  // 0 = process as fast as the cameras provide frames
    bool publish_empty_frames = false;
    TrackerConfig tracker;
    MqttConfig mqtt;
    std::vector<StreamConfig> streams;
};

bool loadConfigFile(const std::string& path, AppConfig& config, std::string& error);
bool validateConfig(const AppConfig& config, std::string& error);

}  // namespace jetson_fall
