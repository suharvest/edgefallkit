#include "app_config.h"

#include "mini_json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace jetson_fall {
namespace {

const JsonValue* objectMember(const JsonValue& object, const char* key) {
    return object.get(key);
}

int integerMember(const JsonValue& object, const char* key, int fallback) {
    const auto* value = objectMember(object, key);
    return value != nullptr && value->isNumber()
        ? static_cast<int>(std::llround(value->numberOr(fallback))) : fallback;
}

float floatMember(const JsonValue& object, const char* key, float fallback) {
    const auto* value = objectMember(object, key);
    return value != nullptr && value->isNumber()
        ? static_cast<float>(value->numberOr(fallback)) : fallback;
}

bool boolMember(const JsonValue& object, const char* key, bool fallback) {
    const auto* value = objectMember(object, key);
    return value != nullptr ? value->boolOr(fallback) : fallback;
}

std::string stringMember(const JsonValue& object, const char* key, const std::string& fallback) {
    const auto* value = objectMember(object, key);
    return value != nullptr ? value->stringOr(fallback) : fallback;
}

void applyFallConfig(const JsonValue& object, FallConfig& config) {
    if (!object.isObject()) return;
    config.hip_drop_speed_threshold = floatMember(object, "hip_drop_speed_threshold", config.hip_drop_speed_threshold);
    config.hip_drop_distance_threshold = floatMember(object, "hip_drop_distance_threshold", config.hip_drop_distance_threshold);
    config.motion_window_sec = floatMember(object, "motion_window_sec", config.motion_window_sec);
    config.torso_angle_threshold_deg = floatMember(object, "torso_angle_threshold_deg", config.torso_angle_threshold_deg);
    config.bbox_aspect_ratio_threshold = floatMember(object, "bbox_aspect_ratio_threshold", config.bbox_aspect_ratio_threshold);
    config.min_suspected_features = integerMember(object, "min_suspected_features", config.min_suspected_features);
    config.confirmation_sec = floatMember(object, "confirmation_sec", config.confirmation_sec);
    config.suspected_timeout_sec = floatMember(object, "suspected_timeout_sec", config.suspected_timeout_sec);
    config.occlusion_grace_sec = floatMember(object, "occlusion_grace_sec", config.occlusion_grace_sec);
    config.recovery_torso_angle_deg = floatMember(object, "recovery_torso_angle_deg", config.recovery_torso_angle_deg);
    config.recovery_aspect_ratio = floatMember(object, "recovery_aspect_ratio", config.recovery_aspect_ratio);
    config.recovery_window_sec = floatMember(object, "recovery_window_sec", config.recovery_window_sec);
    config.cooldown_sec = floatMember(object, "cooldown_sec", config.cooldown_sec);
}

}  // namespace

bool validateConfig(const AppConfig& config, std::string& error) {
    if (config.engine_path.empty()) {
        error = "engine_path must not be empty";
        return false;
    }
    if (config.input_width <= 0 || config.input_height <= 0 ||
        config.input_width > 8192 || config.input_height > 8192) {
        error = "input.width/height must be in 1..8192";
        return false;
    }
    if (!(config.score_threshold >= 0.0f && config.score_threshold <= 1.0f) ||
        !(config.keypoint_threshold >= 0.0f && config.keypoint_threshold <= 1.0f) ||
        !(config.nms_threshold >= 0.0f && config.nms_threshold <= 1.0f)) {
        error = "score/keypoint/nms thresholds must be in [0,1]";
        return false;
    }
    if (config.streams.empty()) {
        error = "streams must contain at least one RTSP source";
        return false;
    }
    if (config.streams.size() > 32) {
        error = "streams may contain at most 32 sources";
        return false;
    }
    for (std::size_t index = 0; index < config.streams.size(); ++index) {
        const auto& stream = config.streams[index];
        if (stream.enabled && stream.rtsp_url.empty()) {
            error = "streams[" + std::to_string(index) + "] rtsp_url must not be empty";
            return false;
        }
    }
    if (config.mqtt.enabled && (config.mqtt.host.empty() || config.mqtt.port <= 0 || config.mqtt.port > 65535 ||
                                config.mqtt.topic.empty())) {
        error = "mqtt host/port/topic is invalid";
        return false;
    }
    return true;
}

bool loadConfigFile(const std::string& path, AppConfig& config, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open config file: " + path;
        return false;
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    try {
        const JsonValue root = parseJson(contents.str());
        if (!root.isObject()) {
            error = "config root must be an object";
            return false;
        }
        config.engine_path = stringMember(root, "engine_path", config.engine_path);
        if (const auto* input_cfg = objectMember(root, "input"); input_cfg != nullptr && input_cfg->isObject()) {
            config.input_width = integerMember(*input_cfg, "width", config.input_width);
            config.input_height = integerMember(*input_cfg, "height", config.input_height);
        }
        config.score_threshold = floatMember(root, "score_threshold", config.score_threshold);
        config.keypoint_threshold = floatMember(root, "keypoint_threshold", config.keypoint_threshold);
        config.nms_threshold = floatMember(root, "nms_threshold", config.nms_threshold);
        config.max_fps = integerMember(root, "max_fps", config.max_fps);
        config.publish_empty_frames = boolMember(root, "publish_empty_frames", config.publish_empty_frames);

        if (const auto* tracker = objectMember(root, "tracker"); tracker != nullptr && tracker->isObject()) {
            config.tracker.iou_threshold = floatMember(*tracker, "iou_threshold", config.tracker.iou_threshold);
            config.tracker.center_distance_threshold = floatMember(*tracker, "center_distance_threshold", config.tracker.center_distance_threshold);
            config.tracker.max_missed_frames = integerMember(*tracker, "max_missed_frames", config.tracker.max_missed_frames);
        }
        if (const auto* fall = objectMember(root, "fall"); fall != nullptr) applyFallConfig(*fall, config.tracker.fall);
        if (const auto* mqtt = objectMember(root, "mqtt"); mqtt != nullptr && mqtt->isObject()) {
            config.mqtt.enabled = boolMember(*mqtt, "enabled", config.mqtt.enabled);
            config.mqtt.host = stringMember(*mqtt, "host", config.mqtt.host);
            config.mqtt.port = integerMember(*mqtt, "port", config.mqtt.port);
            config.mqtt.username = stringMember(*mqtt, "username", config.mqtt.username);
            config.mqtt.password = stringMember(*mqtt, "password", config.mqtt.password);
            config.mqtt.topic = stringMember(*mqtt, "topic", config.mqtt.topic);
            config.mqtt.keepalive_sec = integerMember(*mqtt, "keepalive_sec", config.mqtt.keepalive_sec);
            config.mqtt.retain = boolMember(*mqtt, "retain", config.mqtt.retain);
            config.mqtt.tls = boolMember(*mqtt, "tls", config.mqtt.tls);
            config.mqtt.ca_file = stringMember(*mqtt, "ca_file", config.mqtt.ca_file);
            config.mqtt.cert_file = stringMember(*mqtt, "cert_file", config.mqtt.cert_file);
            config.mqtt.key_file = stringMember(*mqtt, "key_file", config.mqtt.key_file);
        }
        config.streams.clear();
        if (const auto* streams = objectMember(root, "streams"); streams != nullptr && streams->isArray()) {
            for (const auto& item : std::get<JsonValue::Array>(streams->value)) {
                if (!item.isObject()) continue;
                StreamConfig stream;
                stream.id = stringMember(item, "id", "stream" + std::to_string(config.streams.size()));
                stream.rtsp_url = stringMember(item, "rtsp_url", stringMember(item, "url", ""));
                stream.enabled = boolMember(item, "enabled", stream.enabled);
                stream.reconnect_delay_ms = integerMember(item, "reconnect_delay_ms", stream.reconnect_delay_ms);
                config.streams.push_back(std::move(stream));
            }
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    return validateConfig(config, error);
}

}  // namespace jetson_fall
