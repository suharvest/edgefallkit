#pragma once

#include <mutex>
#include <string>

#include <mosquitto.h>

#include "app_config.h"

namespace jetson_fall {

class MqttPublisher {
public:
    MqttPublisher() = default;
    ~MqttPublisher();
    MqttPublisher(const MqttPublisher&) = delete;
    MqttPublisher& operator=(const MqttPublisher&) = delete;

    bool start(const MqttConfig& config, const std::string& client_id);
    void stop();
    bool publish(const std::string& topic, const std::string& payload,
                 bool retain = false, int qos = 0);
    bool connected() const { return connected_; }

private:
    static void onConnect(struct mosquitto*, void* userdata, int result);
    static void onDisconnect(struct mosquitto*, void* userdata, int result);

    mutable std::mutex mutex_;
    struct mosquitto* client_ = nullptr;
    bool connected_ = false;
    MqttConfig config_;
};

}  // namespace jetson_fall
