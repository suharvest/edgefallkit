#include "mqtt_publisher.h"

#include <algorithm>
#include <iostream>

namespace jetson_fall {

MqttPublisher::~MqttPublisher() { stop(); }

void MqttPublisher::onConnect(struct mosquitto*, void* userdata, int result) {
    auto* publisher = static_cast<MqttPublisher*>(userdata);
    if (publisher == nullptr) return;
    std::lock_guard<std::mutex> lock(publisher->mutex_);
    publisher->connected_ = result == 0;
    if (result != 0) std::cerr << "MQTT connect failed: " << result << '\n';
}

void MqttPublisher::onDisconnect(struct mosquitto*, void* userdata, int) {
    auto* publisher = static_cast<MqttPublisher*>(userdata);
    if (publisher == nullptr) return;
    std::lock_guard<std::mutex> lock(publisher->mutex_);
    publisher->connected_ = false;
}

bool MqttPublisher::start(const MqttConfig& config, const std::string& client_id) {
    stop();
    config_ = config;
    if (!config_.enabled) return true;
    static const int init_result = mosquitto_lib_init();
    if (init_result != MOSQ_ERR_SUCCESS) {
        std::cerr << "mosquitto_lib_init failed: " << init_result << '\n';
        return false;
    }
    client_ = mosquitto_new(client_id.c_str(), true, this);
    if (client_ == nullptr) {
        std::cerr << "mosquitto_new failed\n";
        return false;
    }
    mosquitto_connect_callback_set(client_, &MqttPublisher::onConnect);
    mosquitto_disconnect_callback_set(client_, &MqttPublisher::onDisconnect);
    if (!config_.username.empty() && mosquitto_username_pw_set(client_, config_.username.c_str(),
                                                                config_.password.empty() ? nullptr : config_.password.c_str()) != MOSQ_ERR_SUCCESS) {
        std::cerr << "mosquitto_username_pw_set failed\n";
        stop();
        return false;
    }
    if (config_.tls) {
        const char* cafile = config_.ca_file.empty() ? nullptr : config_.ca_file.c_str();
        const char* certfile = config_.cert_file.empty() ? nullptr : config_.cert_file.c_str();
        const char* keyfile = config_.key_file.empty() ? nullptr : config_.key_file.c_str();
        const int tls_result = mosquitto_tls_set(client_, cafile, nullptr, certfile, keyfile, nullptr);
        if (tls_result != MOSQ_ERR_SUCCESS) {
            std::cerr << "mosquitto_tls_set failed: " << tls_result << '\n';
            stop();
            return false;
        }
    }
    mosquitto_reconnect_delay_set(client_, 1, 30, true);
    const int connect_result = mosquitto_connect(client_, config_.host.c_str(), config_.port,
                                                 std::max(5, config_.keepalive_sec));
    if (connect_result != MOSQ_ERR_SUCCESS) {
        std::cerr << "MQTT connection failed: " << connect_result << '\n';
        stop();
        return false;
    }
    const int loop_result = mosquitto_loop_start(client_);
    if (loop_result != MOSQ_ERR_SUCCESS) {
        std::cerr << "mosquitto_loop_start failed: " << loop_result << '\n';
        stop();
        return false;
    }
    return true;
}

void MqttPublisher::stop() {
    struct mosquitto* client = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        client = client_;
        client_ = nullptr;
        connected_ = false;
    }
    if (client != nullptr) {
        mosquitto_loop_stop(client, true);
        mosquitto_disconnect(client);
        mosquitto_destroy(client);
    }
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload,
                            bool retain, int qos) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (client_ == nullptr) return false;
    const int result = mosquitto_publish(client_, nullptr, topic.c_str(),
                                         static_cast<int>(payload.size()), payload.data(),
                                         std::clamp(qos, 0, 2), retain);
    return result == MOSQ_ERR_SUCCESS;
}

}  // namespace jetson_fall
