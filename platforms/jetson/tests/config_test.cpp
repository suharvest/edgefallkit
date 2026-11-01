#include "app_config.h"

#include <cassert>
#include <fstream>
#include <iostream>

using namespace jetson_fall;

int main() {
    const std::string path = "/tmp/jetson-fall-config-test.json";
    std::ofstream file(path);
    file << R"({
      "engine_path": "/models/test.engine",
      "input": {"width": 640, "height": 640},
      "fall": {"temporal_confirmation_required": false, "late_confirmation_sec": 1.5, "occlusion_grace_sec": 0.5},
      "mqtt": {"host": "broker", "port": 1884, "username": "u", "password": "p", "topic": "x/{stream_id}"},
      "streams": [{"id": "cam-a", "rtsp_url": "rtsp://127.0.0.1/a"}, {"id": "cam-b", "url": "rtsp://127.0.0.1/b"}]
    })";
    file.close();
    AppConfig config;
    std::string error;
    assert(loadConfigFile(path, config, error));
    assert(config.engine_path == "/models/test.engine");
    assert(config.mqtt.port == 1884 && config.mqtt.username == "u");
    assert(config.streams.size() == 2 && config.streams[1].id == "cam-b");
    assert(!config.tracker.fall.temporal_confirmation_required);
    assert(config.tracker.fall.late_confirmation_sec == 1.5f);
    assert(config.tracker.fall.occlusion_grace_sec == 0.5f);

    // Keys absent from the file keep the strict-temporal defaults.
    std::ofstream defaults(path);
    defaults << R"({
      "engine_path": "/models/test.engine",
      "fall": {"cooldown_sec": 4.0},
      "streams": [{"id": "cam-a", "rtsp_url": "rtsp://127.0.0.1/a"}]
    })";
    defaults.close();
    AppConfig fallback;
    assert(loadConfigFile(path, fallback, error));
    assert(fallback.tracker.fall.temporal_confirmation_required);
    assert(fallback.tracker.fall.late_confirmation_sec == 3.20f);
    assert(fallback.tracker.fall.cooldown_sec == 4.0f);
    std::cout << "config_test passed\n";
}
