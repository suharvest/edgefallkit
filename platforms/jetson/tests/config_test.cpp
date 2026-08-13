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
    std::cout << "config_test passed\n";
}
