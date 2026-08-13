cmake_minimum_required(VERSION 3.18)
project(fall_hailo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_APP "Build the Hailo runtime application" ON)
option(ENABLE_MQTT "Build libmosquitto publisher" ON)

set(JETSON_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../jetson/main)
if(BUILD_APP)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-video-1.0)
  find_package(HailoRT REQUIRED)
  find_library(HAILO_GST_LIBRARY gsthailo PATHS /usr/lib/aarch64-linux-gnu/gstreamer-1.0 REQUIRED)
  if(ENABLE_MQTT)
    find_library(MOSQUITTO_LIBRARY mosquitto REQUIRED)
  endif()
  add_executable(fall-hailo
    src/main.cpp
    src/hailo_pose_decoder.cpp
    src/tracker_hailo.cpp
    ${JETSON_COMMON}/pose.cpp
    ${JETSON_COMMON}/fall_detector.cpp
    ${JETSON_COMMON}/temporal_classifier.cpp)
  target_include_directories(fall-hailo PRIVATE src ${JETSON_COMMON})
  target_link_libraries(fall-hailo PRIVATE PkgConfig::GST HailoRT::libhailort ${HAILO_GST_LIBRARY} pthread)
  if(ENABLE_MQTT)
    target_compile_definitions(fall-hailo PRIVATE HAVE_MOSQUITTO=1)
    target_link_libraries(fall-hailo PRIVATE ${MOSQUITTO_LIBRARY})
  endif()
  target_compile_options(fall-hailo PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(hailo-logic-test
  tests/logic_test.cpp
  src/tracker_hailo.cpp
  ${JETSON_COMMON}/pose.cpp
  ${JETSON_COMMON}/fall_detector.cpp
  ${JETSON_COMMON}/temporal_classifier.cpp)
target_include_directories(hailo-logic-test PRIVATE src ${JETSON_COMMON})
add_test(NAME hailo-logic-test COMMAND hailo-logic-test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME payload-contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/payload_contract_test.py)
add_test(NAME deploy-script COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/deploy_test.py)
