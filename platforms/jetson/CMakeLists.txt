cmake_minimum_required(VERSION 3.18)
project(jetson-fall-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_APP "Build the TensorRT/OpenCV native bridge" ON)
option(BUILD_TESTING "Build host-only unit tests" ON)

add_library(fall_core STATIC
    main/fall_detector.cpp
    main/pose.cpp
    main/temporal_classifier.cpp
    main/yolo_pose.cpp
    main/tracker.cpp
    main/mini_json.cpp
    main/app_config.cpp
    main/result_payload.cpp
)
target_include_directories(fall_core PUBLIC main)
target_compile_options(fall_core PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_APP)
    find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
    find_package(CUDAToolkit QUIET)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/local/include /usr/include/aarch64-linux-gnu)
    find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(MOSQUITTO QUIET IMPORTED_TARGET libmosquitto)
    endif()
    if(OpenCV_FOUND AND CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER AND
       TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        add_library(jetson_fall_trt SHARED
            main/c_api.cpp
            main/trt_runner.cpp
            main/preprocess_cuda.cu
        )
        target_include_directories(jetson_fall_trt PUBLIC main ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(jetson_fall_trt PRIVATE
            fall_core
            ${OpenCV_LIBS}
            ${TENSORRT_LIBRARY}
            CUDA::cudart
        )
        target_compile_options(jetson_fall_trt PRIVATE -Wall -Wextra -Wpedantic)
        set_target_properties(jetson_fall_trt PROPERTIES CUDA_ARCHITECTURES 87)
        set_target_properties(jetson_fall_trt PROPERTIES OUTPUT_NAME jetson_fall_trt)
        install(TARGETS jetson_fall_trt LIBRARY DESTINATION lib)

        # Keep a standalone C++ entry point useful for bring-up when the
        # optional development mosquitto package is present. Compose uses
        # app.py as the primary control plane and calls jetson_fall_trt via
        # ctypes.
        if(MOSQUITTO_FOUND)
            add_executable(jetson-fall-detection
                main/main.cpp
                main/trt_runner.cpp
                main/mqtt_publisher.cpp
                main/preprocess_cuda.cu
            )
            target_include_directories(jetson-fall-detection PRIVATE main ${TENSORRT_INCLUDE_DIR})
            target_link_libraries(jetson-fall-detection PRIVATE
                fall_core
                ${OpenCV_LIBS}
                ${TENSORRT_LIBRARY}
                CUDA::cudart
                PkgConfig::MOSQUITTO
            )
            target_compile_options(jetson-fall-detection PRIVATE -Wall -Wextra -Wpedantic)
            set_target_properties(jetson-fall-detection PROPERTIES CUDA_ARCHITECTURES 87)
            install(TARGETS jetson-fall-detection RUNTIME DESTINATION bin)
        endif()
    else()
        message(WARNING "Skipping runtime: need OpenCV(core,imgproc,videoio), CUDA and TensorRT NvInfer")
    endif()
endif()

if(BUILD_TESTING)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    add_executable(fall_detector_test tests/fall_detector_test.cpp)
    target_link_libraries(fall_detector_test PRIVATE fall_core)
    add_test(NAME fall_detector_test COMMAND fall_detector_test)

    add_executable(yolo_pose_parser_test tests/yolo_pose_parser_test.cpp)
    target_link_libraries(yolo_pose_parser_test PRIVATE fall_core)
    add_test(NAME yolo_pose_parser_test COMMAND yolo_pose_parser_test)

    add_executable(multi_person_tracker_test tests/multi_person_tracker_test.cpp)
    target_link_libraries(multi_person_tracker_test PRIVATE fall_core)
    add_test(NAME multi_person_tracker_test COMMAND multi_person_tracker_test)

    add_executable(config_test tests/config_test.cpp)
    target_link_libraries(config_test PRIVATE fall_core)
    add_test(NAME config_test COMMAND config_test)

    add_executable(temporal_profile_test tests/temporal_profile_test.cpp)
    target_link_libraries(temporal_profile_test PRIVATE fall_core)
    add_test(NAME temporal_profile_test COMMAND temporal_profile_test)
    if(Python3_Interpreter_FOUND)
        add_test(NAME python_app_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_app_test.py)
        add_test(NAME offline_evaluation_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/offline_evaluation_test.py)
        add_test(NAME calibration_sampling_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/calibration_sampling_test.py)
    endif()
endif()
