#include "temporal_classifier.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace jetson_fall;

int main() {
    TemporalClassifier small(TemporalProfile::Yolo11sPose);
    TemporalClassifier medium(TemporalProfile::Yolo11mPose);
    TemporalClassifier int8s(TemporalProfile::YoloV8sInt8Pose);
    TemporalClassifier int8m(TemporalProfile::YoloV8mInt8Pose);
    TemporalClassifier hailo(TemporalProfile::Hailo8YoloV8sPose);
    TemporalFrame frame{};
    for (std::size_t index = 0; index < frame.size(); ++index) {
        frame[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03f;
    }
    TemporalPrediction small_result;
    TemporalPrediction medium_result;
    TemporalPrediction int8s_result;
    TemporalPrediction int8m_result;
    TemporalPrediction hailo_result;
    for (int index = 0; index < 60; ++index) {
        frame[index % frame.size()] += 0.01f;
        const double timestamp = index / 15.0;
        small_result = small.update(frame, timestamp);
        medium_result = medium.update(frame, timestamp);
        int8s_result = int8s.update(frame, timestamp);
        int8m_result = int8m.update(frame, timestamp);
        hailo_result = hailo.update(frame, timestamp);
    }
    assert(std::isfinite(small_result.probability));
    assert(std::isfinite(medium_result.probability));
    assert(std::isfinite(int8s_result.probability));
    assert(std::isfinite(int8m_result.probability));
    assert(std::isfinite(hailo_result.probability));
    assert(std::abs(small_result.probability - medium_result.probability) > 1e-6f);
    assert(std::abs(small_result.probability - int8s_result.probability) > 1e-6f);
    assert(std::abs(int8s_result.probability - int8m_result.probability) > 1e-6f);
    assert(std::abs(small_result.probability - hailo_result.probability) > 1e-6f);
    std::cout << "temporal_profile_test passed\n";
}
