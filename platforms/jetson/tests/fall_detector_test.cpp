#include "fall_detector.h"

#include <cassert>
#include <iostream>

using namespace jetson_fall;

static FallObservation frame(double timestamp, float hip_y, float torso, float aspect) {
    FallObservation observation;
    observation.valid = true;
    observation.timestamp_sec = timestamp;
    observation.hip_y = hip_y;
    observation.torso_angle_deg = torso;
    observation.bbox_aspect_ratio = aspect;
    observation.person_score = 0.9f;
    return observation;
}

int main() {
    FallConfig config;
    config.confirmation_sec = 0.6f;
    config.suspected_timeout_sec = 1.2f;
    config.occlusion_grace_sec = 0.8f;
    config.recovery_window_sec = 0.8f;
    config.cooldown_sec = 1.0f;
    config.temporal_confirmation_required = false;
    FallDetector detector(config);
    FallDetector first_frame(config);
    auto first = first_frame.update(frame(0.0, 0.75f, 72.0f, 1.55f));
    assert(first.state == FallState::Normal && !first.fall_detected && !first.fall_event);
    auto output = detector.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    assert(output.state == FallState::Normal);
    output = detector.update(frame(0.25, 0.72f, 65.0f, 1.45f));
    assert(output.state == FallState::Suspected && !output.fall_event);
    output = detector.update(frame(0.55, 0.75f, 72.0f, 1.55f));
    assert(output.state == FallState::Suspected);
    output = detector.update(frame(1.25, 0.75f, 72.0f, 1.55f));
    assert(output.state == FallState::Fallen && output.fall_event && output.event_id == 1);

    FallDetector occluded(config);
    occluded.update(frame(0.0, 0.5f, 10.0f, 0.65f));
    occluded.update(frame(0.2, 0.7f, 68.0f, 1.45f));
    FallObservation missing;
    missing.timestamp_sec = 0.85;
    output = occluded.update(missing);
    assert(output.state == FallState::Suspected && !output.fall_event);

    FallConfig strict = config;
    strict.temporal_confirmation_required = true;
    FallDetector learned(strict);
    learned.update(frame(0.0, 0.50f, 10.0f, 0.65f));
    learned.update(frame(0.25, 0.72f, 65.0f, 1.45f));
    output = learned.update(frame(1.25, 0.75f, 72.0f, 1.55f));
    assert(output.state == FallState::Suspected && !output.fall_event);
    auto confirmed = frame(1.45, 0.75f, 72.0f, 1.55f);
    confirmed.temporal_available = true;
    confirmed.temporal_positive = true;
    output = learned.update(confirmed);
    assert(output.state == FallState::Fallen && output.fall_event);
    std::cout << "fall_detector_test passed\n";
}
