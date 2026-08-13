#include "tracker.h"

#include <cassert>
#include <iostream>

using namespace jetson_fall;

static PoseDetection person(float x) {
    PoseDetection detection;
    detection.box = {x, 0.5f, 0.2f, 0.5f, 0.9f};
    detection.keypoints.resize(17);
    for (auto& keypoint : detection.keypoints) {
        keypoint.x = x;
        keypoint.y = 0.5f;
        keypoint.confidence = 0.9f;
    }
    // Upright shoulders/hips make the observation valid and avoid accidental
    // geometry alarms in this identity-only test.
    detection.keypoints[static_cast<int>(Joint::LeftShoulder)] = {x - 0.02f, 0.35f, 0.9f};
    detection.keypoints[static_cast<int>(Joint::RightShoulder)] = {x + 0.02f, 0.35f, 0.9f};
    detection.keypoints[static_cast<int>(Joint::LeftHip)] = {x - 0.02f, 0.65f, 0.9f};
    detection.keypoints[static_cast<int>(Joint::RightHip)] = {x + 0.02f, 0.65f, 0.9f};
    return detection;
}

int main() {
    MultiPersonTracker tracker;
    auto active = tracker.update({person(0.25f), person(0.75f)}, 0.0, 640, 480);
    assert(active.size() == 2);
    const auto first_id = active[0]->track_id;
    const auto second_id = active[1]->track_id;
    active = tracker.update({person(0.27f), person(0.73f)}, 0.1, 640, 480);
    assert(active.size() == 2);
    assert(active[0]->track_id == first_id || active[0]->track_id == second_id);
    assert(active[1]->track_id == first_id || active[1]->track_id == second_id);
    assert(active[0]->track_id != active[1]->track_id);
    std::cout << "multi_person_tracker_test passed\n";
}
