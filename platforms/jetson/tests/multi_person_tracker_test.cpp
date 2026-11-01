#include "tracker.h"

#include <cassert>
#include <cmath>
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
    // Pose keeps pixels; the observation (and the temporal frame built from
    // it) must carry hip_y normalised by the frame height.
    for (const auto* track : active) {
        assert(track->observation.valid);
        assert(std::abs(track->observation.hip_y - 0.65f) < 1e-4f);
    }

    // A pose without shoulders is invalid, never a hip_y of 0.
    TrackedPerson headless;
    headless.score = 0.9f;
    headless.box = {0.5f, 0.5f, 0.2f, 0.5f, 0.9f};
    std::vector<Keypoint> points(17);
    points[static_cast<int>(Joint::LeftHip)] = {0.5f, 0.6f, 0.9f};
    headless.pose = Pose(points, 640, 480, 0.25f);
    assert(!observationFromPose(headless, 1.0, 480).valid);
    // One shoulder and one hip are enough (one-side fallback).
    points[static_cast<int>(Joint::RightShoulder)] = {0.5f, 0.3f, 0.9f};
    headless.pose = Pose(points, 640, 480, 0.25f);
    const auto one_side = observationFromPose(headless, 1.0, 480);
    assert(one_side.valid && std::abs(one_side.hip_y - 0.6f) < 1e-4f);
    // A degenerate box is invalid.
    headless.box.h = 0.0f;
    assert(!observationFromPose(headless, 1.0, 480).valid);
    std::cout << "multi_person_tracker_test passed\n";
}
