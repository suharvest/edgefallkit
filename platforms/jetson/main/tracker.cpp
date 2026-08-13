#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace jetson_fall {
namespace {

float centerDistance(const DetectionBox& a, const DetectionBox& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

float midpointX(const Pose& pose, Joint a, Joint b) {
    if (pose.visible(a) && pose.visible(b)) {
        return (pose.at(a).x + pose.at(b).x) * 0.5f;
    }
    if (pose.visible(a)) return pose.at(a).x;
    if (pose.visible(b)) return pose.at(b).x;
    return 0.0f;
}

float midpointY(const Pose& pose, Joint a, Joint b) {
    if (pose.visible(a) && pose.visible(b)) {
        return (pose.at(a).y + pose.at(b).y) * 0.5f;
    }
    if (pose.visible(a)) return pose.at(a).y;
    if (pose.visible(b)) return pose.at(b).y;
    return 0.0f;
}

}  // namespace

FallObservation observationFromPose(const TrackedPerson& track, double timestamp_sec) {
    FallObservation observation;
    observation.timestamp_sec = timestamp_sec;
    observation.person_score = track.score;
    observation.valid = !track.pose.empty() && track.score > 0.0f;
    if (!observation.valid) return observation;

    const float hip_x = midpointX(track.pose, Joint::LeftHip, Joint::RightHip);
    const float hip_y = midpointY(track.pose, Joint::LeftHip, Joint::RightHip);
    const float shoulder_x = midpointX(track.pose, Joint::LeftShoulder, Joint::RightShoulder);
    const float shoulder_y = midpointY(track.pose, Joint::LeftShoulder, Joint::RightShoulder);
    observation.hip_y = hip_y;
    const float dx = shoulder_x - hip_x;
    const float dy = shoulder_y - hip_y;
    if (std::hypot(dx, dy) > 1e-3f) {
        // Angle away from vertical, independent of image aspect ratio because
        // Pose stores source pixels.  The y axis points down in image space.
        observation.torso_angle_deg = std::atan2(std::abs(dx), std::abs(dy)) *
                                      180.0f / 3.14159265358979323846f;
    }
    observation.bbox_aspect_ratio = track.box.h > 1e-6f ? track.box.w / track.box.h : 0.0f;
    return observation;
}

MultiPersonTracker::MultiPersonTracker(TrackerConfig config) : config_(std::move(config)) {
    config_.iou_threshold = std::clamp(config_.iou_threshold, 0.0f, 1.0f);
    config_.center_distance_threshold = std::max(0.0f, config_.center_distance_threshold);
    config_.max_missed_frames = std::max(1, config_.max_missed_frames);
    config_.keypoint_threshold = std::clamp(config_.keypoint_threshold, 0.0f, 1.0f);
}

void MultiPersonTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

std::vector<TrackedPerson*> MultiPersonTracker::update(
    const std::vector<PoseDetection>& detections, double timestamp_sec,
    int frame_width, int frame_height) {
    std::vector<int> assigned_track(detections.size(), -1);
    std::vector<bool> track_used(tracks_.size(), false);

    // Greedy highest-IoU matching.  A candidate with low IoU can still match
    // when its centre is close (e.g. a person crossing an occluder).
    struct Match {
        float score;
        std::size_t detection;
        std::size_t track;
    };
    std::vector<Match> matches;
    for (std::size_t d = 0; d < detections.size(); ++d) {
        for (std::size_t t = 0; t < tracks_.size(); ++t) {
            const float iou = boxIou(detections[d].box, tracks_[t]->box);
            const float distance = centerDistance(detections[d].box, tracks_[t]->box);
            if (iou >= config_.iou_threshold || distance <= config_.center_distance_threshold) {
                matches.push_back({iou + (1.0f - std::min(1.0f, distance)), d, t});
            }
        }
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        return a.score > b.score;
    });
    for (const auto& match : matches) {
        if (assigned_track[match.detection] >= 0 || track_used[match.track]) continue;
        assigned_track[match.detection] = static_cast<int>(match.track);
        track_used[match.track] = true;
    }

    for (std::size_t d = 0; d < detections.size(); ++d) {
        TrackedPerson* track = nullptr;
        if (assigned_track[d] >= 0) {
            track = tracks_[static_cast<std::size_t>(assigned_track[d])].get();
            track->box = detections[d].box;
            track->pose = Pose(detections[d].keypoints, frame_width, frame_height,
                               config_.keypoint_threshold);
            track->score = detections[d].box.score;
            ++track->age;
            track->missed = 0;
        } else {
            auto fresh = std::make_unique<TrackedPerson>();
            fresh->track_id = next_id_++;
            fresh->box = detections[d].box;
            fresh->pose = Pose(detections[d].keypoints, frame_width, frame_height,
                               config_.keypoint_threshold);
            fresh->score = detections[d].box.score;
            fresh->age = 1;
            fresh->fall.setConfig(config_.fall);
            tracks_.push_back(std::move(fresh));
            track = tracks_.back().get();
        }

        track->observation = observationFromPose(*track, timestamp_sec);
        const auto temporal = track->temporal.update(
            makeTemporalFrame(&track->pose, track->observation, frame_width, frame_height),
            timestamp_sec);
        track->observation.temporal_available = true;
        track->observation.temporal_positive = temporal.positive;
        track->observation.temporal_probability = temporal.probability;
        track->output = track->fall.update(track->observation);
    }

    // Preserve state through a short detector miss, then retire the track.
    const std::size_t original_track_count = track_used.size();
    for (std::size_t t = 0; t < original_track_count; ++t) {
        if (track_used[t]) continue;
        auto& track = tracks_[t];
        ++track->missed;
        track->observation = FallObservation{};
        track->observation.timestamp_sec = timestamp_sec;
        const TemporalFrame blank{};
        const auto temporal = track->temporal.update(blank, timestamp_sec);
        track->observation.temporal_available = true;
        track->observation.temporal_positive = temporal.positive;
        track->observation.temporal_probability = temporal.probability;
        track->output = track->fall.update(track->observation);
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const auto& track) {
        return track->missed > config_.max_missed_frames;
    }), tracks_.end());

    std::vector<TrackedPerson*> active;
    active.reserve(tracks_.size());
    for (auto& track : tracks_) {
        if (track->missed == 0) active.push_back(track.get());
    }
    return active;
}

}  // namespace jetson_fall
