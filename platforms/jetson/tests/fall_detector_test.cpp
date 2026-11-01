#include "fall_detector.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace jetson_fall;

namespace {

constexpr float kStandTorso = 8.0f, kStandAspect = 0.45f;
constexpr float kLieTorso = 80.0f, kLieAspect = 2.45f;

struct ScriptedFrame {
    double ts;
    float hip;
    float torso;
    float aspect;
    bool valid;
};

// Standing, a fall_sec fall starting at t_fall, then lying.  [gap_a, gap_b)
// marks invalid pose; stand_up_at switches back to upright.
std::vector<ScriptedFrame> fallFrames(double t_fall, int n, double fall_sec = 0.4,
                                      float hip0 = 0.45f, float hip1 = 0.75f,
                                      double stand_up_at = -1.0,
                                      double gap_a = -1.0, double gap_b = -1.0) {
    std::vector<ScriptedFrame> frames;
    for (int i = 0; i < n; ++i) {
        const double ts = i / 15.0;
        float hip, torso, aspect;
        if ((stand_up_at >= 0.0 && ts >= stand_up_at) || ts < t_fall) {
            hip = hip0; torso = kStandTorso; aspect = kStandAspect;
        } else if (fall_sec > 0 && ts < t_fall + fall_sec) {
            const float k = static_cast<float>((ts - t_fall) / fall_sec);
            hip = hip0 + (hip1 - hip0) * k;
            torso = kStandTorso + (kLieTorso - kStandTorso) * k;
            aspect = kStandAspect + (kLieAspect - kStandAspect) * k;
        } else {
            hip = hip1; torso = kLieTorso; aspect = kLieAspect;
        }
        const bool valid = !(gap_a >= 0.0 && ts >= gap_a && ts < gap_b);
        frames.push_back({ts, hip, torso, aspect, valid});
    }
    return frames;
}

// Replay frames; the temporal gate is positive from positive_from onward.
std::vector<FallOutput> run(FallDetector& detector,
                            const std::vector<ScriptedFrame>& frames,
                            double positive_from) {
    std::vector<FallOutput> outs;
    for (const auto& f : frames) {
        FallObservation o;
        o.valid = f.valid;
        o.timestamp_sec = f.ts;
        o.hip_y = f.hip;
        o.torso_angle_deg = f.torso;
        o.bbox_aspect_ratio = f.aspect;
        o.person_score = 0.9f;
        const bool positive = f.ts >= positive_from;
        o.temporal_available = true;
        o.temporal_positive = positive;
        o.temporal_probability = positive ? 0.99f : 0.1f;
        outs.push_back(detector.update(o));
    }
    return outs;
}

int eventCount(const std::vector<FallOutput>& outs) {
    int n = 0;
    for (const auto& o : outs) n += o.fall_event ? 1 : 0;
    return n;
}

}  // namespace

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

    // ---- scenario suite: strict temporal confirmation policy ----
    // 1. Upright person with a temporal positive on every frame: 0 events.
    {
        FallDetector d({});
        int events = 0;
        for (int i = 0; i < 150; ++i) {
            auto o = frame(i / 15.0, 0.45f, 8.0f, 0.45f);
            o.temporal_available = true; o.temporal_positive = true; o.temporal_probability = 0.99f;
            events += d.update(o).fall_event ? 1 : 0;
        }
        assert(events == 0 && d.state() == FallState::Normal);
    }
    // 7. A temporal positive on an invalid/no-pose frame in Normal: 0 events.
    {
        FallDetector d({});
        d.update(frame(0.0, 0.45f, 8.0f, 0.45f));
        FallObservation o;
        o.timestamp_sec = 0.2;
        o.temporal_available = true; o.temporal_positive = true; o.temporal_probability = 0.99f;
        auto out = d.update(o);
        assert(out.state == FallState::Normal && !out.fall_event);
    }
    // 2. Sit-down: fast hip drop and a wide seated box, torso not lying: 0 events.
    {
        FallDetector d({});
        int events = 0;
        for (int i = 0; i < 150; ++i) {
            const double ts = i / 15.0;
            float hip, torso, aspect;
            if (ts < 1.0) { hip = 0.45f; torso = 8.0f; aspect = 0.45f; }
            else if (ts < 1.8) {
                const float k = static_cast<float>((ts - 1.0) / 0.8);
                hip = 0.45f + 0.2f * k; torso = 8.0f + 22.0f * k; aspect = 0.45f + 0.9f * k;
            } else { hip = 0.65f; torso = 30.0f; aspect = 1.35f; }
            auto o = frame(ts, hip, torso, aspect);
            const bool positive = ts >= 1.2;
            o.temporal_available = true; o.temporal_positive = positive;
            o.temporal_probability = positive ? 0.99f : 0.1f;
            events += d.update(o).fall_event ? 1 : 0;
        }
        assert(events == 0);
    }
    // 3. Real fall (fast hip drop + horizontal + lying + temporal positive):
    // exactly one event, reached through Suspected.
    {
        FallDetector d({});
        auto outs = run(d, fallFrames(1.0, 150), 1.6);
        assert(eventCount(outs) == 1);
        for (size_t i = 0; i < outs.size(); ++i) {
            if (outs[i].fall_event) {
                assert(outs[i].event_id == 1);
                assert(i > 0 && outs[i - 1].state == FallState::Suspected);
            }
        }
    }
    // 4. Occluded fall: pose invalid during the drop so the measured hip
    // speed stays below threshold; displacement arming still alarms once.
    {
        FallDetector d({});
        auto outs = run(d, fallFrames(1.0, 150, 0.0, 0.40f, 0.60f, -1.0, 1.0, 2.0), 2.4);
        float max_speed = 0.0f;
        for (const auto& o : outs) max_speed = std::max(max_speed, o.diagnostics.hip_drop_speed);
        assert(max_speed < 0.25f);
        assert(eventCount(outs) == 1);
    }
    // 5a. Late temporal positive after suspected_timeout while still lying.
    {
        FallDetector d({});
        auto outs = run(d, fallFrames(1.0, 150), 4.0);
        assert(eventCount(outs) == 1);
    }
    // 5b. The person stands up before the late positive: no event.
    {
        FallDetector d({});
        auto outs = run(d, fallFrames(1.0, 150, 0.4, 0.45f, 0.75f, 3.0), 4.0);
        assert(eventCount(outs) == 0);
        assert(outs.back().state == FallState::Normal);
    }
    // 5c. The late-confirmation latch is bounded: a positive long after the
    // arming drop finds the candidate back in Normal.
    {
        FallDetector d({});
        auto outs = run(d, fallFrames(1.0, 200), 9.0);
        assert(eventCount(outs) == 0);
        assert(outs.back().state == FallState::Normal);
    }
    // 6. min_suspected_features = 3: the latched arming motion counts, so a
    // victim lying still (current hip speed ~0) still confirms.
    {
        FallConfig cfg;
        cfg.min_suspected_features = 3;
        FallDetector d(cfg);
        auto outs = run(d, fallFrames(1.0, 150), 1.9);
        assert(eventCount(outs) == 1);
        for (const auto& o : outs) {
            if (o.fall_event) assert(o.diagnostics.hip_drop_speed == 0.0f);
        }
    }
    std::cout << "fall_detector_test passed\n";
}
