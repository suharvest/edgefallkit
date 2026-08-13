#include "result_payload.h"

#include "mini_json.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace jetson_fall {
namespace {

void writeBool(std::ostringstream& output, bool value) { output << (value ? "true" : "false"); }

void writeNumber(std::ostringstream& output, float value) {
    if (!std::isfinite(value)) {
        output << "0";
        return;
    }
    output << std::fixed << std::setprecision(4) << value;
}

std::string stateFor(const FallOutput& output) { return fallStateName(output.state); }

void writeFeatures(std::ostringstream& output, const TrackedPerson& person) {
    const auto& observation = person.observation;
    const auto& diagnostics = person.output.diagnostics;
    output << "{\"hip_y\":";
    writeNumber(output, observation.hip_y);
    output << ",\"person_score\":";
    writeNumber(output, observation.person_score);
    output << ",\"hip_drop_speed\":";
    writeNumber(output, diagnostics.hip_drop_speed);
    output << ",\"hip_drop_distance\":";
    writeNumber(output, diagnostics.hip_drop_distance);
    output << ",\"torso_angle_deg\":";
    writeNumber(output, diagnostics.torso_angle_deg);
    output << ",\"bbox_aspect_ratio\":";
    writeNumber(output, diagnostics.bbox_aspect_ratio);
    output << ",\"evidence_features\":" << diagnostics.evidence_features;
    output << ",\"evidence_score\":";
    writeNumber(output, diagnostics.evidence_score);
    output << ",\"lying_posture\":";
    writeBool(output, diagnostics.lying_posture);
    output << ",\"upright_posture\":";
    writeBool(output, diagnostics.upright_posture);
    output << ",\"in_cooldown\":";
    writeBool(output, diagnostics.in_cooldown);
    output << ",\"temporal_probability\":";
    writeNumber(output, diagnostics.temporal_probability);
    output << ",\"temporal_positive\":";
    writeBool(output, diagnostics.temporal_positive);
    output << ",\"suspected_for_sec\":" << std::fixed << std::setprecision(3)
           << diagnostics.suspected_for_sec;
    output << ",\"recovery_for_sec\":" << std::fixed << std::setprecision(3)
           << diagnostics.recovery_for_sec << '}';
}

void writePerson(std::ostringstream& output, const TrackedPerson& person) {
    output << "{\"track_id\":" << person.track_id
           << ",\"person_detected\":";
    writeBool(output, person.observation.valid);
    output << ",\"person_score\":";
    writeNumber(output, person.score);
    output << ",\"fall_detected\":";
    writeBool(output, person.output.fall_detected);
    output << ",\"fall_event\":";
    writeBool(output, person.output.fall_event);
    output << ",\"event_id\":" << person.output.event_id
           << ",\"state\":\"" << stateFor(person.output) << "\",\"tracking\":";
    writeBool(output, person.observation.valid);
    output << ",\"features\":";
    writeFeatures(output, person);
    output << ",\"bbox\":[";
    writeNumber(output, person.box.x);
    output << ',';
    writeNumber(output, person.box.y);
    output << ',';
    writeNumber(output, person.box.w);
    output << ',';
    writeNumber(output, person.box.h);
    output << "]";
    if (!person.pose.empty()) {
        output << ",\"pose17\":[";
        for (int i = 0; i < static_cast<int>(Joint::Count); ++i) {
            if (i != 0) output << ',';
            const auto joint = static_cast<Joint>(i);
            const auto point = person.pose.at(joint);
            output << '[';
            writeNumber(output, point.x);
            output << ',';
            writeNumber(output, point.y);
            output << ',';
            writeNumber(output, person.pose.confidence(joint));
            output << ']';
        }
        output << ']';
    }
    output << '}';
}

}  // namespace

std::string buildResultJson(const StreamPayload& payload) {
    int fallen_count = 0;
    bool fall_event = false;
    std::uint64_t event_id = 0;
    FallState aggregate_state = FallState::Normal;
    const TrackedPerson* primary = nullptr;
    for (const auto* person : payload.persons) {
        if (person == nullptr) continue;
        if (primary == nullptr || person->score > primary->score) primary = person;
        if (person->output.fall_detected) ++fallen_count;
        fall_event = fall_event || person->output.fall_event;
        event_id = std::max(event_id, person->output.event_id);
        if (person->output.state == FallState::Fallen) aggregate_state = FallState::Fallen;
        else if (aggregate_state == FallState::Normal && person->output.state == FallState::Recovering) aggregate_state = FallState::Recovering;
        else if (aggregate_state == FallState::Normal && person->output.state == FallState::Suspected) aggregate_state = FallState::Suspected;
    }

    std::ostringstream output;
    output << "{\"timestamp\":" << payload.timestamp_ms
           << ",\"frame_id\":" << payload.frame_id
           << ",\"inference_time_ms\":";
    writeNumber(output, payload.inference_time_ms);
    output << ",\"stream_id\":\"" << jsonEscape(payload.stream_id)
           << "\",\"fall_detected\":";
    writeBool(output, fallen_count > 0);
    output << ",\"fall_event\":";
    writeBool(output, fall_event);
    output << ",\"event_id\":" << event_id
           << ",\"state\":\"" << fallStateName(aggregate_state)
           << "\",\"person_detected\":";
    writeBool(output, !payload.persons.empty());
    output << ",\"person_count\":" << payload.persons.size()
           << ",\"fallen_count\":" << fallen_count
           << ",\"tracking\":";
    writeBool(output, !payload.persons.empty());
    output << ",\"persons\":[";
    bool first = true;
    for (const auto* person : payload.persons) {
        if (person == nullptr) continue;
        if (!first) output << ',';
        first = false;
        writePerson(output, *person);
    }
    output << ']';
    // Compatibility consumers expect a top-level features object.  Use the
    // highest-confidence person, while the complete per-person values live in
    // persons[].
    if (primary != nullptr) {
        output << ",\"features\":";
        writeFeatures(output, *primary);
    }
    output << '}';
    return output.str();
}

}  // namespace jetson_fall
