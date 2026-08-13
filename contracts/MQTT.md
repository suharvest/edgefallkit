# MQTT result contract v1

Every runtime publishes the same core JSON shape defined by
[`mqtt-result.schema.json`](mqtt-result.schema.json). Platform-specific fields
may be added, but the required fields, units and meanings must not change.

## Semantics

- `timestamp`: Unix epoch milliseconds. Never monotonic seconds or a float.
- `stream_id`: stable configured source ID. A multi-stream process maintains
  independent trackers, temporal windows and event counters per stream.
- `inference_time_ms`: accelerator inference call only where the API exposes
  it. If a backend can measure only a larger region, document that fact and
  publish the larger measurement separately as `pipeline_ms`.
- `event_id` and `global_event_id`: the same monotonically increasing,
  stream-local event sequence. `event_id_scope` is always
  `stream_global_event_id`. A person's own `persons[].event_id` remains a
  per-track counter for compatibility.
- `fall_event`: a one-message edge. `fall_detected` remains true while any
  retained track is `fallen` or `recovering`.
- `person_count`: currently visible people only.
- `fallen_count`: retained `fallen`/`recovering` tracks, including a short
  configured occlusion gap. This deliberately need not be less than or equal
  to `person_count` during occlusion.
- `person_detected`: `person_count > 0`. `tracking` indicates that at least one
  visible or temporarily retained track exists.
- `state`: the most severe retained state in the order
  `fallen > recovering > suspected > normal`.
- `bbox`: normalized `[center_x, center_y, width, height]`.
- `pose17`: COCO-17 normalized `[x,y,confidence]`; empty when unavailable.
  The optional display-oriented `keypoints` field stays an array for reCamera
  compatibility and may be empty on backends without an OSD representation.
- `features.valid=false` must never originate a new fall event. Missing pose
  may retain or expire state only.

## Topic and broker

Broker host, port, credentials, TLS, QoS/retain policy and topic are runtime
configuration. Multi-stream deployments should use a topic template such as
`fall-detection/{stream_id}/results`; `stream_id` is still required in the
payload so consumers do not depend on topic parsing.

## Conformance

Each platform must keep a representative payload fixture and validate it
against the schema in CI or its host-only tests. A benchmark is not complete
until its MQTT fixture passes this contract.

For dependency-free host/target checks, run:

```bash
python3 contracts/validate_payload.py path/to/payload.json
```
