# Python control-plane audit (HailoRT 4.21)

## Decision

Keep the verified native C++ worker as the production hot path for now. A
Python control plane can safely own configuration, process supervision and
MQTT lifecycle, but direct Python access to `hailonet` tensor metadata is not
deployable from the packages currently installed on `harvest-pi`. Do not copy
or reinterpret the private meta layout with `ctypes`.

The fair cross-platform target is:

```text
Python config/supervision
  -> GStreamer RTSP + hardware decode/convert + hailonet
  -> small ABI-locked C++ Gst element (tensor decode + tracker + temporal)
  -> Python appsink (JSON/protobuf only) -> common MQTT publisher
```

This preserves GStreamer/Hailo as the hot path. Python sees one small result
buffer per video frame, not tensor planes, so it does not add copies or loops
to inference/postprocess. The native executable remains the fallback until the
plugin variant passes the same Spark stream and MQTT contract benchmark.

## Verified host capability

Read-only audit on Fleet `harvest-pi` on 2026-08-13:

- Python 3.13.5, PyGObject 3.50.0 and GStreamer GI 1.26.2 work.
- `hailonet` loads from `libgsthailo.so`; HailoRT/driver/firmware are 4.21.0.
- `hailo_platform` Python bindings are absent.
- Hailo/TAPPAS GIR or typelib metadata is absent.
- `pybind11` and `pybind11-dev` are absent; `libgirepository-2.0-dev` is absent.
- `python3-gst-1.0` is not installed, although the base GI Gst import works.
- TAPPAS is not installed. `hailo-tappas-core` 5.1.0 is merely available from
  the configured package source and must not be assumed ABI-compatible with
  the deployed HailoRT 4.21 plugin.

`hailonet` returns nine output planes as `GstParentBufferMeta`; each parent
buffer contains `GstHailoTensorMeta`. The API is a C++ header/macros
(`gst_tensor_meta_api_get_type`, `GST_TENSOR_META_GET`) with no introspection
typelib. PyGObject/appsink can receive the outer `Gst.Buffer`, but cannot
reliably decode `GstHailoTensorMeta::info`. A direct Python hailonet+appsink
implementation therefore stops at the tensor ABI boundary.

## Migration plan

1. Build an ABI-locked `hailoposejson` C++ GStreamer element in the existing
   builder image. It consumes the parent/tensor metadata using the same Hailo
   4.21 headers, calls the existing decoder/tracker/temporal code, and emits a
   compact result buffer. A GStreamer element is preferred to a pybind bridge:
   it avoids depending on CPython 3.13 and PyGObject pointer internals.
2. Add a dependency-light Python service using only PyGObject and a small MQTT
   client. It creates one pipeline per `stream_id`, receives result buffers via
   appsink, validates the common schema and publishes unchanged topics.
3. Keep `fall-hailo` in the image as `CONTROL_PLANE=native` fallback. Enable
   Python only as an explicit compose profile until parity is proved.
4. In a newly authorized Hailo-exclusive window, compare native and Python
   control planes using the same Spark 640x640@15 stream, fixed duration,
   warm-up, score thresholds and MQTT broker. Record app FPS, hailonet probe
   P95, CPU, RSS, temperature, output coverage and contract failures. Accept
   Python only if FPS is within 2%, probe P95 within 5%, MQTT is 100% valid and
   no extra frame gaps appear.
5. Build the Python variant as a separate multi-stage image. Runtime packages
   should be limited to `python3-minimal`, `python3-gi`, the required Gst GI
   package, GStreamer runtime plugins and MQTT client; continue bind-mounting
   HEF and HailoRT 4.21 libraries. Never install Torch, Ultralytics, NumPy,
   scikit-learn or TAPPAS in the runtime image.

No Python performance number is reported yet: the missing typed tensor bridge
and absence of a fresh Hailo maintenance window make such a claim unverifiable.
The native C++ result remains the accepted Hailo measurement baseline.
