"""RTSP input backends with Rockchip MPP/RGA and OpenCV/FFmpeg fallback."""
from __future__ import annotations

import os
import threading
import time

try:
    import cv2
except ImportError:  # The strict DMA-BUF backend never maps through OpenCV.
    cv2 = None
import numpy as np


_GST_MAP_LOCK = threading.Lock()


class NativeFrame:
    """A borrowed GStreamer DMABUF sample plus immutable capture metadata.

    The sample is deliberately retained until ``close``: the fd belongs to
    the GstBuffer allocator and may be recycled as soon as the sample dies.
    ``close``/``release`` are idempotent so queue drop and error paths can both
    release ownership safely.
    """
    model_width = 640
    model_height = 640

    def __init__(self, sample, *, visible_width, visible_height, visible_rect,
                 pts=None, dts=None, duration=None, colorimetry=None, range=None,
                 planes=(), canvas_width=640, canvas_height=640):
        self.sample = sample
        self.canvas_width = int(canvas_width); self.canvas_height = int(canvas_height)
        self.visible_width = int(visible_width); self.visible_height = int(visible_height)
        self.visible_rect = tuple(int(x) for x in visible_rect)
        self.pts, self.dts, self.duration = pts, dts, duration
        self.colorimetry, self.range = colorimetry, range
        self.planes = tuple(dict(p) for p in planes)
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self):
        with self._lock: return self._closed

    def close(self):
        with self._lock:
            if self._closed: return
            self._closed = True
            sample, self.sample = self.sample, None
        # Dropping the final reference outside the lock avoids allocator
        # callbacks re-entering user code while the state mutex is held.
        del sample

    release = close


def frame_canvas(frame):
    """Return model canvas (W,H) for either NativeFrame or ndarray."""
    if isinstance(frame, NativeFrame):
        return frame.canvas_width, frame.canvas_height
    shape = getattr(frame, "shape", ())
    if len(shape) < 2: raise ValueError("frame has no HxW shape")
    return int(shape[1]), int(shape[0])


def aspect_fit_geometry(source_w: int, source_h: int, size: int = 640):
    """Return the exact resize/pad geometry used by the offline extractor."""
    if source_w <= 0 or source_h <= 0 or size <= 0:
        raise ValueError("source dimensions and size must be positive")
    scale = min(size / source_w, size / source_h)
    scaled_w = int(round(source_w * scale))
    scaled_h = int(round(source_h * scale))
    return scaled_w, scaled_h, (size - scaled_w) // 2, (size - scaled_h) // 2


def pad_scaled_rgb(scaled: np.ndarray, size: int = 640, color: int = 114):
    """Copy an already aspect-fitted RGB image into a square letterbox canvas."""
    if scaled.dtype != np.uint8 or scaled.ndim != 3 or scaled.shape[2] != 3:
        raise ValueError("scaled image must be HWC uint8 RGB")
    height, width = scaled.shape[:2]
    if width > size or height > size:
        raise ValueError("scaled image does not fit the letterbox canvas")
    left, top = (size - width) // 2, (size - height) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[top:top + height, left:left + width] = scaled
    return canvas


def copy_strided_rgb_to_letterbox(data, width: int, height: int, stride: int,
                                  size: int = 640, color: int = 114):
    """Copy mapped, potentially aligned RGB rows into an owned letterbox."""
    if width <= 0 or height <= 0 or stride < width * 3:
        raise ValueError("invalid mapped RGB dimensions or stride")
    if len(data) < stride * height:
        raise ValueError("mapped RGB buffer is shorter than its negotiated layout")
    borrowed = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data,
                          strides=(stride, 3, 1))
    return pad_scaled_rgb(borrowed, size, color)


def copy_strided_nv12_to_letterbox(data, width: int, height: int,
                                   y_stride: int, uv_stride: int,
                                   y_offset: int, uv_offset: int,
                                   size: int = 640, color: int = 114):
    """Convert mapped, strided NV12 into an owned RGB letterbox image."""
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("NV12 dimensions must be positive and even")
    if y_stride < width or uv_stride < width or y_offset < 0 or uv_offset < 0:
        raise ValueError("invalid mapped NV12 strides or offsets")
    required = max(y_offset + y_stride * height,
                   uv_offset + uv_stride * (height // 2))
    if len(data) < required:
        raise ValueError("mapped NV12 buffer is shorter than its negotiated layout")
    y_plane = np.ndarray((height, width), dtype=np.uint8, buffer=data,
                         offset=y_offset, strides=(y_stride, 1))
    uv_plane = np.ndarray((height // 2, width // 2, 2), dtype=np.uint8,
                          buffer=data, offset=uv_offset,
                          strides=(uv_stride, 2, 1))
    rgb = cv2.cvtColorTwoPlane(y_plane, uv_plane, cv2.COLOR_YUV2RGB_NV12)
    scaled_w, scaled_h, _, _ = aspect_fit_geometry(width, height, size)
    if (scaled_w, scaled_h) != (width, height):
        rgb = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    return pad_scaled_rgb(rgb, size, color)


class FFmpegRTSP:
    backend_name = "opencv_ffmpeg"

    def __init__(self, url: str, size: int = 640, transport: str = "tcp", **_):
        self.url = url
        self.size = size
        self.transport = transport
        self.capture = None

    def start(self):
        if cv2 is None:
            raise RuntimeError("OpenCV/FFmpeg backend is unavailable")
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self.transport}")
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"OpenCV/FFmpeg could not open {self.url}")

    def read(self):
        if self.capture is None or not self.capture.isOpened():
            self.start()
        ok, bgr = self.capture.read()
        if not ok:
            self.close()
            return None
        h, w = bgr.shape[:2]
        nw, nh, left, top = aspect_fit_geometry(w, h, self.size)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class GStreamerMPP:
    """GStreamer RTSP -> Rockchip MPP decoder -> RGB model input."""

    backend_name = "gstreamer_mpp"

    def __init__(self, url: str, size: int = 640, transport: str = "tcp",
                 codec: str = "h264", latency_ms: int = 100,
                 appsink_timeout_ms: int = 2000, appsink_queue: int = 3,
                 mpp_output_format: str = "rgb", **_):
        self.url = url
        self.size = size
        self.transport = transport
        self.codec = codec.lower()
        self.latency_ms = latency_ms
        self.timeout_ns = int(appsink_timeout_ms) * 1_000_000
        self.appsink_queue = max(1, int(appsink_queue))
        self.output_format = str(mpp_output_format).lower()
        if self.output_format not in ("rgb", "nv12_cpu", "dma_nv12"):
            raise ValueError("mpp_output_format must be rgb, nv12_cpu, or dma_nv12")
        self.pipeline = self.sink = self.bus = None
        self.Gst = None
        self.GstVideo = None
        self._scaled_size = None
        self._state_lock = threading.RLock()
        self._closed = False

    def _make(self, factory, name):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    def start(self):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtsp", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstRtsp, GstVideo
        except Exception as exc:
            raise RuntimeError("PyGObject GStreamer bindings are unavailable") from exc
        if self.codec not in ("h264", "h265"):
            raise ValueError(f"unsupported RTSP codec for MPP: {self.codec}")
        Gst.init(None)
        self.Gst = Gst
        self.GstVideo = GstVideo
        pipeline = Gst.Pipeline.new("rknn-rtsp")
        source = self._make("rtspsrc", "source")
        depay = self._make(f"rtp{self.codec}depay", "depay")
        parser = self._make(f"{self.codec}parse", "parser")
        decoder = self._make("mppvideodec", "decoder")
        capsfilter = self._make("capsfilter", "rgb-caps")
        sink = self._make("appsink", "sink")
        source.set_property("location", self.url)
        source.set_property("latency", int(self.latency_ms))
        source.set_property("drop-on-latency", True)
        source.set_property("protocols", GstRtsp.RTSPLowerTrans.TCP if self.transport == "tcp"
                            else GstRtsp.RTSPLowerTrans.UDP)
        decoder.set_property("fast-mode", True)
        if self.output_format == "dma_nv12":
            # Rockchip plugin versions expose this as a boolean property;
            # failure is intentional because CPU-backed NV12 violates the
            # native contract.
            try:
                decoder.set_property("dma-feature", True)
            except Exception as exc:
                raise RuntimeError("mppvideodec lacks dma-feature") from exc
        # The parser's first CAPS event supplies coded source dimensions.  Its
        # probe below sets an aspect-fitted MPP/RGA output size before decoder
        # negotiation.  Leaving zero here preserves the source if CAPS lacks
        # dimensions, which is safe and handled by the read-side fallback.
        decoder.set_property("width", 0)
        decoder.set_property("height", 0)
        decoder.set_property("format", 23 if self.output_format in ("nv12_cpu", "dma_nv12") else 15)
        capsfilter.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:DMABuf),format=NV12" if self.output_format == "dma_nv12"
            else ("video/x-raw,format=NV12" if self.output_format == "nv12_cpu" else "video/x-raw,format=RGB")))
        sink.set_property("sync", False)
        # Depth 1 means read() can only ever hand back the frame that arrives
        # next: finish early and it waits, finish late and that frame was
        # already dropped, so any jitter costs a whole frame period. A small
        # queue absorbs that at the cost of at most a couple of frames of
        # latency. drop=True still keeps the reader on recent frames.
        sink.set_property("max-buffers", int(self.appsink_queue))
        sink.set_property("drop", True)
        sink.set_property("emit-signals", False)
        for element in (source, depay, parser, decoder, capsfilter, sink):
            pipeline.add(element)
        if not depay.link(parser) or not parser.link(decoder) or not decoder.link(capsfilter) or not capsfilter.link(sink):
            raise RuntimeError("failed to link MPP GStreamer pipeline")

        def on_parser_event(_pad, probe_info):
            event = probe_info.get_event()
            if event is None or event.type != Gst.EventType.CAPS:
                return Gst.PadProbeReturn.OK
            caps = event.parse_caps()
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if structure is None:
                return Gst.PadProbeReturn.OK
            ok_w, source_w = structure.get_int("width")
            ok_h, source_h = structure.get_int("height")
            if (self.output_format == "rgb" and ok_w and ok_h and
                    source_w > 0 and source_h > 0):
                scaled_w, scaled_h, _, _ = aspect_fit_geometry(
                    source_w, source_h, self.size)
                decoder.set_property("width", scaled_w)
                decoder.set_property("height", scaled_h)
                self._scaled_size = (scaled_w, scaled_h)
            return Gst.PadProbeReturn.OK

        parser.get_static_pad("src").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM, on_parser_event)

        expected_encoding = self.codec.upper()
        def on_pad_added(_source, pad):
            caps = pad.get_current_caps() or pad.query_caps(None)
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if structure and structure.get_string("media") == "video" and structure.get_string("encoding-name") == expected_encoding:
                pad.link(depay.get_static_pad("sink"))

        source.connect("pad-added", on_pad_added)
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer MPP pipeline refused PLAYING state")
        with self._state_lock:
            if self._closed:
                pipeline.set_state(Gst.State.NULL)
                return
            self.pipeline, self.sink, self.bus = pipeline, sink, pipeline.get_bus()

    def _bus_error(self):
        if self.bus is None:
            return None
        message = self.bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if message is None:
            return None
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            return RuntimeError(f"GStreamer MPP error: {error}; {debug or ''}")
        return RuntimeError("GStreamer MPP stream reached EOS")

    def read(self):
        with self._state_lock:
            if self._closed: return None
            if self.pipeline is None: self.start()
            sink = self.sink; timeout_ns = self.timeout_ns
        if sink is None: return None
        error = self._bus_error()
        if error:
            self.close()
            raise error
        sample = sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            error = self._bus_error()
            if error:
                self.close()
                raise error
            # A pull that simply timed out is not fatal. Tearing the pipeline
            # down here meant the next read() rebuilt the RTSP connection from
            # scratch, which cannot finish inside one appsink timeout either --
            # so MPP never survived its own startup and every deployment fell
            # back to CPU decode.
            return None
        sample_caps = sample.get_caps()
        caps = sample_caps.get_structure(0)
        width, height = int(caps.get_value("width")), int(caps.get_value("height"))
        buffer = sample.get_buffer()
        if self.output_format == "dma_nv12":
            return self._native_frame(sample, sample_caps, caps, buffer, width, height)
        # PyGObject's MapInfo property access is not reliable when several
        # appsinks map concurrently. Keep the borrowed-memory window short and
        # process-local; the returned frame always owns its storage.
        with _GST_MAP_LOCK:
            ok, mapped = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("failed to map GStreamer appsink buffer")
            try:
                video_info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
                mapped_data = mapped.data
                if self.output_format == "nv12_cpu":
                    frame = copy_strided_nv12_to_letterbox(
                        mapped_data, width, height,
                        int(video_info.stride[0]), int(video_info.stride[1]),
                        int(video_info.offset[0]), int(video_info.offset[1]),
                        self.size)
                else:
                    # Respect aligned RGB rows. The decoder has already used
                    # RGA for aspect-fit unless source CAPS lacked dimensions.
                    stride = int(video_info.stride[0])
                    if stride < width * 3 or len(mapped_data) < stride * height:
                        raise RuntimeError("invalid RGB stride from mppvideodec")
                    mapped_rgb = np.ndarray((height, width, 3), dtype=np.uint8,
                                            buffer=mapped_data,
                                            strides=(stride, 3, 1))
                    if width <= self.size and height <= self.size:
                        frame = copy_strided_rgb_to_letterbox(
                            mapped_data, width, height, stride, self.size)
                    else:
                        if cv2 is None:
                            raise RuntimeError("OpenCV resize fallback is unavailable")
                        scaled_w, scaled_h, _, _ = aspect_fit_geometry(
                            width, height, self.size)
                        scaled = cv2.resize(mapped_rgb, (scaled_w, scaled_h),
                                            interpolation=cv2.INTER_LINEAR)
                        frame = pad_scaled_rgb(scaled, self.size)
            finally:
                buffer.unmap(mapped)
        return frame

    def _native_frame(self, sample, sample_caps, caps, buffer, width, height):
        """Validate and retain a single DMABUF NV12 sample without mapping it."""
        features = sample_caps.get_features(0)
        if features is None or not features.contains("memory:DMABuf"):
            sample_ref = None; del sample_ref
            raise RuntimeError("dma_nv12 negotiated without memory:DMABuf")
        if str(caps.get_value("format")) != "NV12" or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise RuntimeError("dma_nv12 requires even NV12 dimensions")
        if buffer.n_memory() != 1:
            raise RuntimeError("dma_nv12 supports exactly one GstMemory")
        info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
        if info is None or int(info.finfo.n_planes) != 2:
            raise RuntimeError("invalid NV12 VideoInfo")
        planes = []
        try:
            import gi
            gi.require_version("GstAllocators", "1.0")
            from gi.repository import GstAllocators
            memory = buffer.peek_memory(0)
            if not GstAllocators.is_dmabuf_memory(memory):
                raise RuntimeError("GstMemory is not DMA-BUF")
            fd = int(GstAllocators.dmabuf_memory_get_fd(memory))
        except Exception as exc:
            raise RuntimeError("GstAllocators DMABUF fd extraction unavailable") from exc
        for index in (0, 1):
            offset = int(info.offset[index]); stride = int(info.stride[index])
            plane_h = height if index == 0 else height // 2
            if offset < 0 or stride < width or offset + stride * plane_h <= offset:
                raise RuntimeError("invalid NV12 plane offset/stride")
            planes.append({"fd": fd, "offset": offset, "stride": stride,
                           "width": width, "height": plane_h})
        colorimetry = caps.get_string("colorimetry") if caps.has_field("colorimetry") else None
        rng = caps.get_string("range") if caps.has_field("range") else None
        return NativeFrame(sample, visible_width=width, visible_height=height,
                           visible_rect=(0, 0, width, height), pts=buffer.pts,
                           dts=buffer.dts, duration=buffer.duration,
                           colorimetry=colorimetry, range=rng, planes=planes,
                           canvas_width=self.size, canvas_height=self.size)

    def close(self):
        with self._state_lock:
            self._closed = True
            pipeline, gst = self.pipeline, self.Gst
            self.pipeline = self.sink = self.bus = None
            self._scaled_size = None
        if pipeline is not None and gst is not None:
            pipeline.set_state(gst.State.NULL)


class FallbackVideoSource:
    def __init__(self, primary, fallback=None, strict=False, failure_limit=3):
        self.primary = primary
        self.fallback = fallback
        self.strict = strict
        self.failure_limit = max(1, int(failure_limit))
        self.active = primary
        self.failures = 0

    @property
    def active_backend(self):
        return self.active.backend_name

    def _switch(self, exc):
        if self.strict or self.fallback is None or self.active is self.fallback:
            if exc:
                raise exc
            return
        self.active.close()
        self.active = self.fallback
        self.failures = 0
        print(f"[video] primary failed, switching to {self.active_backend}: {exc}", flush=True)

    def read(self):
        try:
            frame = self.active.read()
        except Exception as exc:
            self._switch(exc)
            return None
        if frame is not None:
            self.failures = 0
            return frame
        self.failures += 1
        if self.failures >= self.failure_limit:
            self._switch(RuntimeError(f"{self.active_backend} returned no frame {self.failures} times"))
        return None

    def close(self):
        self.active.close()
        if self.fallback is not None and self.fallback is not self.active:
            self.fallback.close()


def create_video_source(stream, config, size):
    video = dict(config.get("video", {}))
    video.update(stream.get("video", {}))
    backend = str(video.get("backend", "gstreamer_mpp"))
    strict = bool(video.get("strict", False))
    fallback_name = str(video.get("fallback", "opencv_ffmpeg"))
    if backend not in ("gstreamer_mpp", "opencv_ffmpeg"):
        raise ValueError(f"unsupported video backend: {backend}")
    if fallback_name not in ("none", "opencv_ffmpeg"):
        raise ValueError(f"unsupported video fallback: {fallback_name}")
    kwargs = dict(url=stream["rtsp_url"], size=size,
                  transport=stream.get("transport", "tcp"), **video)
    kwargs.pop("backend", None); kwargs.pop("strict", None); kwargs.pop("fallback", None)
    primary = GStreamerMPP(**kwargs) if backend == "gstreamer_mpp" else FFmpegRTSP(**kwargs)
    fallback = None
    if backend != "opencv_ffmpeg" and fallback_name == "opencv_ffmpeg":
        fallback = FFmpegRTSP(**kwargs)
    return FallbackVideoSource(primary, fallback, strict,
                               int(video.get("failure_limit", 3)))


def validate_backend_config(config):
    """Constructors are intentionally not started during config validation."""
    dummy = {"id": "validate", "rtsp_url": "rtsp://127.0.0.1/validate"}
    source = create_video_source(dummy, config, int(config.get("input_size", 640)))
    source.close()
    from rknn_pose import PoseDecoder
    post = config.get("postprocess", {})
    if str(post.get("backend", "cpp")) not in ("auto", "cpp", "numpy"):
        raise ValueError(f"unsupported postprocess backend: {post.get('backend')}")
    if str(post.get("fallback", "numpy")) not in ("none", "numpy"):
        raise ValueError(f"unsupported postprocess fallback: {post.get('fallback')}")
    rknn = config.get("rknn", {})
    rknn_backend = str(rknn.get("backend", "legacy"))
    if rknn_backend not in ("legacy", "auto", "native"):
        raise ValueError(f"unsupported rknn backend: {rknn_backend}")
    base_video = dict(config.get("video", {}))
    video_configs = [base_video]
    for stream in config.get("streams", []):
        merged = dict(base_video); merged.update(stream.get("video", {}))
        video_configs.append(merged)
    for video in video_configs:
        if str(video.get("mpp_output_format", "rgb")).lower() == "dma_nv12":
            if (rknn_backend != "native" or not bool(rknn.get("strict", False)) or
                    not bool(video.get("strict", False)) or
                    str(video.get("fallback", "opencv_ffmpeg")) != "none"):
                raise ValueError("dma_nv12 requires strict native RKNN and no video fallback")
