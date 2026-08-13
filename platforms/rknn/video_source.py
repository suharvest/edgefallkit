"""RTSP input backends with Rockchip MPP/RGA and OpenCV/FFmpeg fallback."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np


class FFmpegRTSP:
    backend_name = "opencv_ffmpeg"

    def __init__(self, url: str, size: int = 640, transport: str = "tcp", **_):
        self.url = url
        self.size = size
        self.transport = transport
        self.capture = None

    def start(self):
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
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class GStreamerMPP:
    """GStreamer RTSP -> Rockchip MPP decoder (linked to librga) -> RGB appsink."""

    backend_name = "gstreamer_mpp"

    def __init__(self, url: str, size: int = 640, transport: str = "tcp",
                 codec: str = "h264", latency_ms: int = 100,
                 appsink_timeout_ms: int = 2000, **_):
        self.url = url
        self.size = size
        self.transport = transport
        self.codec = codec.lower()
        self.latency_ms = latency_ms
        self.timeout_ns = int(appsink_timeout_ms) * 1_000_000
        self.pipeline = self.sink = self.bus = None
        self.Gst = None

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
            from gi.repository import Gst, GstRtsp
        except Exception as exc:
            raise RuntimeError("PyGObject GStreamer bindings are unavailable") from exc
        if self.codec not in ("h264", "h265"):
            raise ValueError(f"unsupported RTSP codec for MPP: {self.codec}")
        Gst.init(None)
        self.Gst = Gst
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
        decoder.set_property("width", self.size)
        decoder.set_property("height", self.size)
        decoder.set_property("format", 15)  # GstMppVideoDecFormat RGB, verified on both boards.
        capsfilter.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self.size},height={self.size}"))
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.set_property("emit-signals", False)
        for element in (source, depay, parser, decoder, capsfilter, sink):
            pipeline.add(element)
        if not depay.link(parser) or not parser.link(decoder) or not decoder.link(capsfilter) or not capsfilter.link(sink):
            raise RuntimeError("failed to link MPP GStreamer pipeline")

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
        if self.pipeline is None:
            self.start()
        error = self._bus_error()
        if error:
            self.close()
            raise error
        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:
            error = self._bus_error()
            self.close()
            if error:
                raise error
            return None
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map GStreamer appsink buffer")
        try:
            expected = width * height * 3
            frame = np.frombuffer(mapped.data, dtype=np.uint8, count=expected).reshape(height, width, 3).copy()
        finally:
            buffer.unmap(mapped)
        return frame

    def close(self):
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        self.pipeline = self.sink = self.bus = None


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
