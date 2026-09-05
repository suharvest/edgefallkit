import json
import subprocess
import sys
import threading
import time
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import app as app_module
import native_rknn
from app import (CapturedFrame, ContextPoolRuntime, LatestFrameQueue,
                 MqttPublisher, StreamWorker, close_publishers,
                 make_stream_publishers, resolve_context_core_masks,
                 resolve_legacy_core_mask)
from video_source import NativeFrame, frame_canvas, validate_backend_config


class ContextPoolTest(unittest.TestCase):
    def test_legacy_worker_records_backend_failure_and_closes_resources(self):
        closed = []
        class Model:
            def close(self): closed.append("model")
        class Source:
            active_backend = "gstreamer_mpp"
            def read(self): raise RuntimeError("strict backend failed")
            def close(self): closed.append("source")
        class Decoder:
            active_backend = "cpp"
        worker = StreamWorker(
            {"model_path": "/models/test.rknn", "tracker": {}, "postprocess": {}},
            {"id": "a", "rtsp_url": "rtsp://example.invalid/test"}, object())
        old_running = app_module.RUNNING
        app_module.RUNNING = True
        try:
            with patch.object(app_module, "RKNNPose", return_value=Model()), \
                 patch.object(app_module, "create_video_source", return_value=Source()), \
                 patch.object(app_module, "PoseDecoder", return_value=Decoder()):
                worker.run()
            self.assertIsInstance(worker.exception, RuntimeError)
            self.assertIn("strict backend failed", str(worker.exception))
            self.assertEqual(closed, ["source", "model"])
            self.assertFalse(app_module.RUNNING)
        finally:
            app_module.RUNNING = old_running

    def test_validate_rejects_context_masks_in_legacy_mode(self):
        config = {
            "model_path": "/models/test.rknn",
            "mqtt": {"topic": "test/{stream_id}"},
            "video": {"backend": "opencv_ffmpeg", "strict": False},
            "postprocess": {"backend": "numpy", "strict": False},
            "inference_mode": "legacy",
            "context_core_masks": [3],
            "streams": [{"id": "a", "rtsp_url": "rtsp://example.invalid/test"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            process = subprocess.run(
                [sys.executable, str(Path(__file__).parents[1] / "app.py"),
                 "--config", str(path), "--validate"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("legacy does not accept context_core_masks", process.stderr)

    def test_context_core_masks_resolve_and_preserve_scalar_compatibility(self):
        self.assertEqual(resolve_context_core_masks({"context_core_masks": [1, 2]}, 2), [1, 2])
        self.assertEqual(resolve_context_core_masks({"core_mask": "0x3"}, 2), [3, 3])
        self.assertEqual(resolve_context_core_masks({}, 2), [None, None])
        self.assertEqual(resolve_legacy_core_mask({"core_mask": "0x3"}, {"id": "a"}), 3)
        self.assertEqual(resolve_legacy_core_mask({"core_mask": "0x3"}, {"id": "a", "core_mask": 5}), 5)
        self.assertEqual(resolve_legacy_core_mask({}, {"id": "a"}, lambda: b"rockchip,rk3576\x00"), 3)
        self.assertIsNone(resolve_legacy_core_mask({}, {"id": "a"}, lambda: b"rockchip,rk3588\x00"))
        self.assertIsNone(resolve_legacy_core_mask({}, {"id": "a"}, lambda: b"unknown"))
        self.assertIsNone(resolve_legacy_core_mask({"core_policy": "runtime_auto"}, {"id": "a"},
                                                   lambda: b"rockchip,rk3576\x00"))
        self.assertIsNone(resolve_legacy_core_mask({}, {"id": "a"},
                                                   lambda: b"rockchip,rk3576\x00", False))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            resolve_legacy_core_mask({"core_mask": 0}, {"id": "a"})
        with self.assertRaisesRegex(ValueError, "one positive integer"):
            resolve_context_core_masks({"context_core_masks": [1]}, 2)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_context_core_masks({"context_core_masks": [1, 2], "core_mask": 3}, 2)
        with self.assertRaisesRegex(ValueError, "JSON list"):
            resolve_context_core_masks({"context_core_masks": (1, 2)}, 2)

    def test_duplicate_enabled_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ContextPoolRuntime([{"id": "a"}, {"id": "a"}], 1,
                               lambda s: None, lambda _: None, lambda *_: None)
        with self.assertRaisesRegex(ValueError, "at least one"):
            ContextPoolRuntime([], 1, lambda s: None, lambda _: None, lambda *_: None)

    def test_latest_queue_drops_stale_and_preserves_capture_timestamp(self):
        q = LatestFrameQueue()
        first = CapturedFrame("old", 12.5, 1.0, 1)
        newest = CapturedFrame("new", 12.7, 1.2, 3)
        self.assertTrue(q.put(first)); self.assertTrue(q.put(CapturedFrame("mid", 12.6, 1.1, 2)))
        self.assertTrue(q.put(newest))
        self.assertEqual(q.dropped, 2)
        self.assertEqual(q.get().timestamp, newest.timestamp)
        self.assertEqual(q.get(), None)

    def test_latest_queue_releases_dropped_and_closed_native_frames(self):
        class Sample: pass
        first = NativeFrame(Sample(), visible_width=640, visible_height=640,
                            visible_rect=(0, 0, 640, 640))
        second = NativeFrame(Sample(), visible_width=640, visible_height=640,
                             visible_rect=(0, 0, 640, 640))
        q = LatestFrameQueue()
        self.assertTrue(q.put(CapturedFrame(first, 1, 0, 1)))
        self.assertTrue(q.put(CapturedFrame(second, 2, 0, 2)))
        self.assertTrue(first.closed)
        self.assertFalse(second.closed)
        q.close()
        self.assertTrue(second.closed)
        self.assertFalse(q.put(CapturedFrame(first, 3, 0, 3)))
        self.assertEqual(frame_canvas(second), (640, 640))

    def test_mqtt_publisher_is_async_and_closes_after_drain(self):
        sent = []
        with patch.object(MqttPublisher, "_send", side_effect=lambda *item: sent.append(item)):
            publisher = MqttPublisher({"host": "unused", "queue_size": 8})
            publisher.publish("topic/a", {"frame_id": 1})
            publisher.publish("topic/b", {"frame_id": 2})
            publisher.close(timeout=1)
        self.assertEqual(sent, [("topic/a", {"frame_id": 1}),
                                ("topic/b", {"frame_id": 2})])
        self.assertFalse(publisher.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            publisher.publish("topic/c", {})

    def test_mqtt_background_failure_is_reported_on_close(self):
        with patch.object(MqttPublisher, "_send", side_effect=ValueError("encode failed")):
            publisher = MqttPublisher({"host": "unused", "queue_size": 8})
            publisher.publish("topic/a", {"frame_id": 1})
            with self.assertRaisesRegex(RuntimeError, "publisher failed"):
                publisher.close(timeout=1)
        self.assertIsInstance(publisher.exception, ValueError)
        self.assertFalse(publisher.thread.is_alive())

    def test_mqtt_close_excludes_late_publish(self):
        entered = threading.Event(); release = threading.Event(); closed = []
        def send(*_): entered.set(); release.wait(1)
        with patch.object(MqttPublisher, "_send", side_effect=send):
            publisher = MqttPublisher({"host": "unused", "queue_size": 8})
            publisher.publish("topic/a", {"frame_id": 1})
            self.assertTrue(entered.wait(.5))
            closer = threading.Thread(target=lambda: (publisher.close(), closed.append(True)))
            closer.start()
            self.assertTrue(publisher.stop_event.wait(.5))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                publisher.publish("topic/b", {"frame_id": 2})
            release.set(); closer.join(1)
        self.assertEqual(closed, [True])
        self.assertTrue(publisher.outbox.empty())

    def test_stream_publishers_use_independent_client_ids_and_all_close(self):
        created = []
        class Publisher:
            def __init__(self, cfg): self.cfg = cfg; self.closed = False; created.append(self)
            def close(self): self.closed = True
        with patch.object(app_module, "MqttPublisher", Publisher):
            publishers = make_stream_publishers(
                {"host": "broker", "client_id": "edgefall"},
                [{"id": "cam-1"}, {"id": "cam-2"}, {"id": "cam-3"}])
        self.assertEqual([publishers[key].cfg["client_id"] for key in publishers],
                         ["edgefall-cam-1", "edgefall-cam-2", "edgefall-cam-3"])
        self.assertEqual(len({id(value) for value in publishers.values()}), 3)
        close_publishers(publishers)
        self.assertTrue(all(publisher.closed for publisher in created))

    def test_dma_backend_requires_strict_native_pair(self):
        base = {"video": {"backend": "gstreamer_mpp", "mpp_output_format": "dma_nv12",
                           "strict": True, "fallback": "none"},
                "rknn": {"backend": "native", "strict": True}}
        validate_backend_config(base)
        for rknn in ({"backend": "legacy", "strict": True},
                     {"backend": "native", "strict": False},
                     {"backend": "typo", "strict": True}):
            config = dict(base); config["rknn"] = rknn
            with self.assertRaises(ValueError):
                validate_backend_config(config)

    def test_native_runtime_destroys_handle_when_contract_validation_fails(self):
        class Function:
            def __init__(self, result=None): self.result = result
            def __call__(self, *_): return self.result
        class Recorder(Function):
            def __init__(self, output): super().__init__(); self.output = output
            def __call__(self, value): self.output.append(value)
        class Library:
            def __init__(self):
                self.hybrid_last_error = Function(b"bad outputs")
                self.hybrid_create = Function(123)
                self.hybrid_model_width = Function(640)
                self.hybrid_model_height = Function(640)
                self.hybrid_output_count = Function(0)
                self.hybrid_output_ndims = Function(0)
                self.hybrid_output_dim = Function(0)
                self.hybrid_output_elems = Function(0)
                self.hybrid_infer_pose_nv12_fd = Function(0)
                self.destroyed = []
                self.hybrid_destroy = Recorder(self.destroyed)
        library = Library()
        with patch.object(native_rknn.ctypes, "CDLL", return_value=library):
            with self.assertRaisesRegex(RuntimeError, "exactly 9 outputs"):
                native_rknn.NativeRuntime("model.rknn")
        self.assertEqual(library.destroyed, [123])

    def test_round_robin_fairness_and_cleanup(self):
        streams = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        runtime = None
        models = []
        seen = []
        class Source:
            def __init__(self): self.closed = False
            def read(self): return None
            def close(self): self.closed = True
        class Model:
            def close(self): models.append("closed")
        runtime = ContextPoolRuntime(streams, 1, lambda s: Source(), lambda _: Model(),
                                     lambda s, m, item: seen.append(s["id"]))
        for index, stream in enumerate(streams):
            runtime.queues[stream["id"]].put(CapturedFrame(index, 100 + index, 0, index))
        self.assertEqual([runtime._next()[0]["id"] for _ in range(3)], ["a", "b", "c"])
        runtime.start(); time.sleep(.01); runtime.stop()
        self.assertEqual(len(models), 1)

    def test_busy_stream_is_not_dispatched_to_two_contexts_and_backend_is_carried(self):
        streams = [{"id": "a"}, {"id": "b"}]
        runtime = ContextPoolRuntime(streams, 1, lambda s: None, lambda _: None, lambda *_: None)
        runtime.queues["a"].put(CapturedFrame("a", 1, 0, 1, "gstreamer_mpp"))
        runtime.queues["b"].put(CapturedFrame("b", 2, 0, 1, "gstreamer_mpp"))
        stream, item = runtime._next()
        self.assertEqual(stream["id"], "a")
        self.assertEqual(item.source_backend, "gstreamer_mpp")
        stream, item = runtime._next()
        self.assertEqual(stream["id"], "b")
        runtime.queues["a"].put(CapturedFrame("a2", 3, 0, 2, "opencv_ffmpeg"))
        self.assertEqual(runtime._next(), (None, None))
        with runtime._lock:
            runtime._busy.discard("a")
        self.assertEqual(runtime._next()[1].source_backend, "opencv_ffmpeg")

    def test_pool_uses_one_inflight_context_and_stops(self):
        streams = [{"id": "a"}, {"id": "b"}]
        entered = threading.Event(); release = threading.Event(); active = [0]; max_active = [0]
        class Source:
            def read(self): return None
            def close(self): pass
        class Model:
            def close(self): pass
        def handler(stream, model, item):
            active[0] += 1; max_active[0] = max(max_active[0], active[0]); entered.set()
            release.wait(.2); active[0] -= 1
        runtime = ContextPoolRuntime(streams, 1, lambda s: Source(), lambda _: Model(), handler)
        runtime.queues["a"].put(CapturedFrame(1, 1, 0, 1)); runtime.start()
        self.assertTrue(entered.wait(.5)); runtime.stop(); release.set()
        self.assertLessEqual(max_active[0], 1)

    def test_empty_capture_honors_interruptible_reconnect_delay(self):
        reads = [0]
        class Source:
            def read(self): reads[0] += 1; return None
            def close(self): pass
        class Model:
            def close(self): pass
        runtime = ContextPoolRuntime(
            [{"id": "a", "reconnect_delay_ms": 200}], 1,
            lambda _: Source(), lambda _: Model(), lambda *_: None)
        runtime.start()
        time.sleep(.05)
        self.assertEqual(reads[0], 1)
        started = time.monotonic()
        runtime.stop(timeout=.5)
        self.assertLess(time.monotonic() - started, .2)

    def test_blocked_source_stop_is_bounded_and_source_closes_once(self):
        released = threading.Event(); close_count = [0]
        class Source:
            active_backend = "fake"
            def read(self): released.wait(10); return None
            def close(self): close_count[0] += 1; released.set()
        class Model:
            def close(self): pass
        runtime = ContextPoolRuntime([{"id": "a"}], 1, lambda s: Source(), lambda _: Model(), lambda *_: None)
        runtime.start()
        started = time.monotonic(); runtime.stop(timeout=.05); elapsed = time.monotonic() - started
        released.set()
        self.assertLess(elapsed, 1.0)
        # The capture thread is allowed to finish after the bounded join, but
        # cleanup must remain idempotent when the caller stops again.
        runtime.stop(timeout=.05)
        self.assertLessEqual(close_count[0], 1)

    def test_cleanup_continues_after_source_and_model_close_errors(self):
        closed = []
        class Source:
            def __init__(self, name): self.name = name
            def read(self): return None
            def close(self):
                closed.append(self.name)
                if self.name == "a": raise RuntimeError("source close")
        class Model:
            def close(self): raise RuntimeError("model close")
        runtime = ContextPoolRuntime([{"id": "a"}, {"id": "b"}], 1,
                                     lambda s: Source(s["id"]), lambda _: Model(), lambda *_: None)
        runtime.start(); time.sleep(.01); runtime.stop(timeout=.1)
        self.assertEqual(set(closed), {"a", "b"})
        self.assertIsNotNone(runtime.exception)

    def test_timeout_never_releases_model_owned_by_live_context(self):
        entered = threading.Event(); release = threading.Event()
        class Source:
            def read(self): return None
            def close(self): pass
        class Model:
            def __init__(self): self.closed = False
            def close(self): self.closed = True
        model = Model()
        def handler(*_):
            entered.set(); release.wait(2)
        runtime = ContextPoolRuntime([{"id": "a"}], 1, lambda s: Source(), lambda _: model, handler)
        runtime.queues["a"].put(CapturedFrame("frame", 1, 0, 1))
        runtime.start(); self.assertTrue(entered.wait(.5))
        runtime.stop(timeout=.01)
        self.assertFalse(model.closed)
        release.set()
        deadline = time.monotonic() + 1
        while not model.closed and time.monotonic() < deadline: time.sleep(.005)
        self.assertTrue(model.closed)

    def test_start_failure_closes_constructed_but_unstarted_model_and_preserves_error(self):
        class Source:
            def read(self): return None
            def close(self): pass
        class Model:
            def __init__(self): self.closed = False
            def close(self): self.closed = True
        models = []
        def make_model(_):
            model = Model(); models.append(model); return model
        runtime = ContextPoolRuntime([{"id": "a"}], 2, lambda s: Source(), make_model, lambda *_: None)
        original_start = threading.Thread.start
        def fail_second_start(thread):
            if thread.name == "rknn-context-1": raise RuntimeError("start failure")
            return original_start(thread)
        with patch.object(threading.Thread, "start", fail_second_start):
            with self.assertRaisesRegex(RuntimeError, "start failure"):
                runtime.start()
        self.assertEqual(len(models), 2)
        self.assertTrue(models[1].closed)


if __name__ == "__main__": unittest.main()
