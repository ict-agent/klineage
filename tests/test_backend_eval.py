from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.backend.runtime import bench_events
from klineage.harness import eval as worker
from klineage.harness import timing
from klineage.serialization import to_json


class EventTimingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "klineage.backend.runtime.device_runtime",
                side_effect=lambda backend: backend.runtime(),
            )
        )
        self.enterContext(
            patch(
                "klineage.backend.runtime.load_torch",
                side_effect=lambda backend: backend.torch(),
            )
        )
        self.enterContext(patch("klineage.backend.runtime.current_device", side_effect=lambda backend: backend.device()))

    def backend(self, name="hip-events"):
        calls = []
        stream = object()

        class Event:
            def __init__(self, *, enable_timing):
                self.assert_timing = enable_timing

            def record(self, target):
                calls.append(("record", target))

            def synchronize(self):
                calls.append("event-sync")

            def elapsed_time(self, end):
                calls.append("elapsed")
                return 2.0

        runtime = SimpleNamespace(
            Event=Event,
            current_stream=lambda device: stream,
            device=lambda device: nullcontext(),
            synchronize=lambda device: calls.append("sync"),
        )
        backend = SimpleNamespace(
            kind="hygon" if name == "hip-events" else "ascend",
            timing=SimpleNamespace(NAME=name, DEFAULT_POLICY=timing.TimingPolicy(cold_l2=False)),
            device_type="cuda" if name == "hip-events" else "npu",
            device=lambda: "device:0",
            runtime=lambda: runtime,
            torch=lambda: SimpleNamespace(inference_mode=nullcontext),
        )
        return backend, calls, stream

    def test_event_samples_and_scope(self):
        for name in ("hip-events", "npu-events"):
            with self.subTest(name=name):
                backend, calls, stream = self.backend(name)
                policy = timing.TimingPolicy(
                    warmup=1, repeat=2, trials=2, cold_l2=False
                )
                result = bench_events(
                    lambda value, calls=calls: calls.append(value),
                    backend=backend,
                    args=("run",),
                    policy=policy,
                )
                self.assertEqual(result.samples_ms, ((2.0, 2.0), (2.0, 2.0)))
                self.assertEqual(result.backend_used, name)
                self.assertEqual(result.details["scope"], "current_stream_interval")
                self.assertEqual(calls.count("run"), 6)
                self.assertEqual(calls.count(("record", stream)), 8)
                self.assertEqual(calls.count("event-sync"), 4)
                self.assertLess(calls.index("event-sync"), calls.index("elapsed"))
                self.assertTrue(timing.verify_performance(result, result))

    def test_backend_bench_dispatch(self):
        from klineage.backend import get_backend

        for language in ("hip", "ascendc"):
            backend = get_backend(language)
            fake, calls, stream = self.backend(backend.timing.NAME)
            policy = timing.TimingPolicy(warmup=1, repeat=2, trials=1, cold_l2=False)
            with (
                self.subTest(language=language),
                patch(
                    "klineage.backend.runtime.device_runtime",
                    return_value=fake.runtime(),
                ),
                patch("klineage.backend.runtime.load_torch", return_value=fake.torch()),
                patch(
                    "klineage.backend.runtime.current_device",
                    return_value=fake.device(),
                ),
            ):
                result = backend.bench(
                    lambda value: calls.append(value), args=("run",), policy=policy
                )
            self.assertEqual(result.backend_used, backend.timing.NAME)
            self.assertEqual(result.samples_ms, ((2.0, 2.0),))
            self.assertEqual(calls.count("run"), 3)
            self.assertIn(("record", stream), calls)

    def test_events_reject_cold_cache(self):
        backend, calls, _ = self.backend()
        with self.assertRaisesRegex(ValueError, "cold_l2"):
            bench_events(lambda: None, backend=backend, policy=timing.TimingPolicy())
        self.assertEqual(calls, [])

    def test_event_backend_mismatch(self):
        backend, _, _ = self.backend()
        result = bench_events(lambda: None, backend=backend)
        other = replace(result, backend_used="npu-events")
        self.assertFalse(timing.verify_performance(result, other))
        bad_scope = replace(
            result, details={**result.details, "scope": "kernel_activity"}
        )
        self.assertFalse(timing.verify_performance(bad_scope, result))

    def test_events_use_input_device(self):
        backend, _, _ = self.backend()
        runtime = backend.runtime()
        runtime.device = Mock(return_value=nullcontext())
        value = SimpleNamespace(device="npu:2")
        bench_events(lambda value: None, backend=backend, args=(value,))
        runtime.device.assert_called_once_with("npu:2")


class BackendWorkerTests(unittest.TestCase):
    def test_rejects_empty_reference(self):
        from kernel_fixtures import kernel


        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(worker, "device_runtime"),
            patch.object(
                worker,
                "trace_module",
                side_effect=AssertionError("Malformed reference reached execution"),
            ),
            self.assertRaisesRegex(ValueError, "artifact envelope"),
        ):
            worker.evaluate_request(
                {
                    "kernel": json.loads(to_json(kernel())),
                    "reference": {},
                    "config": {"build_root": directory},
                }
            )

    def test_clone_preserves_layout(self):
        import torch

        examples = (
            torch.arange(24).reshape(4, 6)[:, 1::2],
            torch.tensor([3.0]).expand(3, 4),
            torch.empty_strided((0, 3), (9, 2)),
        )
        for value in examples:
            with self.subTest(shape=value.shape, stride=value.stride()):
                cloned = worker.clone_value(value)
                self.assertEqual(cloned.stride(), value.stride())
                torch.testing.assert_close(cloned, value)
                if value.numel():
                    self.assertNotEqual(cloned.data_ptr(), value.data_ptr())
                    original = value[0, 0].item()
                    cloned[0, 0] = -99
                    self.assertEqual(value[0, 0].item(), original)

    def test_python_inputs_get_device(self):
        backend = SimpleNamespace(device=lambda: "npu:1")
        value = object()
        calls = []

        def make_inputs(*, seed, device):
            calls.append((seed, device))
            return {"x": value}

        with patch.object(worker, "require_tensor", return_value=value), patch.object(worker, "current_device", return_value="npu:1"):
            worker.load_inputs(SimpleNamespace(make_inputs=make_inputs), 7, backend)
        self.assertEqual(calls, [(7, "npu:1")])

    def test_trace_inputs_use_device(self):
        backend = SimpleNamespace(device=lambda: "npu:1")
        value = object()
        module = SimpleNamespace(definition={}, workloads=({},))
        with (
            patch.object(worker, "trace_inputs", return_value={"x": value}) as inputs,
            patch.object(worker, "current_device", return_value="npu:1"),
            patch.object(worker, "require_tensor", return_value=value) as require,
        ):
            self.assertEqual(worker.load_inputs(module, 7, backend), {"x": value})
        inputs.assert_called_once_with({}, {}, seed=7, device="npu:1")
        require.assert_called_once_with(value, "problem input 'x'", backend)

    def test_inspection_language(self):
        payload = {
            "problem_name": "add",
            "definition": {},
            "workloads": ({},),
            "language": "ascendc",
            "platform": "ascend-910b-cann8",
        }
        with (
            patch.object(worker.EvaluationRuntime, "inspect", return_value=payload),
            patch.object(worker, "Problem") as problem,
        ):
            worker.inspect_problem(Path("reference.py"), Path("work"))
        self.assertEqual(problem.call_args.kwargs["language"], "ascendc")

    def test_worker_dispatches_timer(self):
        from kernel_fixtures import kernel as kernel_fixture


        for name in ("hip-events", "npu-events"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                backend = SimpleNamespace(timing=SimpleNamespace(NAME=name, DEFAULT_POLICY=timing.TimingPolicy(cold_l2=False)))
                kernel = kernel_fixture()
                accepted = worker.ValidationResult(True, True, True, 1.0)
                with (
                    patch.object(worker, "restore", return_value=kernel),
                    patch.object(type(kernel), "build_kernel", return_value=lambda: None),
                    patch.object(worker, "get_backend", return_value=backend),
                    patch.object(worker, "trace_module"),
                    patch.object(worker, "load_inputs", return_value={}) as inputs,
                    patch.object(worker, "compute_reference"),
                    patch.object(worker, "check_problem"),
                    patch.object(worker, "tensor_strides", return_value={}),
                    patch.object(worker, "device_runtime") as runtime,
                    patch.object(worker, "CallableKernelEvaluator") as evaluator,
                ):
                    evaluator.return_value.evaluate.return_value = accepted
                    result = worker.evaluate_request(
                        {
                            "kernel": {},
                            "config": {"build_root": directory},
                        }
                    )
                self.assertEqual(result, accepted)
                self.assertFalse(evaluator.call_args.kwargs["policy"].cold_l2)
                self.assertIs(inputs.call_args.args[2], backend)
                runtime.assert_called_once_with(backend)


class TimingDispatchTests(unittest.TestCase):
    def test_detects_then_benches(self):
        from kernel_fixtures import kernel

        from klineage.backend import backends, registry
        from klineage.backend.timing import TimingPolicy, TimingResult, bench

        for backend in backends():
            current = kernel(language=backend.language, platform=backend.kind)
            object.__setattr__(current, "function", lambda value: value)
            policy = TimingPolicy(warmup=1, repeat=1, trials=1, cold_l2=False)
            expected = TimingResult(((1.0,),), 1.0, backend.timing.NAME)
            events = []
            with (
                self.subTest(backend=backend.kind),
                patch.object(
                    registry,
                    "detect_backend",
                    side_effect=lambda: events.append("detect") or backend,
                ),
                patch.object(
                    type(backend),
                    "bench",
                    side_effect=lambda *args, **kwargs: (
                        events.append("bench") or expected
                    ),
                ) as measure,
            ):
                self.assertIs(bench(current, args=(3,), policy=policy), expected)
            self.assertEqual(events, ["detect", "bench"])
            measure.assert_called_once_with(current, args=(3,), policy=policy)

    def test_rejects_target_mismatch(self):
        from kernel_fixtures import kernel

        from klineage.backend import get_backend, registry
        from klineage.backend.timing import bench

        backend = get_backend("hip")
        with (
            patch.object(registry, "detect_backend", return_value=backend),
            patch.object(type(backend), "bench") as measure,
            self.assertRaisesRegex(ValueError, "kernel target"),
        ):
            bench(kernel())
        measure.assert_not_called()

    def test_detection_failure(self):
        from kernel_fixtures import kernel

        from klineage.backend import registry
        from klineage.backend.timing import bench

        with (
            patch.object(
                registry, "detect_backend", side_effect=RuntimeError("no backend")
            ),
            self.assertRaisesRegex(RuntimeError, "no backend"),
        ):
            bench(kernel())


if __name__ == "__main__":
    unittest.main()
