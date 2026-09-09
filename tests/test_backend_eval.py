from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.harness import eval as worker
from klineage.harness import timing


class EventTimingTests(unittest.TestCase):
    def test_factory_uses_backend(self):
        from klineage.backend import get_backend

        for language in ("cuda", "hip", "ascendc"):
            with self.subTest(language=language):
                backend = get_backend(language)
                policy = timing.timing_policy(backend)
                timer = timing.make_timer(backend, policy)
                expected = (
                    timing.FlashInferCuptiTimer
                    if language == "cuda"
                    else timing.DeviceEventTimer
                )
                self.assertIsInstance(timer, expected)
                self.assertEqual(policy.cold_l2, language == "cuda")

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
            timing_backend=name,
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
                timer = timing.DeviceEventTimer(backend, policy)
                result = timer.measure(
                    lambda value, calls=calls: calls.append(value), args=("run",)
                )
                self.assertEqual(result.samples_ms, ((2.0, 2.0), (2.0, 2.0)))
                self.assertEqual(result.backend_used, name)
                self.assertEqual(result.details["scope"], "current_stream_interval")
                self.assertEqual(calls.count("run"), 6)
                self.assertEqual(calls.count(("record", stream)), 8)
                self.assertEqual(calls.count("event-sync"), 4)
                self.assertLess(calls.index("event-sync"), calls.index("elapsed"))
                self.assertTrue(timing.verify_performance(result, result))

    def test_events_reject_cold_cache(self):
        backend, calls, _ = self.backend()
        with self.assertRaisesRegex(ValueError, "cold_l2"):
            timing.DeviceEventTimer(backend, timing.TimingPolicy())
        self.assertEqual(calls, [])

    def test_events_reject_cuda(self):
        backend, _, _ = self.backend("cupti")
        with self.assertRaisesRegex(ValueError, "CUPTI"):
            timing.DeviceEventTimer(backend)

    def test_event_backend_mismatch(self):
        backend, _, _ = self.backend()
        result = timing.DeviceEventTimer(backend).measure(lambda: None)
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
        timing.DeviceEventTimer(backend).measure(lambda value: None, args=(value,))
        runtime.device.assert_called_once_with("npu:2")


class BackendWorkerTests(unittest.TestCase):
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

        with patch.object(worker, "require_tensor", return_value=value):
            worker.load_inputs(SimpleNamespace(make_inputs=make_inputs), 7, backend)
        self.assertEqual(calls, [(7, "npu:1")])

    def test_trace_inputs_use_device(self):
        backend = SimpleNamespace(device=lambda: "npu:1")
        value = object()
        module = SimpleNamespace(definition={}, workload={})
        with (
            patch.object(worker, "trace_inputs", return_value={"x": value}) as inputs,
            patch.object(worker, "require_tensor", return_value=value) as require,
        ):
            self.assertEqual(worker.load_inputs(module, 7, backend), {"x": value})
        inputs.assert_called_once_with({}, {}, seed=7, device="npu:1")
        require.assert_called_once_with(value, "problem input 'x'", backend)

    def test_inspection_language(self):
        payload = {
            "problem_name": "add",
            "definition": {},
            "workload": {},
            "language": "ascendc",
            "platform": "ascend-910b-cann8",
        }
        with (
            patch.object(worker.EvaluationRuntime, "inspect", return_value=payload),
            patch.object(worker, "ProblemSpec") as problem,
        ):
            worker.inspect_problem(Path("reference.py"), Path("work"))
        self.assertEqual(problem.call_args.kwargs["language"], "ascendc")

    def test_profile_rejects_backend(self):
        from problem_fixtures import problem_spec

        from klineage.errors import ActionError
        from klineage.kernel import Kernel

        for language, platform in (("hip", "hygon-gfx928"), ("ascendc", "ascend-910b")):
            with self.subTest(language=language):
                kernel = Kernel(
                    "target",
                    problem_spec("add", language, platform),
                    source_files={"kernel.cpp": "source"},
                )
                runtime = worker.EvaluationRuntime()
                with (
                    patch.object(runtime, "collect_profile") as collect,
                    self.assertRaisesRegex(ActionError, "profiling.*unsupported"),
                ):
                    runtime.profile(
                        kernel,
                        build_root=Path("build"),
                        log_dir=Path("logs"),
                        options=Mock(),
                    )
                collect.assert_not_called()

    def test_worker_dispatches_timer(self):
        from klineage.kernel import Kernel

        for name in ("hip-events", "npu-events"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                backend = SimpleNamespace(timing_backend=name, runtime=Mock())
                kernel = Mock()
                accepted = worker.ValidationResult(True, True, True, 1.0)
                with (
                    patch.object(Kernel, "from_dict", return_value=kernel),
                    patch.object(worker, "kernel_backend", return_value=backend),
                    patch.object(worker, "trace_module"),
                    patch.object(worker, "load_inputs", return_value={}) as inputs,
                    patch.object(worker, "compute_reference"),
                    patch.object(worker, "check_problem"),
                    patch.object(worker, "tensor_strides", return_value={}),
                    patch.object(worker, "make_timer") as timer,
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
                self.assertIs(timer.call_args.args[0], backend)
                self.assertFalse(timer.call_args.args[1].cold_l2)
                self.assertIs(inputs.call_args.args[2], backend)
                backend.runtime.assert_called_once()


if __name__ == "__main__":
    unittest.main()
