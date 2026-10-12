from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from kernel_fixtures import make_kernel, write_kernel
from problem_fixtures import io_problem

from klineage.backend.cuda import CudaKernel
from klineage.backend.cuda import timing as cuda_timing
from klineage.backend.runtime import bench_events
from klineage.harness import eval as worker
from klineage.harness import timing
from klineage.harness.eval import CallableKernelEvaluator, CallInputs, ProblemRuntime
from klineage.harness.timing import TimingPolicy, TimingResult, verify_performance


def timing_result(latency_ms=1.0, *, samples=None, backend="cupti") -> TimingResult:
    return TimingResult(
        samples_ms=samples or ((latency_ms,),),
        median_ms=latency_ms,
        backend_used=backend,
        details={
            "cupti_version": "13.0",
            "policy": {
                "warmup": 1,
                "repeat": 1,
                "trials": len(samples) if samples else 1,
                "cold_l2": True,
            },
        },
    )


class FakeBackend:
    def __init__(self, latencies: list[float]):
        self.policy = TimingPolicy(warmup=1, repeat=1, trials=1)
        self.timing = SimpleNamespace(DEFAULT_POLICY=self.policy)
        self.latencies = list(latencies)
        self.calls: list[
            tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []

    def bench(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
        policy=None,
    ) -> TimingResult:
        self.calls.append((fn, args, {}))
        return timing_result(self.latencies.pop(0))


class TimingTests(unittest.TestCase):

    def test_bench_samples_and_args(self):
        samples = ([3.0, 1.0, 2.0], [9.0, 5.0, 7.0], [4.0, 4.0, 4.0])
        benchmark = Mock(side_effect=samples)
        policy = TimingPolicy(warmup=4, repeat=3, cold_l2=False, trials=3)
        fn = lambda x: x * 3
        with (
            patch.object(cuda_timing, "probe_cupti", return_value="13.2.1"),
            patch.object(
                cuda_timing, "load_flashinfer_benchmark", return_value=benchmark
            ),
        ):
            result = cuda_timing.bench(fn, args=(2,), policy=policy)
        self.assertEqual(result.samples_ms, tuple(tuple(values) for values in samples))
        self.assertEqual(result.median_ms, 4.0)
        self.assertEqual(result.backend_used, "cupti")
        self.assertEqual(benchmark.call_count, 3)
        self.assertEqual(
            benchmark.call_args.kwargs,
            {
                "fn": fn,
                "dry_run_iters": 4,
                "repeat_iters": 3,
                "cold_l2_cache": False,
                "use_cuda_graph": False,
                "input_args": (2,),
                "input_kwargs": {},
            },
        )


class CallableEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = io_problem(
            inputs={"value": {}},
            outputs={"result": {}},
            name="unit-test",
            language="python",
            platform="sm120",
        )

    def evaluator(self, runtime, *, backend):
        self.enterContext(patch.object(worker, "bench", side_effect=backend.bench))
        return CallableKernelEvaluator(runtime, policy=backend.timing.DEFAULT_POLICY)

    def build(self, name="candidate", body="return value"):
        sources = {
            "config.toml": '[solution]\nname="test"\ndefinition="test"\nauthor="test"\n'
            '[build]\nlanguage="python"\nentry_point="kernel.py::run"\n',
            "solution/kernel.py": f"def run(value): {body}\n",
        }
        return CudaKernel.load_kernel(write_kernel(make_kernel(name, self.problem, sources), self.root / name))

    def runtime(self, reference=lambda x: x, inputs=(1,)):
        return ProblemRuntime(
            make_inputs=lambda: CallInputs(inputs),
            reference=reference,
            check_outputs=lambda actual, expected: actual == expected,
        )

    def test_success_uses_runtime_reference_and_same_backend(self):
        candidate = self.build(body="return sum(value)")
        backend = FakeBackend([1.0])
        result = self.evaluator(
            self.runtime(sum, ([1, 2, 3],)),
            backend=backend,
        ).evaluate(candidate)
        self.assertTrue(result.accepted)
        self.assertEqual(result.latency_ms, 1.0)
        self.assertIsNone(result.reference_latency_ms)
        self.assertEqual(len(backend.calls), 1)
        self.assertIs(backend.calls[0][0], candidate)

    def test_rejects_paired_slowdown(self):
        result = self.evaluator(
            self.runtime(),
            backend=FakeBackend([2.0, 1.0]),
        ).evaluate(self.build(), reference=self.build("reference"))
        self.assertTrue(result.compile_passed)
        self.assertTrue(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(result.latency_ms, 2.0)
        self.assertEqual(result.reference_latency_ms, 1.0)

    def test_incorrect_output_skips_timing(self):
        backend = FakeBackend([])
        result = self.evaluator(self.runtime(), backend=backend).evaluate(
            self.build(body="return value - 1"),
        )
        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(backend.calls, [])

    def test_timing_failure(self):
        class BrokenBackend:
            timing = SimpleNamespace(DEFAULT_POLICY=TimingPolicy(warmup=1, repeat=1, trials=1))

            def bench(self, *args, **kwargs):
                raise RuntimeError("profiler failed")

        result = self.evaluator(self.runtime(), backend=BrokenBackend()).evaluate(
            self.build()
        )
        self.assertTrue(result.compile_passed)
        self.assertTrue(result.correctness_passed)
        self.assertFalse(result.profile_passed)

    def test_rejects_invalid_evidence(self):
        valid = timing_result(1.0)
        invalid = (
            replace(valid, backend_used="cuda-events"),
            replace(valid, samples_ms=((1.0, 1.0),)),
            replace(valid, details={**valid.details, "cupti_version": "12.9"}),
        )
        for evidence in invalid:
            with self.subTest(evidence=evidence):
                backend = Mock(bench=Mock(return_value=evidence))
                result = self.evaluator(self.runtime(), backend=backend).evaluate(
                    self.build()
                )
                self.assertTrue(result.compile_passed)
                self.assertTrue(result.correctness_passed)
                self.assertFalse(result.profile_passed)


class PerformanceGateTests(unittest.TestCase):

    def test_checks_timing_records(self):
        current, reference = (timing_result(), timing_result())
        self.assertFalse(verify_performance(replace(current, median_ms=0.5), reference))
        with self.assertRaises(ValueError):
            timing_result(samples=((float("nan"),),))

    def test_matching_policies(self):
        current, reference = (timing_result(), timing_result())
        current.details["policy"]["cold_l2"] = False
        self.assertFalse(verify_performance(current, reference))


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

    def test_events_reject_cold_cache(self):
        backend, calls, _ = self.backend()
        with self.assertRaisesRegex(ValueError, "cold_l2"):
            bench_events(lambda: None, backend=backend, policy=timing.TimingPolicy())
        self.assertEqual(calls, [])

    def test_events_use_input_device(self):
        backend, _, _ = self.backend()
        runtime = backend.runtime()
        runtime.device = Mock(return_value=nullcontext())
        value = SimpleNamespace(device="npu:2")
        bench_events(lambda value: None, backend=backend, args=(value,))
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
