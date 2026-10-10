from __future__ import annotations

from klineage.serialization import from_json
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from kernel_fixtures import edit_kernel, make_kernel, write_kernel
from problem_fixtures import io_problem

from klineage.artifact.kernel import Kernel
from klineage.backend.cuda import CudaKernel
from klineage.backend.cuda import timing as cuda_timing
from klineage.harness import eval as worker
from klineage.harness.eval import (
    CallableKernelEvaluator,
    CallInputs,
    ProblemRuntime,
)
from klineage.harness.timing import (
    TimingPolicy,
    TimingResult,
    verify_performance,
)
from klineage.serialization import to_json


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
    def test_cupti_sample_count(self):
        policy = TimingPolicy(warmup=1, repeat=2, trials=1)
        with (
            patch.object(cuda_timing, "probe_cupti", return_value="13.0.0"),
            patch.object(
                cuda_timing,
                "load_flashinfer_benchmark",
                return_value=lambda **kwargs: [1.0],
            ),
            self.assertRaisesRegex(ValueError, "sample count"),
        ):
            cuda_timing.bench(lambda: None, policy=policy)

    def test_default_policy(self):
        self.assertEqual(cuda_timing.DEFAULT_POLICY, TimingPolicy(warmup=10, repeat=50, cold_l2=True, trials=3))

    def test_requires_current_cupti(self):
        for probe in (
            Mock(side_effect=ModuleNotFoundError("cupti")),
            Mock(return_value="12.9.0"),
        ):
            with (
                self.subTest(probe=probe),
                patch.object(cuda_timing, "probe_cupti", probe),
                patch.object(cuda_timing, "load_flashinfer_benchmark") as benchmark,
                self.assertRaisesRegex(RuntimeError, "strict CUPTI"),
            ):
                cuda_timing.bench(lambda: None)
            benchmark.assert_not_called()

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

    def test_reference_is_only_timed(self):
        candidate = self.build(body="return value + 1")
        reference = self.build("reference", "return value + 1")
        oracle_calls = []

        def oracle(value):
            oracle_calls.append(value)
            return value + 1

        backend = FakeBackend([1.0, 2.0])
        result = self.evaluator(
            self.runtime(oracle, (2,)),
            backend=backend,
        ).evaluate(candidate, reference=reference)
        self.assertTrue(result.accepted)
        self.assertEqual(result.latency_ms, 1.0)
        self.assertEqual(result.reference_latency_ms, 2.0)
        self.assertEqual(oracle_calls, [2])
        self.assertIs(backend.calls[1][0], reference)

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

    def test_reference_requires_same_abi(self):
        candidate = self.build()
        reference = edit_kernel(
            candidate,
            problem=io_problem(
                inputs=(),
                outputs=(),
                name="unit-test",
                language="python",
                platform="sm120",
            ),
        )
        result = self.evaluator(self.runtime(), backend=FakeBackend([])).evaluate(
            candidate,
            reference=reference,
        )
        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)

    def test_unbuilt_kernel_fails_gate(self):
        value = from_json(to_json(self.build()), Kernel)
        backend = FakeBackend([])
        result = self.evaluator(self.runtime(), backend=backend).evaluate(value)
        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(backend.calls, [])

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

    def test_evaluator_enforces_arity(self):
        result = self.evaluator(
            self.runtime(inputs=()),
            backend=FakeBackend([]),
        ).evaluate(self.build())
        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)

    def test_interrupt_propagates(self):
        evaluator = self.evaluator(self.runtime(), backend=FakeBackend([]))
        with self.assertRaises(KeyboardInterrupt):
            evaluator.evaluate(self.build(body="raise KeyboardInterrupt"))


class PerformanceGateTests(unittest.TestCase):
    def test_performance_threshold(self):
        self.assertTrue(verify_performance(*(timing_result(1.0), timing_result(0.99))))
        self.assertFalse(verify_performance(*(timing_result(1.0), timing_result(0.98))))

    def test_requires_cupti(self):
        current, reference = (timing_result(), timing_result())
        self.assertFalse(
            verify_performance(replace(current, backend_used="cuda_event"), reference)
        )
        self.assertFalse(verify_performance(replace(current, details={}), reference))

    def test_checks_timing_records(self):
        current, reference = (timing_result(), timing_result())
        self.assertFalse(verify_performance(replace(current, median_ms=0.5), reference))
        with self.assertRaises(ValueError):
            timing_result(samples=((float("nan"),),))

    def test_matching_policies(self):
        current, reference = (timing_result(), timing_result())
        current.details["policy"]["cold_l2"] = False
        self.assertFalse(verify_performance(current, reference))

    def test_logs_failed_trial(self):
        current = timing_result(samples=((1.0,), (1.0,), (2.0,)))
        reference = timing_result(samples=((1.0,), (1.0,), (1.0,)))
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertFalse(verify_performance(current, reference))
        record = json.loads(output.getvalue().split("Performance gate: ", 1)[1])
        self.assertEqual(record["candidate"], json.loads(to_json(current)))
        self.assertEqual(record["reference"], json.loads(to_json(reference)))
        self.assertEqual(record["trial_ratios"], [1.0, 1.0, 0.5])
        self.assertEqual(record["overall_ratio"], 1.0)
        self.assertFalse(record["passed"])
        self.assertIn("trial 3", record["reason"])

    def test_logs_invalid_evidence(self):
        current, reference = (timing_result(), timing_result())
        current.details["policy"]["repeat"] = 50
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertFalse(verify_performance(current, reference))
        record = json.loads(output.getvalue().split("Performance gate: ", 1)[1])
        self.assertFalse(record["passed"])
        self.assertIn("sample count", record["reason"])
        self.assertEqual(record["candidate"], json.loads(to_json(current)))

    def test_logs_passed_gate(self):
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertTrue(
                verify_performance(*(timing_result(1.0), timing_result(2.0)))
            )
        record = json.loads(output.getvalue().split("Performance gate: ", 1)[1])
        self.assertTrue(record["passed"])
        self.assertEqual(record["trial_ratios"], [2.0])
        self.assertEqual(record["minimum_ratio"], 0.99)


if __name__ == "__main__":
    unittest.main()
