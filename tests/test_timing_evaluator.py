from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from problem_fixtures import io_problem

from klineage.artifact.kernel import Kernel
from klineage.contract import ABIValue
from klineage.harness.eval import (
    CallableKernelEvaluator,
    CallInputs,
    ProblemRuntime,
)
from klineage.harness.timing import (
    FlashInferCuptiTimer,
    TimingPolicy,
    TimingResult,
    verify_performance,
)


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


class FakeTimer:
    def __init__(self, latencies: list[float]):
        self.policy = TimingPolicy(warmup=1, repeat=1, trials=1)
        self.latencies = list(latencies)
        self.calls: list[
            tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = []

    def measure(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
    ) -> TimingResult:
        self.calls.append((fn, args, {}))
        return timing_result(self.latencies.pop(0))


class TimingTests(unittest.TestCase):
    def test_cupti_checks_sample_count(self):
        timer = FlashInferCuptiTimer(
            TimingPolicy(warmup=1, repeat=2, trials=1),
            benchmark=lambda **kwargs: [1.0],
            backend_probe=lambda: "13.0.0",
        )
        with self.assertRaisesRegex(ValueError, "sample count"):
            timer.measure(lambda: None)

    def test_default_policy_is_strict_cupti(self):
        self.assertEqual(
            TimingPolicy(),
            TimingPolicy(
                warmup=10,
                repeat=50,
                cold_l2=True,
                trials=3,
            ),
        )

    def test_strict_mode_rejects_missing_or_old_cupti_before_benchmark(self):
        calls: list[dict[str, Any]] = []

        def benchmark(**kwargs: Any) -> list[float]:
            calls.append(kwargs)
            return [1.0]

        for probe in (
            lambda: (_ for _ in ()).throw(ModuleNotFoundError("cupti")),
            lambda: "12.9.0",
        ):
            with self.subTest(probe=probe):
                timer = FlashInferCuptiTimer(
                    benchmark=benchmark,
                    backend_probe=probe,
                )
                with self.assertRaisesRegex(RuntimeError, "strict CUPTI"):
                    timer.measure(lambda: None)
        self.assertEqual(calls, [])

    def test_cupti_timer_preserves_samples_and_uses_median_of_trials(self):
        samples = iter(([3.0, 1.0, 2.0], [9.0, 5.0, 7.0], [4.0, 4.0, 4.0]))
        calls: list[dict[str, Any]] = []

        def benchmark(**kwargs: Any) -> list[float]:
            calls.append(kwargs)
            return list(next(samples))

        policy = TimingPolicy(
            warmup=4,
            repeat=3,
            cold_l2=False,
            trials=3,
        )
        timer = FlashInferCuptiTimer(
            policy,
            benchmark=benchmark,
            backend_probe=lambda: "13.2.1",
        )
        fn = lambda x: x * 3
        result = timer.measure(fn, args=(2,))

        self.assertEqual(
            result.samples_ms,
            ((3.0, 1.0, 2.0), (9.0, 5.0, 7.0), (4.0, 4.0, 4.0)),
        )
        self.assertEqual(result.median_ms, 4.0)
        self.assertEqual(result.backend_used, "cupti")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["fn"], fn)
        self.assertEqual(calls[0]["dry_run_iters"], 4)
        self.assertEqual(calls[0]["repeat_iters"], 3)
        self.assertFalse(calls[0]["cold_l2_cache"])
        self.assertFalse(calls[0]["use_cuda_graph"])
        self.assertEqual(calls[0]["input_args"], (2,))
        self.assertEqual(calls[0]["input_kwargs"], {})


class CallableEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = io_problem(
            inputs=(ABIValue("value"),),
            outputs=(ABIValue("result"),),
            name="unit-test",
            language="python",
            platform="cpu",
        )

    def build(self, name="candidate", body="return value"):
        sources = {
            "config.toml": '[solution]\nname="test"\ndefinition="test"\nauthor="test"\n'
            '[build]\nlanguage="python"\nentry_point="kernel.py::run"\n',
            "solution/kernel.py": f"def run(value): {body}\n",
        }
        return Kernel.from_sources(
            sources, self.problem, name=name, build_root=self.root
        )

    def runtime(self, reference=lambda x: x, inputs=(1,)):
        return ProblemRuntime(
            make_inputs=lambda: CallInputs(inputs),
            reference=reference,
            check_outputs=lambda actual, expected: actual == expected,
        )

    def test_success_uses_runtime_reference_and_same_timer(self):
        candidate = self.build(body="return sum(value)")
        timer = FakeTimer([1.0])
        result = CallableKernelEvaluator(
            self.runtime(sum, ([1, 2, 3],)),
            timer=timer,
        ).evaluate(candidate)
        self.assertTrue(result.accepted)
        self.assertEqual(result.latency_ms, 1.0)
        self.assertIsNone(result.reference_latency_ms)
        self.assertEqual(len(timer.calls), 1)
        self.assertIs(timer.calls[0][0], candidate)

    def test_reference_is_only_timed(self):
        candidate = self.build(body="return value + 1")
        reference = self.build("reference", "return value + 1")
        oracle_calls = []

        def oracle(value):
            oracle_calls.append(value)
            return value + 1

        timer = FakeTimer([1.0, 2.0])
        result = CallableKernelEvaluator(
            self.runtime(oracle, (2,)),
            timer=timer,
        ).evaluate(candidate, reference=reference)
        self.assertTrue(result.accepted)
        self.assertEqual(result.latency_ms, 1.0)
        self.assertEqual(result.reference_latency_ms, 2.0)
        self.assertEqual(oracle_calls, [2])
        self.assertIs(timer.calls[1][0], reference)

    def test_rejects_paired_slowdown(self):
        result = CallableKernelEvaluator(
            self.runtime(),
            timer=FakeTimer([2.0, 1.0]),
        ).evaluate(self.build(), reference=self.build("reference"))
        self.assertTrue(result.compile_passed)
        self.assertTrue(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(result.latency_ms, 2.0)
        self.assertEqual(result.reference_latency_ms, 1.0)

    def test_reference_requires_same_abi(self):
        candidate = self.build()
        reference = replace(
            candidate,
            problem=io_problem(
                inputs=(),
                outputs=(),
                name="unit-test",
                language="python",
                platform="cpu",
            ),
        )
        result = CallableKernelEvaluator(self.runtime(), timer=FakeTimer([])).evaluate(
            candidate,
            reference=reference,
        )
        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)

    def test_unbuilt_kernel_fails_gate(self):
        value = Kernel.from_dict(self.build().to_dict())
        timer = FakeTimer([])
        result = CallableKernelEvaluator(self.runtime(), timer=timer).evaluate(value)
        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(timer.calls, [])

    def test_incorrect_output_skips_timing(self):
        timer = FakeTimer([])
        result = CallableKernelEvaluator(self.runtime(), timer=timer).evaluate(
            self.build(body="return value - 1"),
        )
        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(timer.calls, [])

    def test_timing_failure(self):
        class BrokenTimer:
            policy = TimingPolicy(warmup=1, repeat=1, trials=1)

            def measure(self, *args, **kwargs):
                raise RuntimeError("profiler failed")

        result = CallableKernelEvaluator(self.runtime(), timer=BrokenTimer()).evaluate(
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
                timer = Mock(measure=Mock(return_value=evidence))
                result = CallableKernelEvaluator(self.runtime(), timer=timer).evaluate(
                    self.build()
                )
                self.assertTrue(result.compile_passed)
                self.assertTrue(result.correctness_passed)
                self.assertFalse(result.profile_passed)

    def test_evaluator_enforces_arity(self):
        result = CallableKernelEvaluator(
            self.runtime(inputs=()),
            timer=FakeTimer([]),
        ).evaluate(self.build())
        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)

    def test_interrupt_propagates(self):
        evaluator = CallableKernelEvaluator(self.runtime(), timer=FakeTimer([]))
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
        self.assertEqual(record["candidate"], current.to_dict())
        self.assertEqual(record["reference"], reference.to_dict())
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
        self.assertEqual(record["candidate"], current.to_dict())

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
