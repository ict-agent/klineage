from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from klineage.contract import ABIValue, EvaluatorInterface, KernelABI
from klineage.harness.callable_eval import (
    CallableKernelEvaluator,
    CallInputs,
    ProblemRuntime,
    PythonEntrypointLoader,
)
from klineage.harness.timing import (
    FlashInferCuptiTimer,
    TimingPolicy,
    TimingResult,
)
from klineage.kernel import Kernel, TargetContext


def timing_result(latency_ms: float, *, backend: str = "fake") -> TimingResult:
    return TimingResult(
        samples_ms=((latency_ms,),),
        median_ms=latency_ms,
        backend_used=backend,
    )


def kernel(name: str, path: Path | None = None) -> Kernel:
    return Kernel(
        name=name,
        source="# source-directory submission",
        context=TargetContext("unit-test", "python", "cpu"),
        artifact_path=path,
        abi=KernelABI(
            inputs=(ABIValue("value"),),
            outputs=(ABIValue("result"),),
            interface=EvaluatorInterface(),
        ),
    )


class FakeLoader:
    def __init__(self, functions: Mapping[str, Callable[..., Any]]) -> None:
        self.functions = dict(functions)
        self.calls: list[str] = []

    def load(self, value: Kernel) -> Callable[..., Any]:
        self.calls.append(value.name)
        loaded = self.functions[value.name]
        if isinstance(loaded, Exception):
            raise loaded
        return loaded


class FakeTimer:
    def __init__(self, latencies: list[float]) -> None:
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
    def test_default_policy_is_strict_cupti(self) -> None:
        self.assertEqual(
            TimingPolicy(),
            TimingPolicy(
                warmup=10,
                repeat=50,
                cold_l2=True,
                trials=3,
            ),
        )

    def test_strict_mode_rejects_missing_or_old_cupti_before_benchmark(self) -> None:
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

    def test_cupti_timer_preserves_samples_and_uses_median_of_trials(self) -> None:
        samples = iter(([3.0, 1.0, 2.0], [9.0, 5.0, 7.0], [4.0, 4.0, 4.0]))
        calls: list[dict[str, Any]] = []

        def benchmark(**kwargs: Any) -> list[float]:
            calls.append(kwargs)
            return list(next(samples))

        policy = TimingPolicy(
            warmup=4,
            repeat=7,
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
        self.assertEqual(calls[0]["repeat_iters"], 7)
        self.assertFalse(calls[0]["cold_l2_cache"])
        self.assertFalse(calls[0]["use_cuda_graph"])
        self.assertEqual(calls[0]["input_args"], (2,))
        self.assertEqual(calls[0]["input_kwargs"], {})

class CallableEvaluatorTests(unittest.TestCase):
    def test_python_loader_imports_declared_source_directory_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helper.py").write_text("OFFSET = 2\n", encoding="utf-8")
            (root / "submission.py").write_text(
                "def load():\n"
                "    from helper import OFFSET\n"
                "    return lambda value: value + OFFSET\n",
                encoding="utf-8",
            )

            loaded = PythonEntrypointLoader().load(kernel("candidate", root))

        self.assertEqual(loaded(3), 5)

    def test_python_loader_rejects_a_symlinked_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real.py").write_text(
                "def load():\n    return lambda value: value\n",
                encoding="utf-8",
            )
            (root / "submission.py").symlink_to(root / "real.py")

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                PythonEntrypointLoader().load(kernel("candidate", root))

    def test_success_uses_runtime_reference_and_same_timer(self) -> None:
        candidate = kernel("candidate")
        loader = FakeLoader({"candidate": lambda values: sum(values)})
        timer = FakeTimer([1.0, 2.0])
        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(args=([1, 2, 3],)),
            reference=lambda values: sum(values),
            check_outputs=lambda actual, expected: actual == expected,
        )
        evaluator = CallableKernelEvaluator(runtime, loader=loader, timer=timer)

        result = evaluator.evaluate(candidate)

        self.assertTrue(result.accepted)
        self.assertEqual(result.latency_ms, 1.0)
        self.assertEqual(result.reference_latency_ms, 2.0)
        self.assertEqual(loader.calls, ["candidate"])
        self.assertEqual(len(timer.calls), 2)
        self.assertIs(timer.calls[0][0], loader.functions["candidate"])
        self.assertIs(timer.calls[1][0], runtime.reference)

    def test_reference_kernel_is_timed_but_runtime_oracle_checks_correctness(
        self,
    ) -> None:
        candidate = kernel("candidate")
        reference = kernel("reference")
        loader = FakeLoader(
            {
                "candidate": lambda value: value + 1,
                "reference": lambda value: value + 1,
            }
        )
        runtime_reference_calls: list[int] = []

        def oracle(value: int) -> int:
            runtime_reference_calls.append(value)
            return value + 1

        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(args=(2,)),
            reference=oracle,
            check_outputs=lambda actual, expected: actual == expected,
        )
        timer = FakeTimer([1.0, 1.0])
        result = CallableKernelEvaluator(
            runtime,
            loader=loader,
            timer=timer,
        ).evaluate(candidate, reference=reference)

        self.assertTrue(result.accepted)
        self.assertEqual(loader.calls, ["candidate", "reference"])
        self.assertEqual(runtime_reference_calls, [2])
        self.assertIs(timer.calls[1][0], loader.functions["reference"])

    def test_reference_kernel_must_have_the_same_abi(self) -> None:
        candidate = kernel("candidate")
        reference = kernel("reference")
        reference = Kernel(
            name=reference.name,
            source=reference.source,
            context=reference.context,
            artifact_path=reference.artifact_path,
            abi=KernelABI(
                interface=EvaluatorInterface(loader="load_reference"),
            ),
        )
        loader = FakeLoader(
            {
                "candidate": lambda value: value,
                "reference": lambda value: value,
            }
        )
        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(args=(1,)),
            reference=lambda value: value,
            check_outputs=lambda actual, expected: actual == expected,
        )

        result = CallableKernelEvaluator(
            runtime,
            loader=loader,
            timer=FakeTimer([]),
        ).evaluate(candidate, reference=reference)

        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertIn("ABIs do not match", result.details["build_load"]["error"])
        self.assertEqual(loader.calls, [])

    def test_load_failure_only_fails_build_gate_and_skips_later_layers(self) -> None:
        value = kernel("candidate")

        class BrokenLoader:
            def load(self, _: Kernel) -> Callable[..., Any]:
                raise RuntimeError("compiler failed")

        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(),
            reference=lambda: None,
            check_outputs=lambda actual, expected: True,
        )
        result = CallableKernelEvaluator(
            runtime,
            loader=BrokenLoader(),
            timer=FakeTimer([]),
        ).evaluate(value)

        self.assertFalse(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(result.details["build_load"]["error_type"], "RuntimeError")
        self.assertTrue(result.details["correctness"]["skipped"])
        self.assertTrue(result.details["timing"]["skipped"])

    def test_incorrect_output_fails_correctness_and_skips_timing(self) -> None:
        value = kernel("candidate")
        timer = FakeTimer([])
        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(args=(1,)),
            reference=lambda x: x + 1,
            check_outputs=lambda actual, expected: actual == expected,
        )
        result = CallableKernelEvaluator(
            runtime,
            loader=FakeLoader({"candidate": lambda x: x - 1}),
            timer=timer,
        ).evaluate(value)

        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(timer.calls, [])
        self.assertIn("rejected", result.details["correctness"]["reason"])

    def test_timing_exception_only_fails_profile_gate(self) -> None:
        value = kernel("candidate")

        class BrokenTimer:
            policy = TimingPolicy(warmup=1, repeat=1, trials=1)

            def measure(self, *_: Any, **__: Any) -> TimingResult:
                raise RuntimeError("profiler failed")

        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(args=(1,)),
            reference=lambda x: x,
            check_outputs=lambda actual, expected: actual == expected,
        )
        result = CallableKernelEvaluator(
            runtime,
            loader=FakeLoader({"candidate": lambda x: x}),
            timer=BrokenTimer(),
        ).evaluate(value)

        self.assertTrue(result.compile_passed)
        self.assertTrue(result.correctness_passed)
        self.assertFalse(result.profile_passed)
        self.assertEqual(result.details["timing"]["error_type"], "RuntimeError")

    def test_callable_evaluator_enforces_abi_arity(self) -> None:
        value = kernel("candidate")
        evaluator = CallableKernelEvaluator(
            ProblemRuntime(
                make_inputs=lambda: CallInputs(),
                reference=lambda: 1,
                check_outputs=lambda actual, expected: actual == expected,
            ),
            loader=FakeLoader({"candidate": lambda: 1}),
            timer=FakeTimer([]),
        )

        result = evaluator.evaluate(value)

        self.assertTrue(result.compile_passed)
        self.assertFalse(result.correctness_passed)
        self.assertIn("ABI declares 1", result.details["correctness"]["error"])

    def test_control_flow_exceptions_are_not_swallowed(self) -> None:
        value = kernel("candidate")

        class InterruptedLoader:
            def load(self, _: Kernel) -> Callable[..., Any]:
                raise KeyboardInterrupt

        runtime = ProblemRuntime(
            make_inputs=lambda: CallInputs(),
            reference=lambda: None,
            check_outputs=lambda actual, expected: True,
        )
        evaluator = CallableKernelEvaluator(
            runtime,
            loader=InterruptedLoader(),
            timer=FakeTimer([]),
        )
        with self.assertRaises(KeyboardInterrupt):
            evaluator.evaluate(value)


if __name__ == "__main__":
    unittest.main()
