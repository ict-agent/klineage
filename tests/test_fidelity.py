from __future__ import annotations

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import require_cuda_source_bundle, require_pure_cuda
from klineage.errors import ValidationGateError
from klineage.harness.eval import ValidationResult
from klineage.harness.fidelity import (
    EvidenceMode,
    check_evidence,
    verify_mechanism,
    verify_performance,
)
from klineage.kernel import Kernel, TargetContext


def measurement(candidate: float = 1.0, reference: float = 1.0) -> ValidationResult:
    def timing(latency: float) -> dict:
        return {
            "backend_used": "cupti",
            "samples_ms": [[latency], [latency]],
            "median_ms": latency,
            "details": {
                "cupti_version": "13.0.0",
                "policy": {"warmup": 1, "repeat": 1, "trials": 2, "cold_l2": True},
            },
        }

    return ValidationResult(
        True, True, True, candidate, reference,
        {"timing": {"candidate": timing(candidate), "reference": timing(reference)}},
    )


def citation(path: str, quote: str) -> dict:
    return {"path": path, "start": 1, "end": 1, "quote": quote}


def compact_audit() -> dict:
    return {
        "standalone": {
            "status": "standalone", "reason": "ABI launches candidate computation.",
            "candidate": [{"path": "candidate", "start": 1, "end": 1}],
        },
        "checks": [
            {
                "mechanism": axis, "status": "preserved", "reason": "Source agrees.",
                "expert": [{"path": "expert", "start": 1, "end": 1}],
                "candidate": [{"path": "candidate", "start": 1, "end": 1}],
            }
            for axis in ("compute", "tiling", "pipeline", "layout", "scheduling")
        ],
    }


class FidelityTests(unittest.TestCase):
    def test_performance_threshold(self) -> None:
        self.assertTrue(verify_performance(measurement(1.0, 0.99)).accepted)
        self.assertFalse(verify_performance(measurement(1.0, 0.98)).accepted)

    def test_requires_cupti(self) -> None:
        self.assertFalse(verify_performance(ValidationResult(True, True, True, 1, 1)).accepted)
        result = measurement()
        result.details["timing"]["candidate"]["backend_used"] = "cuda_event"
        self.assertFalse(verify_performance(result).accepted)

    def test_checks_timing_records(self) -> None:
        for field, value in (("median_ms", 0.5), ("samples_ms", [[float("nan")]])):
            with self.subTest(field=field):
                result = measurement()
                result.details["timing"]["candidate"][field] = value
                self.assertFalse(verify_performance(result).accepted)
        result = measurement()
        result.details["timing"]["candidate"]["details"]["policy"]["repeat"] = 2
        self.assertFalse(verify_performance(result).accepted)

    def test_checks_sample_count(self) -> None:
        result = measurement()
        for side in ("candidate", "reference"):
            result.details["timing"][side]["details"]["policy"]["repeat"] = 50
        self.assertFalse(verify_performance(result).accepted)

    def test_rejects_library_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "expert.cuh").write_text("void library_launch();")
            for header in ('"expert.cuh"', f'"{root / "expert.cuh"}"'):
                source = f'#include {header}\n__global__ void unused() {{}}'
                with self.assertRaises(ValidationGateError):
                    require_pure_cuda(source, repository=root)
        for source in (
            '#include <cutlass/gemm/device/gemm.h>\n__global__ void unused() {}',
            '__global__ void unused() {} void launch() { cublasGemmEx(); }',
            '#define HEADER "expert.cuh"\n#include HEADER\n__global__ void unused() {}',
        ):
            with self.assertRaises(ValidationGateError):
                require_pure_cuda(source)
        require_pure_cuda('#include <cuda_runtime.h>\n#include <mma.h>\n__global__ void compute() {}')

    def test_bundle_stays_standalone(self) -> None:
        loader = "from torch.utils.cpp_extension import load_inline\n"
        device = "__global__ void compute() {}"
        require_cuda_source_bundle({"submission.py": loader + f'cuda_source = """{device}"""'})
        with self.assertRaises(ValidationGateError):
            require_cuda_source_bundle({
                "submission.py": loader + 'cuda_source = """#include <cutlass/gemm/device/gemm.h>\n'
                + device + '"""',
            })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "expert.cuh").write_text("void expert();")
            with self.assertRaises(ValidationGateError):
                require_cuda_source_bundle({
                    "submission.py": loader,
                    "kernel.cu": '#include "expert.cuh"\n' + device,
                }, repository=root)

    def test_checks_every_trial(self) -> None:
        result = measurement()
        for side in ("candidate", "reference"):
            timing = result.details["timing"][side]
            timing["samples_ms"] = [[1.0], [1.0], [1.0]]
            timing["details"]["policy"]["trials"] = 3
        result.details["timing"]["candidate"]["samples_ms"][-1] = [2.0]
        self.assertFalse(verify_performance(result).accepted)

    def test_checks_source_quotes(self) -> None:
        valid = citation("candidate.cu", "mma.sync;")
        checked = check_evidence([valid], {"candidate.cu": "mma.sync;\n"})
        self.assertEqual(len(checked[0]["sha256"]), 64)
        with self.assertRaises(ValueError):
            check_evidence([valid], {"candidate.cu": "float scalar;\n"})
        with self.assertRaises(ValueError):
            check_evidence([], {"candidate.cu": "mma.sync;\n"})
        with self.assertRaises(ValueError):
            check_evidence(
                [{"path": "candidate.cu", "start": 1, "end": 1}],
                {"candidate.cu": "mma.sync;\n"},
            )
        ranges = check_evidence(
            [{"path": "candidate.cu", "start": 1, "end": 1}],
            {"candidate.cu": "mma.sync;\n"}, EvidenceMode.RANGES,
        )
        self.assertEqual(ranges, checked)

    def test_materializes_audit_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "__global__ void compute() {}\n"
            context = TargetContext("gemm", "cuda", "sm80")
            expert = Kernel("expert", source, context)
            report = compact_audit()
            result = verify_mechanism(lambda *_: report, expert, expert, root)
            self.assertTrue(result["passed"], result)
            evidence = [result["standalone"]["candidate"][0]]
            evidence.extend(
                check[role][0]
                for check in result["checks"] for role in ("expert", "candidate")
            )
            for item in evidence:
                self.assertEqual(item["quote"], source.rstrip("\n"))
                self.assertEqual(item["sha256"], hashlib.sha256(source.encode()).hexdigest())
            self.assertNotIn("quote", report["standalone"]["candidate"][0])

    def test_bounds_audit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "\n".join(f"compute_{line}();" for line in range(13))
            expert = Kernel("expert", source, TargetContext("gemm", "cuda", "sm80"))
            for role in ("standalone", "expert", "candidate"):
                for count, end, passed in ((0, 1, False), (5, 1, False), (1, 13, False), (4, 12, True)):
                    with self.subTest(role=role, count=count, end=end):
                        report = compact_audit()
                        path = "expert" if role == "expert" else "candidate"
                        items = [{
                            "path": path, "start": 1, "end": end,
                            "quote": "\n".join(source.splitlines()[:end]),
                        }] * count
                        target = report["standalone"] if role == "standalone" else report["checks"][0]
                        target["candidate" if role == "standalone" else role] = items
                        # Supply quotes elsewhere to isolate bounds from range support.
                        for check in [report["standalone"], *report["checks"]]:
                            for side in ("expert", "candidate"):
                                for item in check.get(side, []):
                                    item.setdefault("quote", source.splitlines()[0])
                        result = verify_mechanism(lambda *_: report, expert, expert, root)
                        self.assertEqual(result["passed"], passed, result)

    def test_rejects_bad_audit_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "compute();\n"
            expert = Kernel("expert", source, TargetContext("gemm", "cuda", "sm80"))
            for change in (
                {"path": "../secret.cuh"}, {"path": "/tmp/secret.cuh"},
                {"path": "missing.cuh"}, {"path": []},
                {"start": 0}, {"start": True}, {"end": 2}, {"end": 0},
                {"quote": "forged();"}, {"quote": None},
            ):
                for role in ("standalone", "expert", "candidate"):
                    with self.subTest(change=change, role=role):
                        report = compact_audit()
                        target = report["standalone"] if role == "standalone" else report["checks"][0]
                        target["candidate" if role == "standalone" else role][0].update(change)
                        result = verify_mechanism(lambda *_: report, expert, expert, root)
                        self.assertFalse(result["passed"], result)

    def test_requires_known_mechanism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "__global__ void compute() {}"
            context = TargetContext("gemm", "cuda", "sm80")
            expert = Kernel("expert", source, context)
            candidate = Kernel("candidate", source, context)
            report = {
                "checks": [
                    {
                        "mechanism": axis,
                        "status": "preserved",
                        "reason": "The same computation is present.",
                        "expert": [citation("expert", source)],
                        "candidate": [citation("candidate", source)],
                    }
                    for axis in ("compute", "tiling", "pipeline", "layout", "scheduling")
                ]
            }
            report["standalone"] = {
                "status": "standalone", "reason": "ABI executes candidate device code.",
                "candidate": [citation("candidate", candidate.source)],
            }
            ask = lambda *_: deepcopy(report)
            self.assertTrue(verify_mechanism(ask, expert, candidate, root)["passed"])
            report["standalone"]["status"] = "delegated"
            self.assertFalse(verify_mechanism(ask, expert, candidate, root)["passed"])
            report["standalone"]["status"] = "standalone"
            report["checks"][0]["status"] = "unknown"
            self.assertFalse(verify_mechanism(ask, expert, candidate, root)["passed"])
            report["checks"].pop()
            self.assertFalse(verify_mechanism(ask, expert, candidate, root)["passed"])

    def test_rejects_malformed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = TargetContext("gemm", "cuda", "sm80")
            expert = Kernel("expert", "void compute() {}", context)
            reports = [None, [], {"checks": [None] * 5}]
            reports.append({"checks": [
                {"mechanism": axis, "expert": [None]}
                for axis in ("compute", "tiling", "pipeline", "layout", "scheduling")
            ]})
            for report in reports:
                with self.subTest(report=report):
                    ask = lambda *_: report
                    self.assertFalse(verify_mechanism(ask, expert, expert, root)["passed"])
            def invalid_json(*_):
                raise StructuredOutputError("invalid JSON")
            self.assertFalse(verify_mechanism(invalid_json, expert, expert, root)["passed"])

    def test_requires_library_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "library.cuh").write_text("mma.sync;\n")
            source = '#include "library.cuh"'
            context = TargetContext("gemm", "cuda", "sm80")
            expert = Kernel("expert", source, context)
            candidate = Kernel("candidate", "mma.sync;", context)
            report = {"checks": [
                {
                    "mechanism": axis, "status": "preserved", "reason": "Source agrees.",
                    "expert": [citation("expert", source)],
                    "candidate": [citation("candidate", candidate.source)],
                }
                for axis in ("compute", "tiling", "pipeline", "layout", "scheduling")
            ]}
            report["standalone"] = {
                "status": "standalone", "reason": "ABI executes candidate device code.",
                "candidate": [citation("candidate", candidate.source)],
            }
            ask = lambda *_: deepcopy(report)
            self.assertFalse(verify_mechanism(ask, expert, candidate, root)["passed"])
            report["checks"][0]["expert"].append(citation("library.cuh", "mma.sync;"))
            self.assertTrue(verify_mechanism(ask, expert, candidate, root)["passed"])
            report["checks"][0]["expert"][-1].pop("quote")
            result = verify_mechanism(ask, expert, candidate, root)
            self.assertTrue(result["passed"], result)
            library = result["checks"][0]["expert"][-1]
            self.assertEqual(library["quote"], "mma.sync;")
            self.assertEqual(library["sha256"], hashlib.sha256(b"mma.sync;\n").hexdigest())
            report["checks"][0]["expert"][-1]["path"] = "../library.cuh"
            self.assertFalse(verify_mechanism(ask, expert, candidate, root)["passed"])


if __name__ == "__main__":
    unittest.main()
