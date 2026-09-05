"""Check baseline ordering, model selection and experiment evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_BASELINE = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(_BASELINE))
import run as runner


class BaselineRunTests(unittest.TestCase):
    def test_suite_order(self):
        jobs = list(runner._jobs())
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len(set(jobs)), 30)
        self.assertEqual(jobs[0], ("KDA", "astra", "gemm"))
        self.assertTrue(all(model == "astra" for _, model, _ in jobs[:15]))
        self.assertTrue(all(model == "luna" for _, model, _ in jobs[15:]))
        self.assertEqual(runner._BUDGET_SECONDS, 43_200)

    def test_model_and_resume(self):
        command = runner._codex_command("luna", Path("final.txt"), "session-id")
        self.assertEqual(command[:3], ["codex", "exec", "resume"])
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn('model_reasoning_effort="max"', command)
        self.assertEqual(command[-2:], ["session-id", "-"])
        command = runner._codex_command("astra", Path("final.txt"))
        self.assertIn("gpt-6-astra", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)

    def test_reject_reference_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            solution = work / "runs" / "seed" / "solution"
            solution.mkdir(parents=True)
            (solution / "kernel.py").write_text("from reference import torch_ref\n")
            (work / "runs" / "seed.json").write_text(json.dumps({
                "status": "passed", "candidate": {"median_ms": 0.1},
            }))
            with self.assertRaisesRegex(RuntimeError, "no passing"):
                runner._best_candidate(work)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class BaselineCheckTests(unittest.TestCase):
    def test_topk_rejects_duplicates(self):
        import torch
        import evaluate

        values = torch.tensor([[3.0, 3.0, 1.0]])
        expected = torch.topk(values, 2)
        with self.assertRaisesRegex(AssertionError, "repeats"):
            evaluate._check_topk(torch.tensor([[0, 0]], dtype=torch.int32), expected, values)
        evaluate._check_topk(torch.tensor([[1, 0]], dtype=torch.int32), expected, values)

    def test_source_snapshot(self):
        import evaluate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "kernel.py").write_text("original\n")
            hashes = evaluate._snapshot(source, root / "frozen")
            (source / "kernel.py").write_text("changed\n")
            self.assertEqual((root / "frozen" / "kernel.py").read_text(), "original\n")
            self.assertEqual(len(hashes["kernel.py"]), 64)


if __name__ == "__main__":
    unittest.main()
