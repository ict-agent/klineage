"""Check baseline ordering, model selection and experiment evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_BASELINE = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(_BASELINE))
import run as runner
import _process as process


class BaselineRunTests(unittest.TestCase):
    def test_capacity_is_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            trace.write_text(json.dumps({
                "type": "turn.failed", "error": {
                    "message": "Selected model is at capacity. Please try a different model.",
                },
            }) + "\n")
            self.assertTrue(process._capacity_error(trace))
            trace.write_text(json.dumps({"type": "error", "message": "Invalid model"}) + "\n")
            self.assertFalse(process._capacity_error(trace))

    def test_resume_keeps_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            state = {
                "status": "failed", "started_at": 100, "deadline": 43_300,
                "session": "old-session", "turns": 1, "error": "capacity",
            }
            path.write_text(json.dumps(state))
            with patch.object(runner.time, "time", return_value=200):
                resumed = runner._resume_state(path)
            self.assertEqual(resumed["started_at"], 100)
            self.assertEqual(resumed["deadline"], 43_300)
            self.assertEqual(resumed["session"], "old-session")
            self.assertEqual(resumed["turns"], 1)

    def test_resume_continues_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output" / "KDA" / "astra" / "gemm"
            output.mkdir(parents=True)
            state_path = output / "status.json"
            state_path.write_text(json.dumps({
                "status": "failed", "started_at": 100, "deadline": 43_300,
                "session": "old-session", "turns": 1,
            }))

            def execute(command, work, trace, timeout, prompt):
                self.assertEqual(command[-2:], ["old-session", "-"])
                self.assertEqual(trace.name, "0002.jsonl")
                self.assertEqual(timeout, 42_800)
                trace.parent.mkdir(parents=True)
                trace.write_text('{"type":"turn.completed"}\n')
                clock.return_value = 43_000
                return 0

            with (
                patch.object(runner, "_OUTPUT", root / "output"),
                patch.object(runner, "_WORK", root / "work"),
                patch.object(runner.time, "time", return_value=200) as clock,
                patch.object(runner, "_execute", side_effect=execute),
                patch.object(runner, "_save_rollout"),
                patch.object(runner, "_finish", return_value={"status": "passed"}),
                patch.object(runner, "_archive"),
                patch.object(runner, "_publish"),
            ):
                runner._run_job("KDA", "astra", "gemm", runner._Existing.RESUME)
            state = json.loads(state_path.read_text())
            self.assertEqual(state["deadline"], 43_300)
            self.assertEqual(state["started_at"], 100)
            self.assertEqual(state["turns"], 2)
            self.assertEqual(state["status"], "validated")

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
    def test_serializes_evaluations(self):
        script = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import evaluate
import _process
root, name = Path(sys.argv[2]), sys.argv[3]
_process._GPU_LOCK_PATH = root / 'gpu.lock'

def measure(args):
    entered = time.monotonic()
    time.sleep(0.3)
    return {'status': 'passed', 'entered': entered, 'exited': time.monotonic()}

evaluate._evaluate = measure
(root / (name + '.ready')).touch()
deadline = time.monotonic() + 10
while len(list(root.glob('*.ready'))) != 2:
    if time.monotonic() >= deadline:
        raise RuntimeError('second evaluator did not start')
    time.sleep(0.01)
sys.argv = ['evaluate', '--problem', 'gemm', '--reference', str(root / 'reference.py'),
            '--output', str(root / (name + '.json'))]
evaluate._main()
"""
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            workers = [subprocess.Popen(
                [sys.executable, "-c", script, str(_BASELINE), temporary, name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=environment,
            ) for name in ("left", "right")]
            try:
                for worker in workers:
                    _, error = worker.communicate(timeout=20)
                    self.assertEqual(worker.returncode, 0, error)
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.wait()
            rows = sorted(
                (json.loads((Path(temporary) / f"{name}.json").read_text())
                 for name in ("left", "right")),
                key=lambda row: row["entered"],
            )
            self.assertLessEqual(rows[0]["exited"], rows[1]["entered"])

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
