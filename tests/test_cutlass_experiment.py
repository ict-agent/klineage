from __future__ import annotations

import ast
import unittest
from pathlib import Path


class CutlassExperimentContractTests(unittest.TestCase):
    def test_init_script_only_declares_inputs_and_calls_init(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/cutlass_gemm/run_init.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "init"
        ]
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

        self.assertEqual(len(calls), 1)
        self.assertFalse(
            imports
            & {
                "argparse",
                "subprocess",
                "DockerGemmEvaluator",
                "CodexRunner",
                "run_lineage",
            }
        )


if __name__ == "__main__":
    unittest.main()
