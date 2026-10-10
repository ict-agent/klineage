"""Campaigns retain selected agent resources and server-owned acceptance."""

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from problem_fixtures import problem_spec

from klineage.harness import AgentRuntime
from scripts import server_campaign


class Runner:
    def __init__(self, workdir, **options):
        self.workdir = workdir
        self.options = options


class ServerCampaignTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_cli_agent_selection(self):
        arguments = [
            "server_campaign.py",
            "--server-python",
            sys.executable,
            "--agent-python",
            sys.executable,
            "--agent-device-node",
            "/dev/null",
            "--agent-read-only",
            str(self.root),
            "--agent-env",
            "CUDA_VISIBLE_DEVICES=0",
        ]
        with patch("sys.argv", arguments), patch.object(server_campaign, "run") as run:
            server_campaign.main()
        agent = run.call_args.args[0].agent
        self.assertEqual(agent.python, Path(sys.executable).absolute())
        self.assertEqual(tuple(agent.device_nodes), (Path("/dev/null"),))
        self.assertEqual(tuple(agent.read_only), (self.root,))
        self.assertEqual(agent.environment["CUDA_VISIBLE_DEVICES"], "0")

    def test_server_uses_endpoint(self):
        ready = {"endpoint": "http://localhost:9000", "pid": 123}
        (self.root / "server-ready.json").write_text(json.dumps(ready))
        args = argparse.Namespace(
            server_python=Path(sys.executable), backend="cuda",
            device_node=[],
        )
        with (
            patch.object(server_campaign, "runtime_environment", return_value={"PATH": "/usr/bin"}),
            patch.object(server_campaign.shutil, "which", return_value="/usr/bin/timeout"),
            patch.object(server_campaign, "remaining", return_value=30),
            patch.object(server_campaign.subprocess, "Popen") as launch,
            patch.object(server_campaign, "Client") as client,
            patch.object(server_campaign, "stop_process") as stop,
        ):
            with server_campaign.server_process(args, self.root, self.root / "problem", 30) as result:
                self.assertEqual(result, (client.return_value, ready["pid"]))
        client.assert_called_once_with(ready["endpoint"])
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--problem") + 1], str(self.root / "problem"))
        client.return_value.health.assert_called_once_with()
        self.assertEqual(set(launch.call_args.kwargs["env"]), {"PATH", "KLINEAGE_BACKEND", "PYTHONPATH"})
        stop.assert_called_once_with(launch.return_value)

    def test_runtime_state_and_prompt(self):
        # Stop at generation; exercising wiring needs no provider or device process.
        python = self.root / "python"
        python.symlink_to(sys.executable)
        agent = AgentRuntime(python=python, device_nodes=(Path("/dev/null"),))
        problem = problem_spec()
        client = Mock()
        client.endpoint = "http://evaluation.example"
        client.problem.return_value = problem
        problem_dir = self.root / "problem"
        problem_dir.mkdir()
        client.info.return_value = {
            "backend": "cuda", "problem_paths": {problem.name: str(problem_dir)},
        }
        action = Mock(prompt="task\n")
        action.run.side_effect = RuntimeError("stop-before-generation")
        args = argparse.Namespace(
            agent=agent,
            budget=1800,
            output=self.root,
            model="test-model",
            backend="cuda",
            server_python=Path(sys.executable),
            problem=self.root / "problem",
            base_url="http://provider.example",
            device_node=[],
        )

        with (
            patch.object(
                server_campaign,
                "server_process",
                return_value=nullcontext((client, 123)),
            ),
            patch.object(server_campaign, "resolve_provider", return_value=object()),
            patch.object(
                server_campaign, "usage_proxy", return_value=nullcontext(object())
            ),
            patch.object(server_campaign, "runner_class", return_value=Runner),
            patch.object(server_campaign, "Apply", return_value=action) as apply,
            patch.object(server_campaign.signal, "signal"),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "stop-before-generation"),
        ):
            server_campaign.run(args)
        self.assertIs(apply.call_args.kwargs["agent"], agent)
        self.assertIs(action.runner.options["agent"], agent)
        state = json.loads(
            next(self.root.glob("sandbox-campaign-*/status.json")).read_text()
        )
        self.assertEqual(state["agent_python"], str(agent.python))
        self.assertEqual(state["agent_device_nodes"], ["/dev/null"])
        self.assertIn(str(agent.python), action.prompt)
        self.assertIn("ncu", action.prompt)
        self.assertIn("Client.evaluate", action.prompt)
        self.assertIn("never overlap", action.prompt)


if __name__ == "__main__":
    unittest.main()
