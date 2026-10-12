from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.harness import ClaudeCodeRunner, RunnerError, create_runner, session_id
from klineage.harness.provider import AnthropicAuth, anthropic_settings

SESSION = "1b66f15c-3f5b-4e62-b4d6-029f778d39e7"
FAKE_CLAUDE = '''
import json, os, pathlib, sys, time
prompt = sys.stdin.read()
args = sys.argv[1:]
print(json.dumps({"type":"system", "subtype":"init", "session_id":"SESSION"}), flush=True)
print(json.dumps({"type":"probe", "args":args, "prompt":prompt,
                  "env":dict(os.environ), "state":os.environ["CLAUDE_CONFIG_DIR"]}), flush=True)
state = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"]) / "marker"
if "--resume" in args:
    assert state.read_text() == "saved"
state.write_text("saved")
if "TIMEOUT" in prompt:
    time.sleep(60)
if "FAIL_EXIT" in prompt:
    print("simulated CLI failure", file=sys.stderr)
    sys.exit(7)
if "NO_RESULT" in prompt:
    sys.exit(0)
result = {"type":"result", "subtype":"success", "is_error":False, "result":"done", "session_id":"SESSION"}
if "RESULT_ERROR" in prompt:
    result.update(subtype="error_during_execution", is_error=True)
if "EMPTY_RESULT" in prompt:
    result["result"] = ""
print(json.dumps(result), flush=True)
'''.replace("SESSION", SESSION)


class ClaudeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_native_environment(self):
        os.environ.update(ANTHROPIC_BASE_URL="https://gateway.test/anthropic", ANTHROPIC_MODEL="provider-model", ANTHROPIC_AUTH_TOKEN="secret")
        settings = anthropic_settings()
        self.assertEqual(settings["model"], "provider-model")
        self.assertEqual(settings["http_headers"], {"authorization": "Bearer secret"})
        self.assertEqual(settings["api_base_url"], "https://gateway.test/anthropic")

    def test_explicit_precedence(self):
        os.environ.update(KLINEAGE_MODEL="environment-model", KLINEAGE_API_BASE_URL="https://environment.test", KLINEAGE_API_KEY_ENV="CUSTOM_KEY", CUSTOM_KEY="environment-key", ANTHROPIC_API_KEY="native-key")
        settings = anthropic_settings(model="explicit", api_base_url="https://explicit.test", api_key="explicit-key", auth_mode=AnthropicAuth.BEARER)
        self.assertEqual(settings["model"], "explicit")
        self.assertEqual(settings["api_base_url"], "https://explicit.test")
        self.assertEqual(settings["http_headers"], {"authorization": "Bearer explicit-key"})

    def test_default_api_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "native-key"
        self.assertEqual(anthropic_settings()["http_headers"], {"x-api-key": "native-key"})

    def test_endpoint_secret_boundary(self):
        os.environ.update(ANTHROPIC_BASE_URL="https://old.test", ANTHROPIC_API_KEY="old-secret")
        with self.assertRaisesRegex(ValueError, "different endpoint"):
            anthropic_settings(api_base_url="https://new.test")
        with self.assertRaisesRegex(ValueError, "MISSING"):
            anthropic_settings(api_base_url="https://new.test", api_key_env="MISSING")

    def test_invalid_settings(self):
        for options in ({"api_key_env":""}, {"auth_mode":""}, {"api_key_env":"BAD-NAME"}, {"api_base_url":5}, {"api_base_url":""}, {"auth_mode":"unknown"}, {"model":""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                anthropic_settings(api_key="key", **options)


class ClaudeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.work = self.root / "work"
        self.program = self.root / "claude"
        self.program.write_text(f"#!{sys.executable}\n" + FAKE_CLAUDE)
        self.program.chmod(0o755)
        self.enterContext(patch.dict(os.environ, {
            "KLINEAGE_RUNNER":"claude-code", "KLINEAGE_MODEL":"test-model",
            "KLINEAGE_API_BASE_URL":"https://provider.test", "KLINEAGE_API_KEY_ENV":"TEST_MODEL_KEY",
            "TEST_MODEL_KEY":"host-secret", "ANTHROPIC_AUTH_TOKEN":"unrelated-secret",
            "KLINEAGE_SERVER":"",
        }))

    def runner(self, **options):
        return create_runner(self.work, claude_bin=self.program, timeout=10, **options)

    def test_command_and_resume(self):
        runner = self.runner(reasoning_effort="high")
        self.assertIsInstance(runner, ClaudeCodeRunner)
        result = runner("generate", run_id="first")
        self.assertEqual(result.final_message, "done")
        self.assertEqual(result.trace_path.parent.parent.name, "agent-runs")
        rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
        probe = next(row for row in rows if row["type"] == "probe")
        args, env = probe["args"], probe["env"]
        self.assertIn("--bare", args)
        self.assertEqual(args[args.index("--setting-sources") + 1], "")
        self.assertIn("--strict-mcp-config", args)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sandbox-provider")
        self.assertTrue(env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:"))
        self.assertNotIn("host-secret", json.dumps(probe))
        self.assertNotIn("unrelated-secret", json.dumps(probe))
        self.assertNotIn("CODEX_HOME", env)
        self.assertIn("workspace/AGENTS.md", probe["prompt"])
        self.assertEqual(session_id(result.trace_path), SESSION)
        runner.resume_id = session_id(result.trace_path)
        second = runner("continue", run_id="second")
        self.assertEqual(second.final_message, "done")
        self.assertEqual((second.trace_path.parent / "final_message.txt").read_text(), "done")

    def test_failures_retain_trace(self):
        for prompt in ("RESULT_ERROR", "NO_RESULT", "EMPTY_RESULT", "FAIL_EXIT"):
            with self.subTest(prompt=prompt), self.assertRaises(RunnerError) as caught:
                self.runner()(prompt, run_id=prompt)
            self.assertTrue(caught.exception.trace_path.is_file())
            self.assertEqual(session_id(caught.exception.trace_path), SESSION)

    def test_explicit_runner_selection(self):
        from klineage.harness import CodexRunner
        with patch.dict(os.environ, {"CODEX_HOME":str(self.root / "codex-home")}):
            runner = create_runner(self.work, runner="codex", codex_bin="/bin/true")
        self.assertIsInstance(runner, CodexRunner)
        with self.assertRaises(ValueError):
            create_runner(self.work, runner="invalid")

    def test_action_selects_claude(self):
        from klineage.action.action import Action
        with patch("klineage.harness.claude_code_runner.ClaudeCodeRunner", return_value=object()) as factory:
            action = Action("task", self.work)
        self.assertIs(action.runner, factory.return_value)
