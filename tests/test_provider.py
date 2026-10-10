import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.harness.provider import (
    AUTO_COMPACT_TOKENS,
    CONTEXT_WINDOW,
    Provider,
    resolve_provider,
    runner_class,
    usage_proxy,
)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.root)}))

    def test_named_credentials_only(self):
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "unrelated"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TEST_KEY"):
                resolve_provider("model", "https://example.test/v1", key_env="TEST_KEY")
            explicit = resolve_provider(
                "model", "https://example.test/v1", key_env="TEST_KEY", api_key="explicit",
            )
            self.assertEqual(explicit.key, "explicit")
        with patch.dict(os.environ, {"TEST_KEY": "environment"}):
            provider = resolve_provider("model", "https://example.test/v1", key_env="TEST_KEY")
            self.assertEqual(provider.key, "environment")

    def test_proxy_is_optional(self):
        provider = Provider("model", "https://example.test/v1", "TEST_KEY", "private")
        with patch("klineage.harness.provider.UsageProxy") as proxy:
            with usage_proxy(provider) as selected:
                self.assertIs(selected, provider)
            proxy.assert_not_called()

    def test_key_not_in_repr(self):
        provider = Provider("model", "https://example.test/v1", "TEST_KEY", "private-value")
        self.assertNotIn("private-value", repr(provider))

    def test_proxy_keeps_environment(self):
        provider = Provider("model", "https://example.test/v1", "TEST_KEY", "private-value")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"TEST_KEY": "existing"}):
            with usage_proxy(provider, Path(temporary)):
                self.assertEqual(os.environ["TEST_KEY"], "existing")
            self.assertEqual(os.environ["TEST_KEY"], "existing")

    def test_catalog_symlink_rejected(self):
        provider = Provider("DeepSeek-V4.1-Flash", "https://example.test/v1", "TEST_KEY", "secret")
        runner_type = runner_class(provider, "high")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            runner_type(work, codex_bin="/bin/true")
            catalog = work / ".klineage/model.json"
            catalog.unlink()
            secret = root / "host-file"
            secret.write_text("host content")
            catalog.symlink_to(secret)
            with self.assertRaises(OSError):
                runner_type(work, codex_bin="/bin/true")
            self.assertEqual(secret.read_text(), "host content")

    def test_provider_model_catalog(self):
        provider = Provider("DeepSeek-V4.1-Flash", "https://example.test/v1", "TEST_KEY", "secret")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = runner_class(provider, "high")(root, codex_bin="/bin/true")
            command = runner.command(root / "reply.txt")
            overrides = dict(command[index + 1].split("=", 1) for index, arg in enumerate(command) if arg == "-c")
            path = Path(json.loads(overrides["model_catalog_json"]))
            catalog = path.read_text()
            model, = json.loads(catalog)["models"]
            self.assertEqual(model["slug"], provider.model)
            self.assertEqual(model["context_window"], CONTEXT_WINDOW)
            self.assertEqual(model["max_context_window"], CONTEXT_WINDOW)
            self.assertNotIn(provider.key, catalog)

    def test_model_context_on_resume(self):
        provider = Provider("DeepSeek-V4.1-Flash", "https://example.test/v1", "TEST_KEY", "unused")
        runner_type = runner_class(provider, "high")
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"KLINEAGE_MODEL": "environment-model"}),
        ):
            root = Path(temporary)
            for resume_id in (None, "recorded-session"):
                with self.subTest(resume_id=resume_id):
                    runner = runner_type(root, codex_bin="/bin/true", resume_id=resume_id)
                    command = runner.command(root / "reply.txt")
                    global_args = command[:command.index("exec")]
                    self.assertIn(f"model_context_window={CONTEXT_WINDOW}", global_args)
                    self.assertIn(f"model_auto_compact_token_limit={AUTO_COMPACT_TOKENS}", global_args)
                    self.assertEqual(command.count("--model"), 1)
                    self.assertEqual(command[command.index("--model") + 1], provider.model)
                    self.assertEqual("resume" in command, resume_id is not None)
                    subcommand = "resume" if resume_id else "exec"
                    self.assertGreater(command.index("--model"), command.index(subcommand))

    def test_model_override_catalog(self):
        provider = Provider("provider-model", "https://example.test/v1", "TEST_KEY", "unused")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = runner_class(provider, "high")(
                root, codex_bin="/bin/true", model="explicit-model",
            )
            command = runner.command(root / "reply.txt")
            self.assertEqual(command.count("--model"), 1)
            self.assertEqual(command[command.index("--model") + 1], "explicit-model")
            model, = json.loads(runner.model_catalog.read_text())["models"]
            self.assertEqual(model["slug"], "explicit-model")
            self.assertEqual(model["display_name"], "explicit-model")

    def test_campaign_context_overrides(self):
        (self.root / "config.toml").write_text(
            'model_context_window = 65536\n'
            'model_auto_compact_token_limit = 60000\n'
            'model_catalog_json = "host-models.json"\n'
        )
        provider = Provider("campaign-model", "https://example.test/v1", "TEST_KEY", "secret")
        runner = runner_class(provider, "high")(self.root / "work", codex_bin="/bin/true")
        command = runner.command(self.root / "reply.txt")
        overrides = dict(
            command[index + 1].split("=", 1)
            for index, value in enumerate(command) if value == "-c"
        )
        self.assertEqual(int(overrides["model_context_window"]), CONTEXT_WINDOW)
        self.assertEqual(int(overrides["model_auto_compact_token_limit"]), AUTO_COMPACT_TOKENS)
        self.assertEqual(Path(json.loads(overrides["model_catalog_json"])), runner.model_catalog)
