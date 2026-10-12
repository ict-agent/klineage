from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.harness import provider


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.home)}, clear=True))

    def config(self, text):
        (self.home / "config.toml").write_text(text)

    def auth(self, value):
        (self.home / "auth.json").write_text(json.dumps(value))

    def test_empty_home(self):
        self.assertEqual(provider.model_settings(), {
            "model": None, "reasoning_effort": None, "api_base_url": None,
            "api_key_env": None, "api_key": None, "http_headers": {}, "options": {},
        })

    def test_model_options(self):
        self.config('''model = "configured-model"
model_reasoning_effort = "high"
model_context_window = 100000
model_auto_compact_token_limit = 90000
service_tier = "priority"
model_reasoning_summary = "concise"
model_verbosity = "low"
model_catalog_json = "catalog.json"
developer_instructions = "do not inherit"
''')
        settings = provider.model_settings()
        self.assertEqual(settings["model"], "configured-model")
        self.assertEqual(settings["reasoning_effort"], "high")
        self.assertEqual(settings["options"], {
            "model_context_window": 100000, "model_auto_compact_token_limit": 90000,
            "service_tier": "priority", "model_reasoning_summary": "concise",
            "model_verbosity": "low", "model_catalog_json": str(self.home / "catalog.json"),
        })

    def test_custom_provider(self):
        self.config('''model = "configured"
model_provider = "gateway"
[model_providers.gateway]
base_url = "https://provider.test/v1"
env_key = "PROVIDER_KEY"
wire_api = "responses"
http_headers = {"X-Static" = "static-private"}
env_http_headers = {"X-Secret" = "HEADER_KEY", "X-Optional" = "ABSENT"}
''')
        os.environ.update(PROVIDER_KEY="provider-secret", HEADER_KEY="header-secret")
        settings = provider.model_settings()
        self.assertEqual(settings["api_base_url"], "https://provider.test/v1")
        self.assertEqual(settings["api_key_env"], "PROVIDER_KEY")
        self.assertEqual(settings["api_key"], "provider-secret")
        self.assertEqual(settings["http_headers"], {"X-Static": "static-private", "X-Secret": "header-secret"})
        self.assertFalse(settings["options"])

    def test_explicit_and_env_precedence(self):
        self.config('''model = "configured"
model_reasoning_effort = "low"
model_provider = "gateway"
[model_providers.gateway]
base_url = "https://configured.test/v1"
env_key = "OLD_KEY"
http_headers = {"X-Private" = "old-header"}
''')
        os.environ.update(KLINEAGE_MODEL="environment", KLINEAGE_API_BASE_URL="https://environment.test/v1",
                          KLINEAGE_API_KEY_ENV="NEW_KEY", NEW_KEY="environment-key", OLD_KEY="old-key")
        settings = provider.model_settings(model="explicit", reasoning_effort="high", api_key="explicit-key")
        self.assertEqual(settings["model"], "explicit")
        self.assertEqual(settings["reasoning_effort"], "high")
        self.assertEqual(settings["api_base_url"], "https://environment.test/v1")
        self.assertEqual(settings["api_key_env"], "NEW_KEY")
        self.assertEqual(settings["api_key"], "explicit-key")
        self.assertEqual(settings["http_headers"], {})

    def test_new_endpoint_no_secrets(self):
        self.config('''model = "configured"
model_provider = "gateway"
[model_providers.gateway]
base_url = "https://configured.test/v1"
experimental_bearer_token = "old-token"
http_headers = {"X-Private" = "old-header"}
''')
        self.auth({"tokens": {"access_token": "oauth-secret", "account_id": "account"}})
        settings = provider.model_settings(api_base_url="https://new.test/v1")
        self.assertIsNone(settings["api_key"])
        self.assertEqual(settings["http_headers"], {})
        self.assertEqual(settings["model"], "configured")

    def test_same_endpoint_keeps_token(self):
        self.config('''model_provider = "gateway"
[model_providers.gateway]
base_url = "https://configured.test/v1"
experimental_bearer_token = "configured-token"
''')
        settings = provider.model_settings(api_base_url="https://configured.test/v1/")
        self.assertEqual(settings["api_key"], "configured-token")

    def test_header_only_auth(self):
        self.config('''model_provider = "gateway"
[model_providers.gateway]
base_url = "https://configured.test/v1"
env_http_headers = {"api-key" = "HEADER_KEY"}
''')
        os.environ.update(HEADER_KEY="header-secret", AIPING_API_KEY="unrelated-secret")
        settings = provider.model_settings()
        self.assertIsNone(settings["api_key"])
        self.assertEqual(settings["http_headers"], {"api-key": "header-secret"})

    def test_openai_environment_key(self):
        os.environ["OPENAI_API_KEY"] = "environment-key"
        self.auth({"tokens": {"access_token": "oauth-secret", "account_id": "account"}})
        settings = provider.model_settings()
        self.assertEqual(settings["api_base_url"], "https://api.openai.com/v1")
        self.assertEqual(settings["api_key"], "environment-key")
        self.assertEqual(settings["http_headers"], {})

    def test_chatgpt_cached_tokens(self):
        self.config('model = "configured"\nmodel_reasoning_effort = "medium"\n')
        self.auth({"tokens": {"access_token": "cached-access", "account_id": "workspace", "refresh_token": "never-forward"}})
        settings = provider.model_settings()
        self.assertEqual(settings["api_base_url"], "https://chatgpt.com/backend-api/codex")
        self.assertEqual(settings["api_key"], "cached-access")
        self.assertEqual(settings["http_headers"], {"ChatGPT-Account-Id": "workspace"})
        self.assertNotIn("never-forward", repr(settings))

    def test_explicit_endpoint_blocks_oauth(self):
        self.auth({"tokens": {"access_token": "cached-access", "account_id": "workspace"}})
        settings = provider.model_settings(api_base_url="https://another.test/v1")
        self.assertIsNone(settings["api_key"])
        self.assertEqual(settings["http_headers"], {})

    def test_unsupported_provider_auth(self):
        for field in ('query_params = {"api-version" = "v1"}', 'auth = {command = "secret-helper"}',
                      'gateway_oauth = {client_id = "id"}', 'wire_api = "chat"'):
            with self.subTest(field=field):
                self.config('model_provider = "gateway"\n[model_providers.gateway]\nbase_url = "https://provider.test/v1"\n' + field)
                with self.assertRaises(ValueError):
                    provider.model_settings()
