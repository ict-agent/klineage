"""Model configuration and provider credentials for agent sessions."""

from __future__ import annotations

import json
import os
import tomllib
from enum import StrEnum
from pathlib import Path

OPENAI_BASE_URL = "https://api.openai.com/v1"
CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_KEY_ENV = "AIPING_API_KEY"
ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
MODEL_OPTIONS = (
    "model_context_window", "model_auto_compact_token_limit", "service_tier",
    "model_reasoning_summary", "model_verbosity", "model_catalog_json",
)
UNSUPPORTED_PROVIDER_FIELDS = (
    "auth", "gateway_oauth", "aws", "query_params", "model_catalog_url",
)


def codex_credentials(home: Path, store: str | None) -> dict:
    """Read host credentials without logging in or refreshing OAuth tokens."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return {"api_key": key, "api_key_env": "OPENAI_API_KEY", "api_base_url": OPENAI_BASE_URL}
    if store == "keyring":
        raise ValueError("Codex keyring credentials are unsupported; provide an explicit API key")
    path = home / "auth.json"
    if not path.exists():
        return {}
    auth = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(auth, dict):
        raise ValueError("Codex auth.json must contain an object")
    if auth.get("OPENAI_API_KEY"):
        return {"api_key": auth["OPENAI_API_KEY"], "api_key_env": "OPENAI_API_KEY", "api_base_url": OPENAI_BASE_URL}
    tokens = auth.get("tokens") or {}
    if not isinstance(tokens, dict):
        raise ValueError("Codex auth tokens must contain an object")
    if tokens.get("access_token"):
        if not tokens.get("account_id"):
            raise ValueError("Codex ChatGPT credentials require an account_id")
        return {
            "api_key": tokens["access_token"], "api_base_url": CHATGPT_BASE_URL,
            "http_headers": {"ChatGPT-Account-Id": tokens["account_id"]},
        }
    if auth:
        raise ValueError("Codex auth cache has no supported API key or ChatGPT access token")
    return {}


def provider_headers(provider: dict) -> dict[str, str]:
    """Resolve static and environment-backed headers on the host."""
    result = {}
    for option in ("http_headers", "env_http_headers"):
        headers = provider.get(option, {})
        if not isinstance(headers, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in headers.items()
        ):
            raise ValueError(f"provider {option} must map strings to strings")
        for name, value in headers.items():
            if option == "env_http_headers":
                value = os.environ.get(value)
            if value is not None:
                result = {
                    key: item for key, item in result.items() if key.lower() != name.lower()
                }
                result[name] = value
    return result


def model_settings(
    *, model=None, reasoning_effort=None, api_base_url=None, api_key_env=None,
    api_key=None,
) -> dict:
    """Resolve model settings and credentials for the provider bridge.

    Explicit arguments override KLINEAGE environment settings and Codex user
    configuration. A different endpoint cannot inherit stored provider secrets.
    OAuth uses cached access tokens only; host Codex owns their renewal.
    """
    model = model if model is not None else os.environ.get("KLINEAGE_MODEL")
    endpoint = api_base_url if api_base_url is not None else os.environ.get("KLINEAGE_API_BASE_URL") or None
    key_env = api_key_env if api_key_env is not None else os.environ.get("KLINEAGE_API_KEY_ENV") or None
    for option, value in (("api_base_url", endpoint), ("api_key_env", key_env)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{option} must be nonempty text")
    if endpoint is not None:
        endpoint = endpoint.strip()
    supplied_key = api_key if api_key is not None else os.environ.get(key_env or DEFAULT_KEY_ENV)
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    config = {}
    path = home / "config.toml"
    if path.exists():
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    if config.get("profile"):
        raise ValueError("Codex profile selection is unsupported; configure root model settings")

    result = {
        "model": model if model is not None else config.get("model"),
        "reasoning_effort": reasoning_effort if reasoning_effort is not None else config.get("model_reasoning_effort"),
        "api_base_url": endpoint, "api_key_env": key_env, "api_key": api_key,
        "http_headers": {}, "options": {name: config[name] for name in MODEL_OPTIONS if name in config},
    }
    catalog = result["options"].get("model_catalog_json")
    if catalog is not None:
        result["options"]["model_catalog_json"] = str((home / Path(catalog).expanduser()).absolute())
    name = config.get("model_provider", "openai")
    providers = config.get("model_providers", {})
    if not isinstance(providers, dict) or not isinstance(name, str):
        raise ValueError("Codex model_provider and model_providers have invalid types")
    provider = providers.get(name, {})
    if not isinstance(provider, dict):
        raise ValueError("Codex model provider must contain a table")
    credentials = {}
    if (
        name == "openai" and not provider.get("base_url")
        and api_key is None and key_env is None and not provider.get("env_key")
        and "experimental_bearer_token" not in provider
        and (endpoint is None or endpoint.rstrip("/") in (OPENAI_BASE_URL, CHATGPT_BASE_URL))
    ):
        credentials = codex_credentials(home, config.get("cli_auth_credentials_store"))
    configured_url = provider.get("base_url") or credentials.get("api_base_url") or (OPENAI_BASE_URL if name == "openai" else None)
    inherit = endpoint is None or (
        isinstance(endpoint, str) and isinstance(configured_url, str)
        and endpoint.rstrip("/") == configured_url.rstrip("/")
    )
    if inherit:
        for option in UNSUPPORTED_PROVIDER_FIELDS:
            if provider.get(option):
                raise ValueError(f"Codex provider {option} is unsupported")
        if provider.get("wire_api", "responses") != "responses":
            raise ValueError("Codex provider wire_api must be responses")
        if name != "openai" and configured_url is None:
            raise ValueError("Codex custom provider requires base_url")
        result["api_base_url"] = endpoint or provider.get("base_url")
        result["api_key_env"] = key_env or provider.get("env_key")
        result["http_headers"] = provider_headers(provider)
        if api_key is None:
            selected = result["api_key_env"]
            result["api_key"] = os.environ.get(selected) if selected else provider.get("experimental_bearer_token")
        if result["api_key"] is None and result["api_key_env"] is None and (name == "openai" or provider.get("requires_openai_auth")):
            credentials = credentials or codex_credentials(home, config.get("cli_auth_credentials_store"))
            result["api_key"] = credentials.get("api_key")
            result["api_key_env"] = result["api_key_env"] or credentials.get("api_key_env")
            result["api_base_url"] = result["api_base_url"] or credentials.get("api_base_url")
            result["http_headers"] = {**result["http_headers"], **credentials.get("http_headers", {})}
        if result["api_base_url"] is None and (result["api_key"] is not None or result["api_key_env"] is not None):
            result["api_base_url"] = OPENAI_BASE_URL
    else:
        result["api_key_env"] = key_env or DEFAULT_KEY_ENV
        result["api_key"] = supplied_key

    for option in ("model", "api_base_url", "api_key_env", "api_key"):
        value = result[option]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{option} must be nonempty text")
    if result["api_key_env"] is not None and not result["api_key_env"].isidentifier():
        raise ValueError("api_key_env must name an environment variable")
    if result["api_base_url"] is not None:
        result["api_base_url"] = result["api_base_url"].strip()
    return result


class AnthropicAuth(StrEnum):
    API_KEY = "api-key"
    BEARER = "bearer"


def anthropic_settings(
    *, model=None, reasoning_effort=None, api_base_url=None, api_key_env=None,
    api_key=None, auth_mode=None,
) -> dict:
    """Resolve Messages settings without reading host CLI configuration."""
    endpoint = api_base_url if api_base_url is not None else os.environ.get("KLINEAGE_API_BASE_URL")
    configured = os.environ.get("ANTHROPIC_BASE_URL") or ANTHROPIC_API_URL
    endpoint = endpoint if endpoint is not None else configured
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("api_base_url must be nonempty text")
    endpoint = endpoint.strip()
    key_env = api_key_env if api_key_env is not None else os.environ.get("KLINEAGE_API_KEY_ENV")
    if not key_env:
        if endpoint.rstrip("/") != configured.rstrip("/") and api_key is None:
            raise ValueError("a different endpoint requires an explicit api_key or api_key_env")
        key_env = ANTHROPIC_TOKEN_ENV if os.environ.get(ANTHROPIC_TOKEN_ENV) else ANTHROPIC_KEY_ENV
    if not isinstance(key_env, str) or not key_env.isidentifier():
        raise ValueError("api_key_env must name an environment variable")
    key = api_key if api_key is not None else os.environ.get(key_env)
    if not key:
        raise ValueError(f"No API key: set {key_env} on the host")
    selected = auth_mode or os.environ.get("KLINEAGE_AUTH_MODE")
    auth = AnthropicAuth(selected) if selected else (
        AnthropicAuth.BEARER if key_env == ANTHROPIC_TOKEN_ENV else AnthropicAuth.API_KEY
    )
    model = model if model is not None else os.environ.get("KLINEAGE_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    for name, value in (("api_base_url", endpoint), ("api_key", key), ("model", model)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be nonempty text")
    headers = {"x-api-key": key} if auth == AnthropicAuth.API_KEY else {"authorization": "Bearer " + key}
    return {
        "model": model, "reasoning_effort": reasoning_effort,
        "api_base_url": endpoint.strip(), "api_key_env": key_env, "api_key": key,
        "http_headers": headers, "options": {},
    }
