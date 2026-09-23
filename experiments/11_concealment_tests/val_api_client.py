"""Minimal OpenAI-chat-completions-compatible client for the VAL API
(credentials in `.env` at the repo root: VAL_API_KEY, VAL_API_BASE_URL, VAL_MODEL_ID).
No `openai` package is installed in this environment -- uses `requests` directly.

Connectivity confirmed live before any real judge/paraphrase call this session: a
single-token health-check ('Reply with exactly one word: OK') returned HTTP 200 with
the expected content, using `VAL_MODEL_ID=openai-gpt-5.4` at `VAL_API_BASE_URL`.

Every call this module makes should be persisted by the CALLER (raw request +ge raw
response), per the task's explicit non-negotiable persistence requirement -- this
module itself does not silently persist anything; it returns the full raw response
dict so the caller can decide where it goes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / '.env'

_ENV: dict[str, str] | None = None  # loaded lazily -- see _get_env()


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        env[k] = v
    return env


def _get_env() -> dict[str, str]:
    """Loads and caches .env on first use, not at import time -- importing this
    module (directly, or transitively through a module that only needs its pure
    helpers like extract_text/extract_usage) must not require .env to exist. .env is
    gitignored and never committed (correctly), so it is absent in CI and any fresh
    checkout; only an actual chat_completion() call needs real credentials, and only
    that call site should fail without them."""
    global _ENV
    if _ENV is None:
        _ENV = _load_env()
    return _ENV


def chat_completion(messages: list[dict[str, str]], *, max_tokens: int = 800,
                     temperature: float = 0.0, timeout: int = 60) -> dict[str, Any]:
    """One chat-completions call. Returns the FULL raw parsed JSON response (not just
    the extracted text) so callers can persist everything, including token usage."""
    env = _get_env()
    url = env['VAL_API_BASE_URL'].rstrip('/') + '/chat/completions'
    payload = {'model': env['VAL_MODEL_ID'], 'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature}
    response = requests.post(
        url, headers={'Authorization': f'Bearer {env["VAL_API_KEY"]}', 'Content-Type': 'application/json'},
        json=payload, timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_text(raw_response: dict[str, Any]) -> str:
    return raw_response['choices'][0]['message']['content']


def extract_usage(raw_response: dict[str, Any]) -> dict[str, Any]:
    return raw_response.get('usage', {})
