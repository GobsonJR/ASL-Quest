"""LLM provider boundary for AURA, the ASL-Quest project assistant.

Two providers are supported, selected by CHATBOT_PROVIDER (see
backend/settings.py):

- "openrouter" (default, unchanged from before this module supported more
  than one provider): OpenRouter (https://openrouter.ai), reached via its
  OpenAI-compatible chat completions endpoint. Requires OPENROUTER_API_KEY.
- "local": a local Ollama server (https://ollama.com), reached via its native
  chat API -- see chatbot_local_llm.py. Works fully offline; never contacts
  OpenRouter; never requires OPENROUTER_API_KEY.
- "auto": prefer local if reachable, otherwise fall back to OpenRouter if
  it's configured, otherwise raise ChatbotNotConfiguredError.

No LLM SDK dependency exists anywhere in this project (checked
requirements.txt), so both providers are called directly over httpx —
already a required dependency — rather than introducing a new one.
generate_reply()/is_configured()/get_status_info() are the ONLY things the
router and knowledge layer call; they never need to know which provider (or
which HTTP shape) is behind them.

Tests must never hit the network: they patch generate_reply (or the
underlying httpx calls in this module / chatbot_local_llm.py) instead of
relying on a real OPENROUTER_API_KEY or a running Ollama server.

Every exception here carries a message safe to log — none of them ever
include the API key, and the router never forwards these messages verbatim
to the frontend (it maps each type to its own fixed, user-facing string),
except ChatbotNotConfiguredError's own message, which is written to be
safe to show as-is (see raise sites below).
"""

from __future__ import annotations

import httpx

from backend.services import chatbot_local_llm
from backend.services.chatbot_exceptions import (
    ChatbotNotConfiguredError,
    ChatbotProviderError,
    ChatbotRateLimitedError,
    ChatbotTimeoutError,
)
from backend.settings import (
    get_chatbot_api_base_url,
    get_chatbot_api_key,
    get_chatbot_local_base_url,
    get_chatbot_local_model,
    get_chatbot_model,
    get_chatbot_provider,
)

# Re-exported for backward compatibility: existing call sites (router, tests)
# reference these as chatbot_llm.ChatbotNotConfiguredError etc.
__all__ = [
    "ChatbotNotConfiguredError",
    "ChatbotProviderError",
    "ChatbotRateLimitedError",
    "ChatbotTimeoutError",
    "generate_reply",
    "is_configured",
    "get_status_info",
]

MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 30.0
# Sent to OpenRouter per https://openrouter.ai/docs — attributes requests to this
# project in OpenRouter's own dashboards/rankings. Not required for the API to
# work, carries no secret, and OpenRouter never echoes it back to the caller.
APP_REFERER = "https://github.com/GobsonJR/ASL-Quest"
APP_TITLE = "AURA (ASL-Quest Assistant)"


def _openrouter_configured() -> bool:
    return bool(get_chatbot_api_key())


def is_configured() -> bool:
    """Whether the CURRENTLY SELECTED provider has its required settings —
    a static/config-only check, never a network call (in particular, never
    probes OpenRouter, and never probes local Ollama either — that live
    reachability check is local_available in get_status_info(), kept
    separate on purpose)."""
    provider = get_chatbot_provider()
    if provider == "openrouter":
        return _openrouter_configured()
    # "local" always has a usable base_url/model via defaults; "auto" is
    # "configured" the moment either path could work, and local always can.
    return True


def get_status_info() -> dict:
    """Everything GET /chatbot/status needs, in one call. Never includes the
    API key. Only ever probes local Ollama's reachability (never OpenRouter,
    and never when provider="openrouter" — that mode has no local model to
    check) since a cheap localhost probe is safe to run on every status poll,
    while OpenRouter's reachability is only ever discovered by an actual
    message send."""
    provider = get_chatbot_provider()
    checks_local = provider in ("local", "auto")
    local_available = chatbot_local_llm.is_available(get_chatbot_local_base_url()) if checks_local else False
    return {
        "provider": provider,
        "configured": is_configured(),
        "local_available": local_available,
        "local_model": get_chatbot_local_model() if checks_local else None,
        # AURA's project knowledge base (backend/services/chatbot_knowledge.py)
        # is a static, always-available module — never behind a network call —
        # so it's always true. Lets the frontend show a "still works, offline"
        # indicator instead of implying AURA is broken when no LLM provider is
        # reachable (see chatbot_knowledge.fallback_answer, used in that case).
        "knowledge_base_available": True,
        "online_required": False,
    }


def generate_reply(system_prompt: str, history: list[dict[str, str]]) -> str:
    """Sends one request to the configured LLM provider and returns its text reply.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts.
    Always raises one of the exceptions above instead of ever returning a
    partial/garbage answer.
    """
    provider = get_chatbot_provider()

    if provider == "local":
        # Explicit local mode: NEVER falls back to OpenRouter, by design —
        # this is what guarantees a true offline review mode. If Ollama is
        # unreachable this raises ChatbotProviderError/ChatbotTimeoutError
        # (from chatbot_local_llm), handled the same way the router already
        # handles any other provider failure.
        return chatbot_local_llm.generate_reply(
            system_prompt, history, base_url=get_chatbot_local_base_url(), model=get_chatbot_local_model()
        )

    if provider == "auto":
        if chatbot_local_llm.is_available(get_chatbot_local_base_url()):
            return chatbot_local_llm.generate_reply(
                system_prompt, history, base_url=get_chatbot_local_base_url(), model=get_chatbot_local_model()
            )
        if _openrouter_configured():
            return _generate_reply_openrouter(system_prompt, history)
        raise ChatbotNotConfiguredError(
            "AURA has no usable provider right now: no local Ollama server was reachable, "
            "and OPENROUTER_API_KEY is not set. Start Ollama (see README's Offline AURA "
            "setup) or set OPENROUTER_API_KEY (see .env.example)."
        )

    # "openrouter" (default) — unchanged from before CHATBOT_PROVIDER existed.
    return _generate_reply_openrouter(system_prompt, history)


def _generate_reply_openrouter(system_prompt: str, history: list[dict[str, str]]) -> str:
    api_key = get_chatbot_api_key()
    if not api_key:
        raise ChatbotNotConfiguredError(
            "OPENROUTER_API_KEY is not set. An administrator needs to configure it "
            "(see .env.example) before AURA can answer questions."
        )

    # OpenAI-compatible chat completions shape: the system prompt is just the
    # first message (role "system"), not a separate top-level field.
    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        response = httpx.post(
            get_chatbot_api_base_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": APP_REFERER,
                "X-Title": APP_TITLE,
            },
            json={
                "model": get_chatbot_model(),
                "max_tokens": MAX_TOKENS,
                "messages": messages,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        raise ChatbotTimeoutError("AURA's provider request timed out.") from exc
    except httpx.HTTPError as exc:
        # Network-level failure (DNS, connection refused, etc.) — never includes
        # the api_key (httpx doesn't embed header values in its exception text).
        raise ChatbotProviderError(f"AURA's provider request failed: {exc}") from exc

    if response.status_code == 429:
        raise ChatbotRateLimitedError("AURA's provider is rate-limiting requests right now.")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Covers an invalid/revoked API key (401/403), OpenRouter/upstream 5xx,
        # and any other non-2xx status. The status code alone is safe to log;
        # the response body is deliberately not included in case a provider
        # ever echoed headers back.
        raise ChatbotProviderError(
            f"AURA's provider returned HTTP {response.status_code}."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ChatbotProviderError("AURA's provider returned a malformed (non-JSON) response.") from exc

    try:
        choices = payload["choices"]
        text = (choices[0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatbotProviderError("AURA's provider returned an unexpected response shape.") from exc
    if not text:
        raise ChatbotProviderError("AURA's provider returned an empty response.")
    return text
