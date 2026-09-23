"""LLM provider boundary for AURA, the ASL-Quest project assistant.

Provider is OpenRouter (https://openrouter.ai), reached via its OpenAI-
compatible chat completions endpoint — an OPENROUTER_API_KEY works with
OpenRouter's free-tier models, so AURA can run without an Anthropic API key.

No LLM SDK dependency exists anywhere in this project (checked
requirements.txt), so this calls the provider's REST API directly over
httpx — already a required dependency — rather than introducing a new one.
This module is the ONLY place that knows about the provider's HTTP shape;
the router and knowledge layer only ever call generate_reply()/is_configured().

Tests must never hit the network: they patch generate_reply (or the
underlying httpx.post) instead of relying on a real OPENROUTER_API_KEY.

Every exception here carries a message safe to log — none of them ever
include the API key, and the router never forwards these messages verbatim
to the frontend (it maps each type to its own fixed, user-facing string).
"""

from __future__ import annotations

import httpx

from backend.settings import get_chatbot_api_base_url, get_chatbot_api_key, get_chatbot_model

MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 30.0
# Sent to OpenRouter per https://openrouter.ai/docs — attributes requests to this
# project in OpenRouter's own dashboards/rankings. Not required for the API to
# work, carries no secret, and OpenRouter never echoes it back to the caller.
APP_REFERER = "https://github.com/GobsonJR/ASL-Quest"
APP_TITLE = "AURA (ASL-Quest Assistant)"


class ChatbotNotConfiguredError(RuntimeError):
    """No CHATBOT_API_KEY is set — the assistant isn't set up yet."""


class ChatbotProviderError(RuntimeError):
    """The provider call itself failed: bad/invalid key, malformed or empty
    response, or any other HTTP-level failure not covered by a more specific
    subclass below. The base class is kept (rather than renamed) so existing
    callers that catch ChatbotProviderError still catch the specific ones too."""


class ChatbotTimeoutError(ChatbotProviderError):
    """The provider did not respond within REQUEST_TIMEOUT_SECONDS."""


class ChatbotRateLimitedError(ChatbotProviderError):
    """The provider returned HTTP 429 (too many requests)."""


def is_configured() -> bool:
    return bool(get_chatbot_api_key())


def generate_reply(system_prompt: str, history: list[dict[str, str]]) -> str:
    """Sends one request to the configured LLM provider and returns its text reply.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts.
    Always raises one of the exceptions above instead of ever returning a
    partial/garbage answer.
    """
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
