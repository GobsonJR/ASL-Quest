"""Local LLM provider for AURA: Ollama (https://ollama.com), reached via its
native chat API (POST {base_url}/api/chat) -- NOT the OpenAI-compatible shape
OpenRouter uses (see chatbot_llm.py). This is what lets AURA answer
project questions with no internet connection: the model runs entirely on
the machine running the backend, managed by a locally-installed Ollama
server (this module never downloads model weights itself -- see README's
"Offline AURA setup" section for `ollama pull qwen3:4b`).

Uses httpx directly, same as chatbot_llm.py, for the same reason: no LLM SDK
dependency exists anywhere in this project.

Tests must never require a real Ollama server: they patch generate_reply/
is_available (or the underlying httpx calls) instead.
"""

from __future__ import annotations

import httpx

from backend.services.chatbot_exceptions import ChatbotProviderError, ChatbotTimeoutError

# Local generation on a small (4B-class) model over CPU/consumer GPU can
# legitimately take longer than a hosted API -- OpenRouter's own timeout
# (chatbot_llm.REQUEST_TIMEOUT_SECONDS) is 30s; this is deliberately more
# generous rather than surfacing spurious timeouts on slower hardware.
REQUEST_TIMEOUT_SECONDS = 60.0

# Reachability probe only -- kept short since it's used on every /chatbot/status
# poll (in "local"/"auto" mode) and to decide "auto" mode's fallback branch.
AVAILABILITY_TIMEOUT_SECONDS = 2.0


def is_available(base_url: str) -> bool:
    """Cheap reachability check: GET {base_url}/api/tags (Ollama's model-list
    endpoint -- lightweight, doesn't require the target model to be loaded).
    Never raises; any failure (connection refused, DNS, timeout, non-200)
    just means "not available right now". Used by "auto" mode's fallback
    decision and by the /chatbot/status endpoint -- never contacts anything
    but the configured local base_url (no OpenRouter request happens here)."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=AVAILABILITY_TIMEOUT_SECONDS)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def generate_reply(
    system_prompt: str,
    history: list[dict[str, str]],
    base_url: str,
    model: str,
) -> str:
    """Sends one request to a local Ollama server and returns its text reply.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts
    -- the same shape chatbot_llm.generate_reply takes, so the router/scope
    guard/system-prompt code above this never needs to know which provider is
    active. Always raises a ChatbotProviderError subclass instead of ever
    returning a partial/garbage answer (mirrors chatbot_llm.py's contract).
    """
    # Ollama's native chat API: messages is [{"role","content"}, ...] with the
    # system prompt as an ordinary first message (same convention as the
    # OpenAI-compatible shape chatbot_llm.py uses for OpenRouter) --
    # stream=False so this returns one complete JSON object, not an
    # event stream, matching the sync httpx.post used throughout this project.
    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        raise ChatbotTimeoutError("AURA's local model request timed out.") from exc
    except httpx.HTTPError as exc:
        # Connection refused (Ollama not running), DNS failure, etc.
        raise ChatbotProviderError(f"AURA's local model request failed: {exc}") from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ChatbotProviderError(
            f"AURA's local model returned HTTP {response.status_code}."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ChatbotProviderError("AURA's local model returned a malformed (non-JSON) response.") from exc

    try:
        text = (payload["message"]["content"] or "").strip()
    except (KeyError, TypeError) as exc:
        raise ChatbotProviderError("AURA's local model returned an unexpected response shape.") from exc
    if not text:
        raise ChatbotProviderError("AURA's local model returned an empty response.")
    return text
