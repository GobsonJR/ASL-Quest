"""LLM provider boundary for the ASL-Quest Assistant.

No LLM SDK dependency exists anywhere in this project yet (checked
requirements.txt), so this calls the provider's REST API directly over
httpx — already a required dependency — rather than introducing a new one.
This module is the ONLY place that knows about the provider's HTTP shape;
the router and knowledge layer only ever call generate_reply().

Tests must never hit the network: they patch generate_reply (or the
underlying httpx.post) instead of relying on a real CHATBOT_API_KEY.
"""

from __future__ import annotations

import httpx

from backend.settings import get_chatbot_api_key, get_chatbot_model

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 30.0


class ChatbotNotConfiguredError(RuntimeError):
    """No CHATBOT_API_KEY is set — the assistant isn't set up yet."""


class ChatbotProviderError(RuntimeError):
    """The provider call itself failed (network, HTTP status, or bad payload)."""


def is_configured() -> bool:
    return bool(get_chatbot_api_key())


def generate_reply(system_prompt: str, history: list[dict[str, str]]) -> str:
    """Sends one request to the configured LLM provider and returns its text reply.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts, the
    same shape Anthropic's Messages API expects. Raises ChatbotNotConfiguredError
    or ChatbotProviderError instead of ever returning a partial/garbage answer.
    """
    api_key = get_chatbot_api_key()
    if not api_key:
        raise ChatbotNotConfiguredError(
            "CHATBOT_API_KEY is not set. An administrator needs to configure it "
            "(see .env.example) before the ASL-Quest Assistant can answer questions."
        )

    try:
        response = httpx.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": get_chatbot_model(),
                "max_tokens": MAX_TOKENS,
                "system": system_prompt,
                "messages": history,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise ChatbotProviderError(f"ASL-Quest Assistant provider request failed: {exc}") from exc
    except ValueError as exc:
        raise ChatbotProviderError("ASL-Quest Assistant provider returned an invalid response.") from exc

    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    ).strip()
    if not text:
        raise ChatbotProviderError("ASL-Quest Assistant provider returned an empty response.")
    return text
