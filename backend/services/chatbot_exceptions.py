"""Shared exception taxonomy for AURA's LLM providers.

Kept in its own module (rather than defined in chatbot_llm.py, which is where
they used to live) so that the local Ollama provider (chatbot_local_llm.py)
can raise/import them without a circular import: chatbot_llm.py dispatches to
chatbot_local_llm.py, so chatbot_local_llm.py cannot import back from
chatbot_llm.py. chatbot_llm.py re-exports these names (`from
backend.services.chatbot_exceptions import *`-equivalent below), so existing
call sites/tests that reference e.g. chatbot_llm.ChatbotProviderError keep
working unchanged.

Every message here must stay safe to log/display: never include an API key,
a full request/response body, or other provider internals.
"""

from __future__ import annotations


class ChatbotNotConfiguredError(RuntimeError):
    """No usable provider is set up for the current CHATBOT_PROVIDER mode
    (e.g. OpenRouter mode with no OPENROUTER_API_KEY, or auto mode with
    neither a reachable local model nor an OpenRouter key)."""


class ChatbotProviderError(RuntimeError):
    """The provider call itself failed: bad/invalid key, unreachable host,
    malformed or empty response, or any other failure not covered by a more
    specific subclass below. The base class is kept (rather than renamed) so
    existing callers that catch ChatbotProviderError still catch the more
    specific ones too."""


class ChatbotTimeoutError(ChatbotProviderError):
    """The provider did not respond within its configured timeout."""


class ChatbotRateLimitedError(ChatbotProviderError):
    """The provider returned HTTP 429 (too many requests). OpenRouter-specific
    in practice (a local Ollama server has no rate limiting), but kept generic
    in case a future provider needs it too."""
