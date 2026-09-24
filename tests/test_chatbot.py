"""Focused tests for backend/routers/chatbot.py (the ASL-Quest Assistant).

Independent of the existing app test suite: uses its own temporary SQLite
file and creates every table via Base.metadata.create_all, matching the
pattern in test_native.py / test_database_phase2.py. Nothing here touches
data/asl_quest.db.

The external LLM provider is always mocked (chatbot_llm.generate_reply is
patched, or OPENROUTER_API_KEY is left unset) -- this suite never makes a real
network call.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "chatbot-test-secret"
os.environ.pop("OPENROUTER_API_KEY", None)  # unconfigured unless a test opts in

from backend import models  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import ChatbotConversation, ChatbotFeedback, ChatbotMessage  # noqa: E402
from backend.routers.chatbot import MAX_HISTORY_MESSAGES, _build_history  # noqa: E402
from backend.services import chatbot_knowledge, chatbot_llm  # noqa: E402


class ChatbotRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def register(self, username: str) -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": f"{username}@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def create_conversation(self, token: str, title: str | None = None) -> int:
        response = self.client.post("/chatbot/conversations", json={"title": title}, headers=self.auth(token))
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    # --- Authentication -------------------------------------------------------

    def test_conversations_require_auth(self) -> None:
        self.assertEqual(self.client.get("/chatbot/conversations").status_code, 401)
        self.assertEqual(self.client.post("/chatbot/conversations", json={}).status_code, 401)
        self.assertEqual(self.client.get("/chatbot/conversations/1").status_code, 401)
        self.assertEqual(
            self.client.post("/chatbot/conversations/1/messages", json={"content": "hi"}).status_code, 401
        )

    # --- Create / retrieve conversation ----------------------------------------

    def test_create_conversation(self) -> None:
        token = self.register("create_conv_user")
        response = self.client.post("/chatbot/conversations", json={"title": "My chat"}, headers=self.auth(token))
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertEqual(payload["title"], "My chat")
        self.assertEqual(payload["messages"], [])

    def test_conversation_retrieval(self) -> None:
        token = self.register("retrieval_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="Mocked answer."):
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does the Native Sign model work?"},
                headers=self.auth(token),
            )
        response = self.client.get(f"/chatbot/conversations/{conv_id}", headers=self.auth(token))
        self.assertEqual(response.status_code, 200, response.text)
        messages = response.json()["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_unknown_conversation_404(self) -> None:
        token = self.register("unknown_conv_user")
        response = self.client.get("/chatbot/conversations/999999", headers=self.auth(token))
        self.assertEqual(response.status_code, 404)

    # --- Project-related question (mocked provider) ----------------------------

    def test_send_project_related_question_calls_provider(self) -> None:
        token = self.register("project_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="I3D was chosen because...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "Why did we choose I3D for Native Signs?"},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 201, response.text)
        mocked.assert_called_once()
        payload = response.json()
        self.assertEqual(payload["assistant_message"]["content"], "I3D was chosen because...")
        self.assertEqual(payload["assistant_message"]["scope"], "in_scope")
        self.assertEqual(payload["assistant_message"]["provider_status"], "ok")

    def test_assistant_response_persisted(self) -> None:
        token = self.register("persist_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="Persisted reply."):
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How is XP awarded?"},
                headers=self.auth(token),
            )
        with SessionLocal() as db:
            messages = db.query(ChatbotMessage).filter(ChatbotMessage.conversation_id == conv_id).all()
        roles = sorted(m.role for m in messages)
        self.assertEqual(roles, ["assistant", "user"])
        assistant_row = next(m for m in messages if m.role == "assistant")
        self.assertEqual(assistant_row.content, "Persisted reply.")
        self.assertIsNotNone(assistant_row.created_at)

    # --- User isolation ---------------------------------------------------------

    def test_user_isolation(self) -> None:
        token_a = self.register("iso_user_a")
        token_b = self.register("iso_user_b")
        conv_id = self.create_conversation(token_a)

        self.assertEqual(
            self.client.get(f"/chatbot/conversations/{conv_id}", headers=self.auth(token_b)).status_code, 404
        )
        self.assertEqual(
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages", json={"content": "hi"}, headers=self.auth(token_b)
            ).status_code,
            404,
        )
        listing = self.client.get("/chatbot/conversations", headers=self.auth(token_b))
        self.assertEqual(listing.json()["items"], [])

    def test_feedback_isolated_per_user(self) -> None:
        token_a = self.register("fb_user_a")
        token_b = self.register("fb_user_b")
        conv_id = self.create_conversation(token_a)
        with patch.object(chatbot_llm, "generate_reply", return_value="Reply."):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does the database store my progress?"},
                headers=self.auth(token_a),
            )
        message_id = response.json()["assistant_message"]["id"]

        denied = self.client.post(
            f"/chatbot/messages/{message_id}/feedback", json={"helpful": False}, headers=self.auth(token_b)
        )
        self.assertEqual(denied.status_code, 404)

        allowed = self.client.post(
            f"/chatbot/messages/{message_id}/feedback", json={"helpful": True}, headers=self.auth(token_a)
        )
        self.assertEqual(allowed.status_code, 201, allowed.text)
        self.assertTrue(allowed.json()["helpful"])
        with SessionLocal() as db:
            rows = db.query(ChatbotFeedback).filter(ChatbotFeedback.message_id == message_id).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].rating, 1)

    # --- Off-topic handling (backend-enforced, no provider call) ---------------

    def test_off_topic_question_rejected_without_calling_provider(self) -> None:
        token = self.register("offtopic_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What is the capital of France?"},
                headers=self.auth(token),
            )
        mocked.assert_not_called()
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["scope"], "off_topic")
        self.assertEqual(payload["content"], "I can only answer questions related to the ASL-Quest project.")

    def test_various_off_topic_questions_rejected(self) -> None:
        token = self.register("offtopic_variety_user")
        conv_id = self.create_conversation(token)
        for question in ["Tell me a joke", "Who is the president of India?", "What's your favorite movie?"]:
            with patch.object(chatbot_llm, "generate_reply") as mocked:
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": question},
                    headers=self.auth(token),
                )
            mocked.assert_not_called()
            self.assertEqual(response.json()["assistant_message"]["scope"], "off_topic", question)

    def test_project_related_question_answered_normally(self) -> None:
        token = self.register("normal_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="native_sign_progress tracks...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "Explain how native_sign_progress works"},
                headers=self.auth(token),
            )
        mocked.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")

    # --- Missing provider configuration handled honestly ------------------------

    def test_missing_api_key_handled_gracefully_not_crashed(self) -> None:
        token = self.register("no_key_user")
        conv_id = self.create_conversation(token)
        # OPENROUTER_API_KEY is unset for the whole module (see top of file) — this
        # exercises the real chatbot_llm.generate_reply path, not a mock.
        response = self.client.post(
            f"/chatbot/conversations/{conv_id}/messages",
            json={"content": "How does the I3D model work?"},
            headers=self.auth(token),
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["scope"], "in_scope")
        self.assertEqual(payload["provider_status"], "not_configured")
        self.assertIn("OPENROUTER_API_KEY", payload["content"])
        # Never silently answers with a fabricated/general-knowledge reply.
        self.assertNotIn("I3D was chosen", payload["content"])

    def test_provider_error_handled_gracefully(self) -> None:
        token = self.register("provider_error_user")
        conv_id = self.create_conversation(token)
        with patch.object(
            chatbot_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("boom")
        ):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does MediaPipe hand detection work?"},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["provider_status"], "error")

    # --- No API key ever exposed -------------------------------------------------

    def test_api_key_never_exposed_in_response_or_db(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "sk-test-super-secret-value"
        try:
            token = self.register("secret_user")
            conv_id = self.create_conversation(token)
            with patch.object(chatbot_llm, "generate_reply", return_value="Some real answer."):
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "How is XP awarded?"},
                    headers=self.auth(token),
                )
            self.assertNotIn("sk-test-super-secret-value", response.text)
            with SessionLocal() as db:
                rows = db.query(ChatbotMessage).filter(ChatbotMessage.conversation_id == conv_id).all()
            for row in rows:
                self.assertNotIn("sk-test-super-secret-value", row.content)
                self.assertNotIn("sk-test-super-secret-value", row.metadata_json)
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    # --- AURA identity ------------------------------------------------------------

    def test_system_prompt_identifies_as_aura(self) -> None:
        prompt = chatbot_knowledge.build_system_prompt()
        self.assertIn("AURA", prompt)

    # --- Status endpoint (online/configured indicator) -----------------------------

    def test_status_endpoint_reports_unconfigured(self) -> None:
        token = self.register("status_unconfigured_user")
        response = self.client.get("/chatbot/status", headers=self.auth(token))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"configured": False})

    def test_status_endpoint_reports_configured_without_leaking_key(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "sk-status-test-secret"
        try:
            token = self.register("status_configured_user")
            response = self.client.get("/chatbot/status", headers=self.auth(token))
            self.assertEqual(response.json(), {"configured": True})
            self.assertNotIn("sk-status-test-secret", response.text)
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_status_endpoint_requires_auth(self) -> None:
        self.assertEqual(self.client.get("/chatbot/status").status_code, 401)

    # --- Granular provider error handling (router-level mapping) ------------------

    def test_rate_limited_handled_gracefully(self) -> None:
        token = self.register("rate_limited_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", side_effect=chatbot_llm.ChatbotRateLimitedError("429")):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How is XP awarded?"},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["provider_status"], "rate_limited")
        self.assertNotIn("429", payload["content"])

    def test_timeout_handled_gracefully(self) -> None:
        token = self.register("timeout_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", side_effect=chatbot_llm.ChatbotTimeoutError("timed out")):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How is XP awarded?"},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["provider_status"], "timeout")

    def test_malformed_provider_response_handled_gracefully(self) -> None:
        token = self.register("malformed_user")
        conv_id = self.create_conversation(token)
        with patch.object(
            chatbot_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("malformed")
        ):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How is XP awarded?"},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["provider_status"], "error")
        # The internal exception detail is never echoed to the user.
        self.assertNotIn("malformed", payload["content"])

    # --- Bounded conversation context / follow-up handling -------------------------

    def test_conversation_history_is_bounded_to_recent_messages(self) -> None:
        token = self.register("history_user")
        conv_id = self.create_conversation(token)
        with SessionLocal() as db:
            for i in range(20):
                role = "user" if i % 2 == 0 else "assistant"
                db.add(ChatbotMessage(conversation_id=conv_id, role=role, content=f"msg {i}"))
            db.commit()
            conversation = db.query(ChatbotConversation).filter(ChatbotConversation.id == conv_id).first()
            history = _build_history(conversation)
        self.assertEqual(len(history), MAX_HISTORY_MESSAGES)
        self.assertEqual(history[-1]["content"], "msg 19")
        self.assertEqual(history[0]["content"], f"msg {20 - MAX_HISTORY_MESSAGES}")

    def test_followup_phrase_reaches_provider_after_in_scope_turn(self) -> None:
        token = self.register("followup_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="I3D is a video model...") as first_call:
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does I3D work?"},
                headers=self.auth(token),
            )
        first_history = first_call.call_args.args[1]
        self.assertEqual(len(first_history), 1)
        self.assertEqual(first_history[0]["content"], "How does I3D work?")

        # "why?" alone is not a project keyword -- it only passes because the
        # conversation already has a prior in-scope turn to anchor it to.
        with patch.object(chatbot_llm, "generate_reply", return_value="Because it generalized better.") as second_call:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "why?"},
                headers=self.auth(token),
            )
        second_call.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")
        second_history = second_call.call_args.args[1]
        contents = [m["content"] for m in second_history]
        # The provider receives the prior exchange as context, not just the bare "why?".
        self.assertIn("How does I3D work?", contents)
        self.assertIn("I3D is a video model...", contents)
        self.assertEqual(contents[-1], "why?")

    def test_followup_phrase_without_prior_in_scope_turn_is_rejected(self) -> None:
        token = self.register("followup_no_anchor_user")
        conv_id = self.create_conversation(token)
        # First turn is off-topic, so there is no prior in-scope turn to anchor to.
        with patch.object(chatbot_llm, "generate_reply") as mocked:
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "Tell me a joke"},
                headers=self.auth(token),
            )
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "why?"},
                headers=self.auth(token),
            )
        mocked.assert_not_called()
        self.assertEqual(response.json()["assistant_message"]["scope"], "off_topic")

    def test_unrelated_question_still_blocked_mid_conversation(self) -> None:
        # Regression test for a real bug caught during live testing: an earlier
        # version of the follow-up-continuity mechanism used a NEGATIVE deny-list of
        # "obviously generic trivia" shapes (capital-of, weather-in, etc.) to keep a
        # short unrelated question from riding along after an in-scope reply. A
        # short, unrelated, *undeny-listed* question -- "What is the tallest
        # mountain in the world?", which matched none of those fixed patterns --
        # reached the real OpenRouter provider anyway. The fix replaced that
        # deny-list with a positive allow-list of genuine continuation openers
        # (_FOLLOWUP_OPENER_PATTERN); this test locks in that a short-but-unrelated
        # question with a brand new subject of its own stays blocked even
        # immediately after an in-scope exchange.
        token = self.register("midconvo_offtopic_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="It's stored in native_sign_progress."):
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What database stores native sign progress?"},
                headers=self.auth(token),
            )
        with patch.object(chatbot_llm, "generate_reply") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What is the tallest mountain in the world?"},
                headers=self.auth(token),
            )
        mocked.assert_not_called()
        self.assertEqual(response.json()["assistant_message"]["scope"], "off_topic")

    # --- Free-form questions (not a predefined-question/FAQ interface) -------------
    # These exercise the real backend scope guard (backend/services/chatbot_knowledge
    # .py::is_in_scope) end to end via the router, not a mock of it, so a future
    # regression that narrows the guard back to a rigid keyword-only list would be
    # caught here.

    def test_suggested_question_works(self) -> None:
        # Verbatim text from frontend/src/chatbot/ChatbotPage.tsx's SUGGESTED_QUESTIONS.
        token = self.register("suggested_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="It's a ResNet18 classifier...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does A-Z recognition work?"},
                headers=self.auth(token),
            )
        mocked.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")

    def test_arbitrary_typed_asl_quest_question_works(self) -> None:
        # No exact keyword phrase from the old allow-list ("wrong"/"sign" are the
        # only overlaps, both newly added) -- this is a genuinely free-typed question,
        # not a suggestion pulled from the UI.
        token = self.register("arbitrary_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="Nothing is saved as correct...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What happens when a user gets a sign wrong?"},
                headers=self.auth(token),
            )
        mocked.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")

    def test_arbitrary_technical_project_question_works(self) -> None:
        token = self.register("architecture_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="FastAPI backend, React frontend...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "Explain the architecture simply."},
                headers=self.auth(token),
            )
        mocked.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")

    def test_natural_language_variation_works(self) -> None:
        # A paraphrase, not a fixed phrase from PROJECT_KEYWORDS or the suggestions
        # list -- only "handshape" overlaps, which is exactly the point: scope is
        # decided by real content words in the sentence, not a lookup against a
        # fixed question list.
        token = self.register("variation_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="MediaPipe finds the hand first...") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What method figures out which handshape I'm making with my hand?"},
                headers=self.auth(token),
            )
        mocked.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")

    def test_followup_uses_conversation_context_for_pronoun_reference(self) -> None:
        # The exact scenario from the task: "alternatives" shares no words at all
        # with "ResNet18" -- old FOLLOWUP_PHRASES (~16 fixed exact strings) couldn't
        # have matched this either way; it now also matches PROJECT_KEYWORDS directly
        # ("alternatives"), so this is covered twice over (keyword AND continuity).
        token = self.register("context_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="ResNet18 was fast and small...") as first:
            self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "Why did we use ResNet18?"},
                headers=self.auth(token),
            )
        first.assert_called_once()

        with patch.object(chatbot_llm, "generate_reply", return_value="MobileNet or EfficientNet, for example.") as second:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What are the alternatives?"},
                headers=self.auth(token),
            )
        second.assert_called_once()
        self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")
        contents = [m["content"] for m in second.call_args.args[1]]
        self.assertIn("Why did we use ResNet18?", contents)
        self.assertIn("ResNet18 was fast and small...", contents)
        self.assertEqual(contents[-1], "What are the alternatives?")

    def test_capital_of_france_is_blocked_before_provider_call(self) -> None:
        token = self.register("france_q_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "What is the capital of France?"},
                headers=self.auth(token),
            )
        mocked.assert_not_called()
        self.assertEqual(response.json()["assistant_message"]["scope"], "off_topic")

    def test_empty_input_is_rejected_by_validation(self) -> None:
        token = self.register("empty_input_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply") as mocked:
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": ""},
                headers=self.auth(token),
            )
        self.assertEqual(response.status_code, 422)
        mocked.assert_not_called()


class ChatbotLLMProviderTests(unittest.TestCase):
    """Unit tests for backend/services/chatbot_llm.py's own response
    classification, independent of the router/DB layer above. The external
    network call (httpx.post) is always mocked -- no real request is made."""

    def setUp(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "sk-llm-unit-test-key"

    def tearDown(self) -> None:
        os.environ.pop("OPENROUTER_API_KEY", None)

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def _ok_response(self, text: str = "A real answer.") -> httpx.Response:
        response = httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": text}}]}
        )
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        return response

    def test_missing_api_key_raises_not_configured(self) -> None:
        os.environ.pop("OPENROUTER_API_KEY", None)
        with self.assertRaises(chatbot_llm.ChatbotNotConfiguredError):
            chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_successful_response_returns_text(self) -> None:
        with patch("backend.services.chatbot_llm.httpx.post", return_value=self._ok_response("Hello!")):
            result = chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Hello!")

    def test_request_uses_openrouter_endpoint_bearer_auth_and_openai_shape(self) -> None:
        # Locks in the exact OpenRouter request shape: OpenAI-compatible chat
        # completions, system prompt folded into `messages`, Bearer auth (not
        # Anthropic's x-api-key), and the configured model/base URL.
        with patch(
            "backend.services.chatbot_llm.httpx.post", return_value=self._ok_response("Hi there.")
        ) as mocked:
            chatbot_llm.generate_reply(
                "You are AURA.", [{"role": "user", "content": "What is ASL-Quest?"}]
            )
        mocked.assert_called_once()
        args, kwargs = mocked.call_args
        self.assertEqual(args[0], self.OPENROUTER_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-llm-unit-test-key")
        self.assertNotIn("x-api-key", kwargs["headers"])
        self.assertEqual(
            kwargs["json"]["messages"],
            [
                {"role": "system", "content": "You are AURA."},
                {"role": "user", "content": "What is ASL-Quest?"},
            ],
        )
        self.assertEqual(kwargs["json"]["model"], "openrouter/free")
        self.assertNotIn("system", kwargs["json"])  # not a top-level field in this shape

    def test_conversation_history_forwarded_after_system_message(self) -> None:
        history = [
            {"role": "user", "content": "How does I3D work?"},
            {"role": "assistant", "content": "I3D is a video model..."},
            {"role": "user", "content": "why?"},
        ]
        with patch(
            "backend.services.chatbot_llm.httpx.post", return_value=self._ok_response("Because...")
        ) as mocked:
            chatbot_llm.generate_reply("system prompt", history)
        sent_messages = mocked.call_args.kwargs["json"]["messages"]
        self.assertEqual(sent_messages[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(sent_messages[1:], history)

    def test_timeout_raises_timeout_error(self) -> None:
        with patch("backend.services.chatbot_llm.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with self.assertRaises(chatbot_llm.ChatbotTimeoutError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_network_failure_raises_provider_error_not_a_raw_exception(self) -> None:
        # DNS failure / connection refused / offline host -- httpx.ConnectError is a
        # subclass of httpx.HTTPError, the general network-failure branch in
        # generate_reply (distinct from the more specific TimeoutException branch).
        with patch(
            "backend.services.chatbot_llm.httpx.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_rate_limit_status_raises_rate_limited_error(self) -> None:
        response = httpx.Response(429, json={"error": {"message": "rate limited"}})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotRateLimitedError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_invalid_api_key_status_raises_provider_error(self) -> None:
        response = httpx.Response(401, json={"error": {"message": "invalid api key"}})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError) as ctx:
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])
        # The 401 body/detail is never surfaced -- only the status code.
        self.assertNotIn("invalid api key", str(ctx.exception))
        self.assertNotIn("sk-llm-unit-test-key", str(ctx.exception))

    def test_server_error_status_raises_provider_error(self) -> None:
        response = httpx.Response(503, json={"error": {"message": "upstream model unavailable"}})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError) as ctx:
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])
        self.assertNotIn("upstream model unavailable", str(ctx.exception))

    def test_malformed_json_raises_provider_error(self) -> None:
        response = httpx.Response(200, content=b"not json at all")
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_missing_choices_raises_provider_error(self) -> None:
        # Right shape of JSON, wrong/unexpected structure (e.g. an OpenRouter
        # error payload returned with a 200 status).
        response = httpx.Response(200, json={"id": "gen-1", "choices": []})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_empty_content_raises_provider_error(self) -> None:
        response = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": ""}}]})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        with patch("backend.services.chatbot_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])

    def test_api_key_never_appears_in_any_exception_message(self) -> None:
        with patch(
            "backend.services.chatbot_llm.httpx.post", side_effect=httpx.ConnectError("connection refused")
        ):
            with self.assertRaises(chatbot_llm.ChatbotProviderError) as ctx:
                chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])
        self.assertNotIn("sk-llm-unit-test-key", str(ctx.exception))

    def test_api_key_never_appears_in_request_url_or_json_body(self) -> None:
        # The key must travel only in the Authorization header, never in the
        # URL or JSON body (which are more likely to end up in logs/traces).
        with patch(
            "backend.services.chatbot_llm.httpx.post", return_value=self._ok_response("Hi.")
        ) as mocked:
            chatbot_llm.generate_reply("system", [{"role": "user", "content": "hi"}])
        args, kwargs = mocked.call_args
        self.assertNotIn("sk-llm-unit-test-key", args[0])
        self.assertNotIn("sk-llm-unit-test-key", str(kwargs["json"]))


if __name__ == "__main__":
    unittest.main()
