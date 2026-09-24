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
# Provider defaults to "openrouter" unless a test opts into local/auto mode.
os.environ.pop("CHATBOT_PROVIDER", None)
os.environ.pop("CHATBOT_LOCAL_BASE_URL", None)
os.environ.pop("CHATBOT_LOCAL_MODEL", None)

from backend import models, settings  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import ChatbotConversation, ChatbotFeedback, ChatbotMessage  # noqa: E402
from backend.routers.chatbot import MAX_HISTORY_MESSAGES, _build_history  # noqa: E402
from backend.services import chatbot_knowledge, chatbot_llm, chatbot_local_llm  # noqa: E402


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
        self.assertEqual(payload["source"], "knowledge_base")
        # No LLM was reachable, so this answers from the static offline
        # knowledge base instead of an apology -- the exact verified fact from
        # PROJECT_KNOWLEDGE_SECTIONS, not a fabricated/hallucinated reply.
        self.assertIn("I3D", payload["content"])
        self.assertIn(chatbot_knowledge.FALLBACK_PREFIX, payload["content"])

    # --- Offline knowledge-base fallback (no LLM reachable at all) --------------

    def test_fallback_answers_in_scope_question_when_no_provider_configured(self) -> None:
        token = self.register("fallback_xp_user")
        conv_id = self.create_conversation(token)
        response = self.client.post(
            f"/chatbot/conversations/{conv_id}/messages",
            json={"content": "How is XP awarded?"},
            headers=self.auth(token),
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["source"], "knowledge_base")
        self.assertIn("XP", payload["content"])

    def test_fallback_used_on_provider_error_not_just_not_configured(self) -> None:
        token = self.register("fallback_error_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("boom")):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does the database store my progress?"},
                headers=self.auth(token),
            )
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["provider_status"], "error")
        self.assertEqual(payload["source"], "knowledge_base")
        self.assertIn("SQLAlchemy", payload["content"])

    def test_successful_llm_reply_tagged_with_llm_source(self) -> None:
        token = self.register("llm_source_user")
        conv_id = self.create_conversation(token)
        with patch.object(chatbot_llm, "generate_reply", return_value="A real generated answer."):
            response = self.client.post(
                f"/chatbot/conversations/{conv_id}/messages",
                json={"content": "How does A-Z recognition work?"},
                headers=self.auth(token),
            )
        payload = response.json()["assistant_message"]
        self.assertEqual(payload["source"], "llm")

    def test_status_endpoint_reports_knowledge_base_always_available(self) -> None:
        token = self.register("status_kb_user")
        response = self.client.get("/chatbot/status", headers=self.auth(token))
        payload = response.json()
        self.assertTrue(payload["knowledge_base_available"])
        self.assertFalse(payload["online_required"])

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
        payload = response.json()
        self.assertEqual(payload["configured"], False)
        self.assertEqual(payload["provider"], "openrouter")
        # Default provider is openrouter, which never probes local Ollama.
        self.assertEqual(payload["local_available"], False)
        self.assertIsNone(payload["local_model"])

    def test_status_endpoint_reports_configured_without_leaking_key(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "sk-status-test-secret"
        try:
            token = self.register("status_configured_user")
            response = self.client.get("/chatbot/status", headers=self.auth(token))
            payload = response.json()
            self.assertEqual(payload["configured"], True)
            self.assertEqual(payload["provider"], "openrouter")
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
        self.assertEqual(payload["source"], "knowledge_base")
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

    # --- Local/offline provider mode (router-level, end to end) ----------------
    # CHATBOT_PROVIDER=local never touches OpenRouter -- chatbot_local_llm.
    # generate_reply is mocked (never a real Ollama server), and OpenRouter's
    # own httpx.post is also mocked in the "never falls back" tests specifically
    # so a regression that accidentally re-introduces a fallback is caught here,
    # not just at the chatbot_llm unit level (see ChatbotLocalProviderTests).

    def test_local_mode_scope_guard_blocks_before_provider(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        try:
            token = self.register("local_offtopic_user")
            conv_id = self.create_conversation(token)
            with patch.object(chatbot_local_llm, "generate_reply") as mocked:
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "What is the capital of France?"},
                    headers=self.auth(token),
                )
            mocked.assert_not_called()
            self.assertEqual(response.json()["assistant_message"]["scope"], "off_topic")
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)

    def test_local_mode_arbitrary_project_question_reaches_local_provider(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        try:
            token = self.register("local_project_user")
            conv_id = self.create_conversation(token)
            with patch.object(chatbot_local_llm, "generate_reply", return_value="ResNet18 is small and fast.") as mocked:
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "Why did we use ResNet18?"},
                    headers=self.auth(token),
                )
            mocked.assert_called_once()
            payload = response.json()["assistant_message"]
            self.assertEqual(payload["scope"], "in_scope")
            self.assertEqual(payload["provider_status"], "ok")
            self.assertEqual(payload["content"], "ResNet18 is small and fast.")
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)

    def test_local_mode_followup_context_reaches_local_provider(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        try:
            token = self.register("local_followup_user")
            conv_id = self.create_conversation(token)
            with patch.object(chatbot_local_llm, "generate_reply", return_value="ResNet18 was fast and small...") as first:
                self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "Why did we use ResNet18?"},
                    headers=self.auth(token),
                )
            first.assert_called_once()

            with patch.object(chatbot_local_llm, "generate_reply", return_value="MobileNet, for example.") as second:
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "What are the alternatives?"},
                    headers=self.auth(token),
                )
            second.assert_called_once()
            self.assertEqual(response.json()["assistant_message"]["scope"], "in_scope")
            # generate_reply(system_prompt, history, base_url=..., model=...) --
            # history is the second positional argument.
            history = second.call_args.args[1]
            contents = [m["content"] for m in history]
            self.assertIn("Why did we use ResNet18?", contents)
            self.assertIn("ResNet18 was fast and small...", contents)
            self.assertEqual(contents[-1], "What are the alternatives?")
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)

    def test_local_mode_never_calls_openrouter_even_when_configured(self) -> None:
        # Explicit local mode must never reach OpenRouter, even if a real
        # OPENROUTER_API_KEY happens to be configured -- this is what
        # guarantees a true offline review mode (never a silent fallback).
        os.environ["CHATBOT_PROVIDER"] = "local"
        os.environ["OPENROUTER_API_KEY"] = "sk-should-never-be-used-in-local-mode"
        try:
            token = self.register("local_no_openrouter_user")
            conv_id = self.create_conversation(token)
            with patch.object(chatbot_local_llm, "generate_reply", return_value="Local answer.") as local_mock:
                with patch("backend.services.chatbot_llm.httpx.post") as openrouter_mock:
                    response = self.client.post(
                        f"/chatbot/conversations/{conv_id}/messages",
                        json={"content": "How does XP work?"},
                        headers=self.auth(token),
                    )
            local_mock.assert_called_once()
            openrouter_mock.assert_not_called()
            self.assertEqual(response.json()["assistant_message"]["content"], "Local answer.")
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_local_mode_ollama_unreachable_handled_gracefully(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        try:
            token = self.register("local_unreachable_user")
            conv_id = self.create_conversation(token)
            with patch.object(
                chatbot_local_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("connection refused")
            ):
                response = self.client.post(
                    f"/chatbot/conversations/{conv_id}/messages",
                    json={"content": "How does XP work?"},
                    headers=self.auth(token),
                )
            self.assertEqual(response.status_code, 201, response.text)
            payload = response.json()["assistant_message"]
            self.assertEqual(payload["provider_status"], "error")
            self.assertNotIn("connection refused", payload["content"])
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)

    def test_status_endpoint_reports_local_provider_and_never_leaks_key(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        os.environ["OPENROUTER_API_KEY"] = "sk-local-mode-unused-secret"
        try:
            token = self.register("local_status_user")
            with patch.object(chatbot_local_llm, "is_available", return_value=True):
                response = self.client.get("/chatbot/status", headers=self.auth(token))
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["provider"], "local")
            self.assertEqual(payload["configured"], True)
            self.assertEqual(payload["local_available"], True)
            self.assertEqual(payload["local_model"], "qwen3:4b")
            self.assertNotIn("sk-local-mode-unused-secret", response.text)
        finally:
            os.environ.pop("CHATBOT_PROVIDER", None)
            os.environ.pop("OPENROUTER_API_KEY", None)


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


class ChatbotLocalProviderTests(unittest.TestCase):
    """Unit tests for the local Ollama provider (backend/services/
    chatbot_local_llm.py) and chatbot_llm.py's provider dispatch (the
    local/auto branches of generate_reply/is_configured/get_status_info). No
    real Ollama server and no real OpenRouter call is ever made -- the
    underlying httpx calls (or generate_reply/is_available themselves) are
    always mocked."""

    LOCAL_BASE_URL = "http://127.0.0.1:11434"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def setUp(self) -> None:
        for key in ("OPENROUTER_API_KEY", "CHATBOT_PROVIDER", "CHATBOT_LOCAL_BASE_URL", "CHATBOT_LOCAL_MODEL"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in ("OPENROUTER_API_KEY", "CHATBOT_PROVIDER", "CHATBOT_LOCAL_BASE_URL", "CHATBOT_LOCAL_MODEL"):
            os.environ.pop(key, None)

    def _ok_ollama_response(self, text: str = "A local answer.") -> httpx.Response:
        response = httpx.Response(
            200, json={"model": "qwen3:4b", "message": {"role": "assistant", "content": text}, "done": True}
        )
        response.request = httpx.Request("POST", f"{self.LOCAL_BASE_URL}/api/chat")
        return response

    def _ok_openrouter_response(self, text: str = "An OpenRouter answer.") -> httpx.Response:
        response = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})
        response.request = httpx.Request("POST", self.OPENROUTER_URL)
        return response

    # --- 1. Local provider configuration ----------------------------------------

    def test_local_provider_config_defaults(self) -> None:
        self.assertEqual(settings.get_chatbot_provider(), "openrouter")
        self.assertEqual(settings.get_chatbot_local_base_url(), "http://127.0.0.1:11434")
        self.assertEqual(settings.get_chatbot_local_model(), "qwen3:4b")

    def test_local_provider_config_from_env(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "Local"  # case-insensitive
        os.environ["CHATBOT_LOCAL_BASE_URL"] = "http://127.0.0.1:9999/"  # trailing slash
        os.environ["CHATBOT_LOCAL_MODEL"] = "qwen3:1.7b"
        self.assertEqual(settings.get_chatbot_provider(), "local")
        self.assertEqual(settings.get_chatbot_local_base_url(), "http://127.0.0.1:9999")
        self.assertEqual(settings.get_chatbot_local_model(), "qwen3:1.7b")

    # --- 2. Local Ollama request formatting --------------------------------------

    def test_local_request_uses_ollama_chat_endpoint_and_shape(self) -> None:
        with patch(
            "backend.services.chatbot_local_llm.httpx.post", return_value=self._ok_ollama_response("Hi.")
        ) as mocked:
            chatbot_local_llm.generate_reply(
                "You are AURA.", [{"role": "user", "content": "Hi"}], base_url=self.LOCAL_BASE_URL, model="qwen3:4b"
            )
        mocked.assert_called_once()
        args, kwargs = mocked.call_args
        self.assertEqual(args[0], f"{self.LOCAL_BASE_URL}/api/chat")
        self.assertEqual(kwargs["json"]["model"], "qwen3:4b")
        self.assertEqual(kwargs["json"]["stream"], False)
        self.assertEqual(
            kwargs["json"]["messages"],
            [{"role": "system", "content": "You are AURA."}, {"role": "user", "content": "Hi"}],
        )

    # --- 3. Local response parsing ------------------------------------------------

    def test_local_response_parsing_returns_text(self) -> None:
        with patch("backend.services.chatbot_local_llm.httpx.post", return_value=self._ok_ollama_response("The answer.")):
            result = chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")
        self.assertEqual(result, "The answer.")

    # --- 4. Malformed Ollama response -----------------------------------------------

    def test_local_malformed_json_raises_provider_error(self) -> None:
        response = httpx.Response(200, content=b"not json at all")
        response.request = httpx.Request("POST", f"{self.LOCAL_BASE_URL}/api/chat")
        with patch("backend.services.chatbot_local_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")

    def test_local_missing_message_key_raises_provider_error(self) -> None:
        # Right shape of JSON, wrong/unexpected structure.
        response = httpx.Response(200, json={"model": "qwen3:4b", "done": True})
        response.request = httpx.Request("POST", f"{self.LOCAL_BASE_URL}/api/chat")
        with patch("backend.services.chatbot_local_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")

    def test_local_empty_content_raises_provider_error(self) -> None:
        with patch("backend.services.chatbot_local_llm.httpx.post", return_value=self._ok_ollama_response("")):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")

    def test_local_http_error_status_raises_provider_error(self) -> None:
        response = httpx.Response(500, json={"error": "model not found"})
        response.request = httpx.Request("POST", f"{self.LOCAL_BASE_URL}/api/chat")
        with patch("backend.services.chatbot_local_llm.httpx.post", return_value=response):
            with self.assertRaises(chatbot_llm.ChatbotProviderError) as ctx:
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")
        self.assertNotIn("model not found", str(ctx.exception))

    # --- 5. Ollama unavailable ---------------------------------------------------

    def test_local_connection_refused_raises_provider_error(self) -> None:
        with patch("backend.services.chatbot_local_llm.httpx.post", side_effect=httpx.ConnectError("refused")):
            with self.assertRaises(chatbot_llm.ChatbotProviderError):
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")

    def test_local_timeout_raises_timeout_error(self) -> None:
        with patch("backend.services.chatbot_local_llm.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with self.assertRaises(chatbot_llm.ChatbotTimeoutError):
                chatbot_local_llm.generate_reply("sys", [], base_url=self.LOCAL_BASE_URL, model="qwen3:4b")

    def test_is_available_false_on_connection_error(self) -> None:
        with patch("backend.services.chatbot_local_llm.httpx.get", side_effect=httpx.ConnectError("refused")):
            self.assertFalse(chatbot_local_llm.is_available(self.LOCAL_BASE_URL))

    def test_is_available_true_on_200(self) -> None:
        ok = httpx.Response(200, json={"models": []})
        ok.request = httpx.Request("GET", f"{self.LOCAL_BASE_URL}/api/tags")
        with patch("backend.services.chatbot_local_llm.httpx.get", return_value=ok):
            self.assertTrue(chatbot_local_llm.is_available(self.LOCAL_BASE_URL))

    def test_is_available_false_on_non_200(self) -> None:
        bad = httpx.Response(500)
        bad.request = httpx.Request("GET", f"{self.LOCAL_BASE_URL}/api/tags")
        with patch("backend.services.chatbot_local_llm.httpx.get", return_value=bad):
            self.assertFalse(chatbot_local_llm.is_available(self.LOCAL_BASE_URL))

    # --- 6. Explicit local mode never calls OpenRouter ---------------------------

    def test_explicit_local_mode_never_calls_openrouter_even_on_failure(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        os.environ["OPENROUTER_API_KEY"] = "sk-should-never-be-used"
        with patch.object(
            chatbot_local_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("ollama down")
        ) as local_mock:
            with patch("backend.services.chatbot_llm.httpx.post") as openrouter_mock:
                with self.assertRaises(chatbot_llm.ChatbotProviderError):
                    chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        local_mock.assert_called_once()
        openrouter_mock.assert_not_called()

    def test_explicit_local_mode_uses_local_on_success(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "local"
        with patch.object(chatbot_local_llm, "generate_reply", return_value="Local answer.") as local_mock:
            with patch("backend.services.chatbot_llm.httpx.post") as openrouter_mock:
                result = chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Local answer.")
        local_mock.assert_called_once()
        openrouter_mock.assert_not_called()

    # --- 7. Auto mode prefers local when available --------------------------------

    def test_auto_mode_prefers_local_when_available(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "auto"
        os.environ["OPENROUTER_API_KEY"] = "sk-should-not-be-needed"
        with patch.object(chatbot_local_llm, "is_available", return_value=True):
            with patch.object(chatbot_local_llm, "generate_reply", return_value="Local wins.") as local_mock:
                with patch("backend.services.chatbot_llm.httpx.post") as openrouter_mock:
                    result = chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Local wins.")
        local_mock.assert_called_once()
        openrouter_mock.assert_not_called()

    # --- 8. Auto mode uses OpenRouter only when local unavailable and configured --

    def test_auto_mode_falls_back_to_openrouter_when_local_unavailable(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "auto"
        os.environ["OPENROUTER_API_KEY"] = "sk-fallback-key"
        with patch.object(chatbot_local_llm, "is_available", return_value=False):
            with patch(
                "backend.services.chatbot_llm.httpx.post", return_value=self._ok_openrouter_response("Fallback answer.")
            ) as openrouter_mock:
                result = chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Fallback answer.")
        openrouter_mock.assert_called_once()

    def test_auto_mode_does_not_use_openrouter_when_local_available(self) -> None:
        # Complements test_auto_mode_prefers_local_when_available: even with a
        # valid OpenRouter key present, local wins whenever it's reachable.
        os.environ["CHATBOT_PROVIDER"] = "auto"
        os.environ["OPENROUTER_API_KEY"] = "sk-present-but-unused"
        with patch.object(chatbot_local_llm, "is_available", return_value=True):
            with patch.object(chatbot_local_llm, "generate_reply", return_value="Local answer."):
                with patch("backend.services.chatbot_llm.httpx.post") as openrouter_mock:
                    chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        openrouter_mock.assert_not_called()

    # --- 9. No-provider state -----------------------------------------------------

    def test_auto_mode_raises_not_configured_when_neither_available(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "auto"
        with patch.object(chatbot_local_llm, "is_available", return_value=False):
            with self.assertRaises(chatbot_llm.ChatbotNotConfiguredError):
                chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])

    # --- 13. API key never appears in status/errors --------------------------------

    def test_get_status_info_never_includes_api_key(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "auto"
        os.environ["OPENROUTER_API_KEY"] = "sk-status-info-secret"
        with patch.object(chatbot_local_llm, "is_available", return_value=False):
            info = chatbot_llm.get_status_info()
        self.assertNotIn("sk-status-info-secret", str(info))
        self.assertEqual(info["provider"], "auto")

    def test_not_configured_message_never_includes_api_key(self) -> None:
        os.environ["CHATBOT_PROVIDER"] = "auto"
        os.environ["OPENROUTER_API_KEY"] = "sk-error-message-secret"
        # Key is set but local is unavailable and this key is deliberately
        # invalid-shaped -- irrelevant here since is_available gates first.
        with patch.object(chatbot_local_llm, "is_available", return_value=True):
            with patch.object(
                chatbot_local_llm, "generate_reply", side_effect=chatbot_llm.ChatbotProviderError("boom")
            ):
                with self.assertRaises(chatbot_llm.ChatbotProviderError) as ctx:
                    chatbot_llm.generate_reply("sys", [{"role": "user", "content": "hi"}])
        self.assertNotIn("sk-error-message-secret", str(ctx.exception))


class ChatbotKnowledgeFallbackTests(unittest.TestCase):
    """Unit tests for chatbot_knowledge.fallback_answer -- the deterministic,
    LLM-free answer path used whenever no provider is reachable. No network
    call, no DB, no app instance is involved here at all."""

    def test_matches_gamification_section_for_xp_question(self) -> None:
        answer = chatbot_knowledge.fallback_answer("How do I earn XP?")
        self.assertIn("XP", answer)
        self.assertIn(chatbot_knowledge.FALLBACK_PREFIX, answer)

    def test_matches_native_signs_section(self) -> None:
        answer = chatbot_knowledge.fallback_answer("What is Native ASL / Native Signs?")
        self.assertIn("I3D", answer)

    def test_matches_a_to_z_section(self) -> None:
        answer = chatbot_knowledge.fallback_answer("How does the A-Z alphabet recognition work?")
        self.assertIn("ResNet18", answer)

    def test_matches_database_section(self) -> None:
        answer = chatbot_knowledge.fallback_answer("What database does this use?")
        self.assertIn("SQLAlchemy", answer)

    def test_no_keyword_match_returns_no_info_reply_not_a_guess(self) -> None:
        # A message that reached fallback_answer at all already passed the
        # backend scope guard (is_in_scope), but that guard's allow-list is
        # broader than any single knowledge section -- e.g. "chatbot"/
        # "assistant" match PROJECT_KEYWORDS but no SECTION_KEYWORDS group.
        answer = chatbot_knowledge.fallback_answer("assistant")
        self.assertEqual(answer, chatbot_knowledge.NO_INFO_REPLY)

    def test_empty_message_returns_no_info_reply(self) -> None:
        self.assertEqual(chatbot_knowledge.fallback_answer(""), chatbot_knowledge.NO_INFO_REPLY)

    def test_never_calls_any_network_or_llm_code(self) -> None:
        # fallback_answer must be pure/local -- patch generate_reply on both
        # providers to raise if touched, and confirm the fallback still works.
        with patch.object(chatbot_llm, "generate_reply", side_effect=AssertionError("must not call LLM")):
            with patch.object(chatbot_local_llm, "generate_reply", side_effect=AssertionError("must not call LLM")):
                answer = chatbot_knowledge.fallback_answer("How is XP awarded?")
        self.assertIn("XP", answer)


if __name__ == "__main__":
    unittest.main()
