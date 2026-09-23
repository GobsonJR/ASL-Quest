"""Focused tests for backend/routers/chatbot.py (the ASL-Quest Assistant).

Independent of the existing app test suite: uses its own temporary SQLite
file and creates every table via Base.metadata.create_all, matching the
pattern in test_native.py / test_database_phase2.py. Nothing here touches
data/asl_quest.db.

The external LLM provider is always mocked (chatbot_llm.generate_reply is
patched, or CHATBOT_API_KEY is left unset) -- this suite never makes a real
network call.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "chatbot-test-secret"
os.environ.pop("CHATBOT_API_KEY", None)  # unconfigured unless a test opts in

from backend import models  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import ChatbotFeedback, ChatbotMessage  # noqa: E402
from backend.services import chatbot_llm  # noqa: E402


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
        # CHATBOT_API_KEY is unset for the whole module (see top of file) — this
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
        self.assertIn("CHATBOT_API_KEY", payload["content"])
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
        os.environ["CHATBOT_API_KEY"] = "sk-test-super-secret-value"
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
            os.environ.pop("CHATBOT_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
