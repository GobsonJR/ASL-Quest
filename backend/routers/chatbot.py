from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from backend.database import get_db
from backend.models import (
    ChatbotConversation,
    ChatbotFeedback,
    ChatbotMessage,
    NativeSign,
    NativeSignProgress,
    User,
    utcnow,
)
from backend.schemas import ChatbotConversationCreate, ChatbotFeedbackCreate, ChatbotMessageCreate
from backend.security import get_current_user
from backend.services import chatbot_llm
from backend.services.chatbot_knowledge import OFF_TOPIC_REPLY, build_system_prompt, is_in_scope

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Only these substrings trigger fetching this user's own progress into the LLM
# context — Phase 7 asks for the *minimum necessary* context, not every question
# needing a database round trip, and never data belonging to anyone else.
_PROGRESS_CONTEXT_TRIGGERS = (
    "my progress", "my xp", "my level", "my streak", "my mastery",
    "my achievement", "my native", "how am i doing", "my stats",
)

# How many of the most recent messages in a conversation get sent to the
# provider as context for a follow-up (e.g. "what about the native head?").
# Bounded so a long-running conversation doesn't grow the request (and the
# token cost) without limit — 10 messages is 5 user/assistant exchanges,
# comfortably enough for the "why?" / "what about X?" style follow-ups this
# assistant supports.
MAX_HISTORY_MESSAGES = 10


def _serialize_message(message: ChatbotMessage) -> dict:
    try:
        metadata = json.loads(message.metadata_json) if message.metadata_json else {}
    except (TypeError, ValueError):
        metadata = {}
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "scope": metadata.get("scope"),
        "provider_status": metadata.get("provider_status"),
        "created_at": message.created_at.isoformat(),
    }


def _serialize_conversation(conversation: ChatbotConversation, include_messages: bool = False) -> dict:
    payload = {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }
    if include_messages:
        payload["messages"] = [_serialize_message(m) for m in conversation.messages]
    return payload


def _get_owned_conversation(db: Session, conversation_id: int, user: User) -> ChatbotConversation:
    # 404 (not 403) for both "doesn't exist" and "belongs to someone else" —
    # never reveals whether another user's conversation ID exists.
    conversation = (
        db.query(ChatbotConversation)
        .options(selectinload(ChatbotConversation.messages))
        .filter(ChatbotConversation.id == conversation_id, ChatbotConversation.user_id == user.id)
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conversation


def _build_user_context(db: Session, user: User, message: str) -> str | None:
    normalized = message.lower()
    if not any(trigger in normalized for trigger in _PROGRESS_CONTEXT_TRIGGERS):
        return None

    lines: list[str] = []
    progress = user.progress
    if progress is not None:
        lines.append(
            f"XP: {progress.xp} (level {progress.level}). "
            f"Current streak: {progress.current_streak} day(s), best streak {progress.best_streak}. "
            f"Total correct attempts: {progress.total_correct} / {progress.total_attempts}."
        )
    else:
        lines.append("This user has no recorded XP/streak progress yet.")

    total_signs = db.query(NativeSign).filter(NativeSign.active.is_(True)).count()
    if total_signs:
        rows = db.query(NativeSignProgress).filter(NativeSignProgress.user_id == user.id).all()
        started = sum(1 for row in rows if row.attempts > 0)
        mastered = sum(1 for row in rows if row.mastery >= 100.0)
        avg_mastery = round(sum(row.mastery for row in rows) / total_signs, 1) if rows else 0.0
        lines.append(
            f"Native Signs: {started} of {total_signs} signs started, {mastered} mastered, "
            f"overall mastery {avg_mastery}%."
        )

    return "\n".join(lines)


def _build_history(conversation: ChatbotConversation) -> list[dict[str, str]]:
    messages = [
        {"role": m.role, "content": m.content}
        for m in conversation.messages
        if m.role in ("user", "assistant")
    ]
    return messages[-MAX_HISTORY_MESSAGES:]


def _last_assistant_turn_in_scope(conversation: ChatbotConversation) -> bool:
    """Whether the conversation's MOST RECENT assistant reply was in scope — not
    "was any reply ever in scope" (that would let one early in-scope exchange
    permanently unlock unrelated follow-ups for the rest of the conversation). Feeds
    is_in_scope's bounded follow-up-continuity check."""
    assistant_messages = sorted(
        (m for m in conversation.messages if m.role == "assistant"),
        key=lambda m: m.id,
    )
    if not assistant_messages:
        return False
    last = assistant_messages[-1]
    try:
        metadata = json.loads(last.metadata_json) if last.metadata_json else {}
    except (TypeError, ValueError):
        metadata = {}
    return metadata.get("scope") == "in_scope"


@router.get("/status")
def chatbot_status(_: User = Depends(get_current_user)) -> dict:
    """Whether AURA's LLM provider is configured — never the key itself, never
    which provider/model, just enough for the frontend to show an online/setup-
    needed indicator before the user sends a first message."""
    return {"configured": chatbot_llm.is_configured()}


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ChatbotConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conversation = ChatbotConversation(user_id=current_user.id, title=payload.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return _serialize_conversation(conversation, include_messages=True)


@router.get("/conversations")
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conversations = (
        db.query(ChatbotConversation)
        .filter(ChatbotConversation.user_id == current_user.id)
        .order_by(ChatbotConversation.updated_at.desc())
        .all()
    )
    return {"items": [_serialize_conversation(c) for c in conversations]}


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conversation = _get_owned_conversation(db, conversation_id, current_user)
    return _serialize_conversation(conversation, include_messages=True)


@router.post("/conversations/{conversation_id}/messages", status_code=status.HTTP_201_CREATED)
def send_message(
    conversation_id: int,
    payload: ChatbotMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conversation = _get_owned_conversation(db, conversation_id, current_user)

    user_message = ChatbotMessage(
        conversation_id=conversation.id,
        role="user",
        content=payload.content,
    )
    db.add(user_message)
    db.flush()

    # The scope gate runs in the backend, independent of the LLM (Phase 4/9): an
    # off-topic message never reaches the provider at all, so an unconfigured or
    # misbehaving provider can't accidentally answer it either.
    in_scope = is_in_scope(payload.content, last_turn_in_scope=_last_assistant_turn_in_scope(conversation))

    if not in_scope:
        reply_text = OFF_TOPIC_REPLY
        metadata = {"scope": "off_topic"}
    else:
        user_context = _build_user_context(db, current_user, payload.content)
        system_prompt = build_system_prompt(user_context)
        history = _build_history(conversation) + [{"role": "user", "content": payload.content}]
        try:
            reply_text = chatbot_llm.generate_reply(system_prompt, history)
            metadata = {"scope": "in_scope", "provider_status": "ok"}
        except chatbot_llm.ChatbotNotConfiguredError:
            reply_text = (
                "AURA isn't fully set up yet — an administrator needs to configure the "
                "OPENROUTER_API_KEY environment variable before I can answer questions. "
                "See .env.example for details."
            )
            metadata = {"scope": "in_scope", "provider_status": "not_configured"}
        except chatbot_llm.ChatbotRateLimitedError:
            reply_text = "AURA is getting a lot of requests right now. Please wait a moment and try again."
            metadata = {"scope": "in_scope", "provider_status": "rate_limited"}
        except chatbot_llm.ChatbotTimeoutError:
            reply_text = "AURA's response is taking too long right now. Please try again."
            metadata = {"scope": "in_scope", "provider_status": "timeout"}
        except chatbot_llm.ChatbotProviderError:
            # Catch-all for anything else (invalid/revoked key, malformed or empty
            # response, unexpected HTTP status) — deliberately generic so the
            # message never echoes provider internals back to the user.
            reply_text = "I'm having trouble reaching AURA's provider right now. Please try again in a moment."
            metadata = {"scope": "in_scope", "provider_status": "error"}

    assistant_message = ChatbotMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=reply_text,
        metadata_json=json.dumps(metadata),
    )
    db.add(assistant_message)
    conversation.title = conversation.title or payload.content[:60]
    conversation.updated_at = utcnow()
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)

    return {
        "conversation_id": conversation.id,
        "user_message": _serialize_message(user_message),
        "assistant_message": _serialize_message(assistant_message),
    }


@router.post("/messages/{message_id}/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    message_id: int,
    payload: ChatbotFeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    message = (
        db.query(ChatbotMessage)
        .join(ChatbotConversation, ChatbotMessage.conversation_id == ChatbotConversation.id)
        .filter(
            ChatbotMessage.id == message_id,
            ChatbotMessage.role == "assistant",
            ChatbotConversation.user_id == current_user.id,
        )
        .first()
    )
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found.")

    feedback = ChatbotFeedback(
        message_id=message.id,
        rating=1 if payload.helpful else 0,
        feedback=payload.feedback,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {
        "id": feedback.id,
        "message_id": feedback.message_id,
        "helpful": feedback.rating == 1,
        "feedback": feedback.feedback,
        "created_at": feedback.created_at.isoformat(),
    }
