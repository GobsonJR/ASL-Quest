import {
  createConversation,
  sendChatbotMessage,
  type ChatbotSource,
} from "../chatbot/api";

// SIGNO's conversations go through AURA's existing chatbot API (/chatbot/...):
// same auth, same scope gate, same LLM provider config and offline knowledge-base
// fallback. No second chat stack and no keys in the frontend. Progress context for
// questions about XP/streaks is added by the backend (chatbot._build_user_context).

export type SignoAnswer = {
  conversationId: number;
  text: string;
  source: ChatbotSource;
};

export class SignoUnavailableError extends Error {}

export async function askSigno(question: string, conversationId: number | null): Promise<SignoAnswer> {
  try {
    let id = conversationId;
    if (id === null) {
      const conversation = await createConversation("SIGNO");
      id = conversation.id;
    }
    const result = await sendChatbotMessage(id, question);
    return {
      conversationId: id,
      text: result.assistant_message.content,
      source: result.assistant_message.source ?? null,
    };
  } catch (error) {
    if (error instanceof TypeError) {
      // fetch() rejects with TypeError when the backend is unreachable.
      throw new SignoUnavailableError("I can't reach the ASL Quest server right now.");
    }
    throw new SignoUnavailableError(
      error instanceof Error && error.message ? error.message : "The assistant couldn't answer just now."
    );
  }
}
