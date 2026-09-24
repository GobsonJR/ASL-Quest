import { authHeaders } from "../auth/token";

const API = "/api";

export type ChatbotScope = "in_scope" | "off_topic" | null;
export type ChatbotProviderStatus = "ok" | "not_configured" | "error" | "rate_limited" | "timeout" | null;
// Where an in-scope reply's content actually came from -- "llm" (the configured
// provider answered normally) or "knowledge_base" (no LLM was reachable, so
// backend/services/chatbot_knowledge.py::fallback_answer answered instead, from
// ASL-Quest's static offline knowledge base). Older/mocked backends may omit this.
export type ChatbotSource = "llm" | "knowledge_base" | null;

// AURA's active LLM provider -- see backend/services/chatbot_llm.py. "local"
// runs fully offline via Ollama; "openrouter" is the original hosted
// provider; "auto" prefers local when reachable, falling back to OpenRouter.
// Optional fields: older/mocked backends may only return { configured }.
export type ChatbotProvider = "local" | "openrouter" | "auto";

export type ChatbotStatus = {
  configured: boolean;
  provider?: ChatbotProvider;
  local_available?: boolean;
  local_model?: string | null;
  knowledge_base_available?: boolean;
  online_required?: boolean;
};

export type ChatbotMessage = {
  id: number;
  conversation_id: number;
  role: "user" | "assistant";
  content: string;
  scope: ChatbotScope;
  provider_status: ChatbotProviderStatus;
  source: ChatbotSource;
  created_at: string;
};

export type ChatbotConversation = {
  id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages?: ChatbotMessage[];
};

export type ChatbotSendResult = {
  conversation_id: number;
  user_message: ChatbotMessage;
  assistant_message: ChatbotMessage;
};

export type ChatbotFeedbackResult = {
  id: number;
  message_id: number;
  helpful: boolean;
  feedback: string | null;
  created_at: string;
};

async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail ?? fallback;
  } catch {
    return fallback;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) throw new Error(await parseError(response, "ASL-Quest Assistant request failed."));
  return response.json();
}

export function fetchChatbotStatus(): Promise<ChatbotStatus> {
  return request<ChatbotStatus>("/chatbot/status");
}

export function createConversation(title?: string | null): Promise<ChatbotConversation> {
  return request<ChatbotConversation>("/chatbot/conversations", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
}

export function listConversations(): Promise<{ items: ChatbotConversation[] }> {
  return request<{ items: ChatbotConversation[] }>("/chatbot/conversations");
}

export function getConversation(conversationId: number): Promise<ChatbotConversation> {
  return request<ChatbotConversation>(`/chatbot/conversations/${conversationId}`);
}

export function sendChatbotMessage(conversationId: number, content: string): Promise<ChatbotSendResult> {
  return request<ChatbotSendResult>(`/chatbot/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}

export function sendChatbotFeedback(
  messageId: number,
  helpful: boolean,
  feedback?: string | null
): Promise<ChatbotFeedbackResult> {
  return request<ChatbotFeedbackResult>(`/chatbot/messages/${messageId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ helpful, feedback: feedback ?? null }),
  });
}
