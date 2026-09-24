import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { ChatbotPage } from "./ChatbotPage";
import type { ChatbotConversation, ChatbotSendResult } from "./api";

const { fetchChatbotStatus, listConversations, getConversation, createConversation, sendChatbotMessage, sendChatbotFeedback } =
  vi.hoisted(() => ({
    fetchChatbotStatus: vi.fn(),
    listConversations: vi.fn(),
    getConversation: vi.fn(),
    createConversation: vi.fn(),
    sendChatbotMessage: vi.fn(),
    sendChatbotFeedback: vi.fn(),
  }));

vi.mock("../components/Toast", () => ({
  useToast: () => ({ push: vi.fn() }),
}));

vi.mock("./api", () => ({
  fetchChatbotStatus,
  listConversations,
  getConversation,
  createConversation,
  sendChatbotMessage,
  sendChatbotFeedback,
}));

function conversation(overrides: Partial<ChatbotConversation> = {}): ChatbotConversation {
  return {
    id: 1,
    title: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    messages: [],
    ...overrides,
  };
}

function sendResult(userContent: string, assistantContent = "Here's the answer."): ChatbotSendResult {
  return {
    conversation_id: 1,
    user_message: {
      id: 10,
      conversation_id: 1,
      role: "user",
      content: userContent,
      scope: null,
      provider_status: null,
      created_at: "2026-01-01T00:01:00Z",
    },
    assistant_message: {
      id: 11,
      conversation_id: 1,
      role: "assistant",
      content: assistantContent,
      scope: "in_scope",
      provider_status: "ok",
      created_at: "2026-01-01T00:01:01Z",
    },
  };
}

beforeEach(() => {
  fetchChatbotStatus.mockReset().mockResolvedValue({ configured: true });
  listConversations.mockReset().mockResolvedValue({ items: [] });
  getConversation.mockReset();
  createConversation.mockReset().mockResolvedValue(conversation());
  sendChatbotMessage.mockReset();
  sendChatbotFeedback.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("ChatbotPage: suggested questions are shortcuts, not the only input", () => {
  it("labels the suggestions clearly and doesn't restrict the main input to them", async () => {
    render(<ChatbotPage />);

    expect(await screen.findByText("Suggested questions")).toBeInTheDocument();
    expect(screen.getByText(/shortcuts/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask AURA anything about ASL-Quest...")).toBeInTheDocument();
  });

  it("submits a suggested question's exact text unmodified", async () => {
    sendChatbotMessage.mockResolvedValue(sendResult("How does A-Z recognition work?"));
    render(<ChatbotPage />);

    const suggestion = await screen.findByRole("button", { name: "How does A-Z recognition work?" });
    fireEvent.click(suggestion);

    await waitFor(() => expect(sendChatbotMessage).toHaveBeenCalledWith(1, "How does A-Z recognition work?"));
  });
});

describe("ChatbotPage: arbitrary typed questions", () => {
  it("submits manually typed text exactly as typed, not looked up against any fixed list", async () => {
    const typed = "Why is Native Sign recognition kept separate from A-Z?";
    sendChatbotMessage.mockResolvedValue(sendResult(typed));
    render(<ChatbotPage />);

    const textarea = await screen.findByLabelText("Message AURA");
    fireEvent.change(textarea, { target: { value: typed } });
    fireEvent.click(screen.getByRole("button", { name: "Send message to AURA" }));

    await waitFor(() => expect(sendChatbotMessage).toHaveBeenCalledWith(1, typed));
    expect(await screen.findByText(typed)).toBeInTheDocument();
  });

  it("submits on Enter (without Shift) using the exact typed content", async () => {
    const typed = "What are the current limitations of ASL-Quest?";
    sendChatbotMessage.mockResolvedValue(sendResult(typed));
    render(<ChatbotPage />);

    const textarea = await screen.findByLabelText("Message AURA");
    fireEvent.change(textarea, { target: { value: typed } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });

    await waitFor(() => expect(sendChatbotMessage).toHaveBeenCalledWith(1, typed));
  });

  it("does not submit empty or whitespace-only input", async () => {
    render(<ChatbotPage />);

    const textarea = await screen.findByLabelText("Message AURA");
    fireEvent.change(textarea, { target: { value: "   " } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(sendChatbotMessage).not.toHaveBeenCalled();
  });
});

describe("ChatbotPage: send failure surfaces an honest message", () => {
  it("shows the real error instead of silently failing", async () => {
    sendChatbotMessage.mockRejectedValue(new Error("AURA's provider is rate-limiting requests right now."));
    render(<ChatbotPage />);

    const textarea = await screen.findByLabelText("Message AURA");
    fireEvent.change(textarea, { target: { value: "How is XP awarded?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message to AURA" }));

    expect(await screen.findByText("AURA's provider is rate-limiting requests right now.")).toBeInTheDocument();
  });
});
