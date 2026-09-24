import { useEffect, useRef, useState } from "react";
import {
  Card,
  EmptyState,
  PrimaryButton,
  SecondaryButton,
  SectionLabel,
  StatusChip,
} from "../components/AppShell";
import { useToast } from "../components/Toast";
import {
  createConversation,
  fetchChatbotStatus,
  getConversation,
  listConversations,
  sendChatbotFeedback,
  sendChatbotMessage,
  type ChatbotConversation,
  type ChatbotMessage,
  type ChatbotStatus,
} from "./api";

const SUGGESTED_QUESTIONS = [
  "How does A-Z recognition work?",
  "How does Native Sign recognition work?",
  "What is I3D?",
  "How does the database store my progress?",
  "How is XP awarded?",
];

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
}

/** AURA's small circular identity badge, reused in the header and beside every reply. */
function AuraAvatar({ size = "md" }: { size?: "sm" | "md" }) {
  const dimensions = size === "sm" ? "h-7 w-7 text-xs" : "h-11 w-11 text-base";
  return (
    <span
      className={`grid shrink-0 place-items-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-dim)] font-display font-semibold text-[var(--color-ink)] shadow-[0_0_20px_rgba(215,243,106,0.35)] ${dimensions}`}
      aria-hidden="true"
    >
      A
    </span>
  );
}

// "qwen3:4b" -> "Qwen3 4B" -- a friendly label, not the raw Ollama tag.
function formatLocalModelLabel(model: string): string {
  const [name, size] = model.split(":");
  const prettyName = name ? name.charAt(0).toUpperCase() + name.slice(1) : model;
  return size ? `${prettyName} ${size.toUpperCase()}` : prettyName;
}

function AuraStatusIndicator({ status }: { status: ChatbotStatus | null }) {
  if (status === null) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--color-muted)]" aria-hidden="true" />
        Checking status...
      </span>
    );
  }
  // Running on a local Ollama model right now (provider "local" or "auto"
  // with a reachable local server) -- AURA works with no internet in this
  // state, so it gets its own indicator rather than the generic "online" one.
  if (status.local_available) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-[var(--color-success)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" aria-hidden="true" />
        Offline AI{status.local_model ? ` • ${formatLocalModelLabel(status.local_model)}` : ""}
      </span>
    );
  }
  if (status.configured) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-[var(--color-success)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" aria-hidden="true" />
        AURA is online
      </span>
    );
  }
  // No local model reachable and no hosted provider configured -- AURA still
  // answers from its static offline knowledge base (chatbot_knowledge.
  // fallback_answer), so this is not a broken/setup-needed state.
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-[var(--color-success)]">
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" aria-hidden="true" />
      Offline knowledge base
    </span>
  );
}

export function ChatbotPage() {
  const { push } = useToast();

  const [conversations, setConversations] = useState<ChatbotConversation[] | null>(null);
  const [activeConversation, setActiveConversation] = useState<ChatbotConversation | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [status, setStatus] = useState<ChatbotStatus | null>(null);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [votedMessageIds, setVotedMessageIds] = useState<Set<number>>(new Set());

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    fetchChatbotStatus()
      .then((result) => {
        if (!cancelled) setStatus(result);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    listConversations()
      .then(async (response) => {
        if (cancelled) return;
        setConversations(response.items);
        if (response.items.length > 0) {
          const full = await getConversation(response.items[0].id);
          if (!cancelled) setActiveConversation(full);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : "Unable to load AURA.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "end" });
  }, [activeConversation?.messages?.length, sending]);

  async function ensureConversation(): Promise<ChatbotConversation> {
    if (activeConversation) return activeConversation;
    const created = await createConversation();
    setActiveConversation(created);
    setConversations((current) => [created, ...(current ?? [])]);
    return created;
  }

  async function submitMessage(content: string) {
    const trimmed = content.trim();
    if (!trimmed || sending) return;
    setSending(true);
    setSendError(null);
    try {
      const conversation = await ensureConversation();
      const result = await sendChatbotMessage(conversation.id, trimmed);
      setActiveConversation((current) => {
        const base = current ?? conversation;
        return {
          ...base,
          title: base.title ?? trimmed.slice(0, 60),
          messages: [...(base.messages ?? []), result.user_message, result.assistant_message],
        };
      });
      setConversations((current) =>
        (current ?? []).some((c) => c.id === conversation.id)
          ? current!.map((c) => (c.id === conversation.id ? { ...c, title: c.title ?? trimmed.slice(0, 60) } : c))
          : [{ ...conversation, title: conversation.title ?? trimmed.slice(0, 60) }, ...(current ?? [])]
      );
      setInput("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Message failed to send. Please try again.";
      setSendError(message);
      push(message, "error");
    } finally {
      setSending(false);
    }
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    void submitMessage(input);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage(input);
    }
  }

  function startNewConversation() {
    setActiveConversation(null);
    setInput("");
    setSendError(null);
  }

  async function selectConversation(id: number) {
    setLoadError(null);
    try {
      const full = await getConversation(id);
      setActiveConversation(full);
    } catch (err) {
      push(err instanceof Error ? err.message : "Unable to load that conversation.", "error");
    }
  }

  async function handleFeedback(message: ChatbotMessage, helpful: boolean) {
    if (votedMessageIds.has(message.id)) return;
    try {
      await sendChatbotFeedback(message.id, helpful);
      setVotedMessageIds((current) => new Set(current).add(message.id));
      push(helpful ? "Thanks for the feedback!" : "Thanks — we'll use that to improve AURA.", "success");
    } catch (err) {
      push(err instanceof Error ? err.message : "Could not record feedback.", "error");
    }
  }

  const messages = activeConversation?.messages ?? [];

  return (
    <div className="space-y-8 md:space-y-10">
      <div className="glass-panel relative overflow-hidden rounded-[var(--radius-panel)] p-6 md:p-8">
        <div
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(600px 260px at 15% -10%, rgba(215,243,106,0.16) 0%, transparent 55%), radial-gradient(500px 220px at 100% 10%, rgba(243,193,107,0.10) 0%, transparent 50%)",
          }}
          aria-hidden="true"
        />
        <div className="relative flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <AuraAvatar />
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="font-display text-2xl font-medium text-[#eef4f0] md:text-3xl">AURA</h1>
                <StatusChip tone="accent">ASL-QUEST PROJECT ASSISTANT</StatusChip>
              </div>
              <div className="mt-1">
                <AuraStatusIndicator status={status} />
              </div>
            </div>
          </div>
          {conversations && conversations.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {conversations.length > 1 && (
                <select
                  className="input-field w-auto max-w-[16rem]"
                  aria-label="Switch conversation"
                  value={activeConversation?.id ?? ""}
                  onChange={(event) => void selectConversation(Number(event.target.value))}
                >
                  {conversations.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.title ?? `Conversation from ${formatTime(c.created_at)}`}
                    </option>
                  ))}
                </select>
              )}
              <SecondaryButton onClick={startNewConversation}>New conversation</SecondaryButton>
            </div>
          )}
        </div>
      </div>

      {loading ? (
        <div className="grid min-h-[40vh] place-items-center text-[var(--color-mist)]">Loading AURA...</div>
      ) : loadError ? (
        <EmptyState icon="⚠️" title="Unable to load AURA" description={loadError} />
      ) : (
        <Card className="flex min-h-[55vh] flex-col overflow-hidden p-0">
          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5 md:px-6" role="log" aria-live="polite">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-5 py-10 text-center">
                <AuraAvatar />
                <div>
                  <SectionLabel>Ask AURA about ASL-Quest</SectionLabel>
                  <p className="mx-auto mt-2 max-w-sm text-sm text-[var(--color-mist)]">
                    Type any question in your own words — A-Z recognition, Word Spelling, Native Signs, the
                    database, architecture, limitations, anything about how the project works. AURA only answers
                    questions about ASL-Quest.
                  </p>
                </div>
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.1em] text-[var(--color-muted)]">
                    Suggested questions
                  </p>
                  <p className="mt-1 text-xs text-[var(--color-muted)]">Shortcuts — not the only questions AURA can answer.</p>
                  <div className="mt-2 flex flex-wrap justify-center gap-2">
                    {SUGGESTED_QUESTIONS.map((question) => (
                      <button
                        key={question}
                        type="button"
                        className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel-soft)]/60 px-3.5 py-1.5 text-xs font-medium text-[var(--color-mist)] transition hover:border-[var(--color-accent)]/40 hover:text-[#eef4f0] disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => void submitMessage(question)}
                        disabled={sending}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  voted={votedMessageIds.has(message.id)}
                  onFeedback={(helpful) => void handleFeedback(message, helpful)}
                />
              ))
            )}

            {sending && (
              <div className="flex items-center gap-2">
                <AuraAvatar size="sm" />
                <p className="animate-pulse-glow inline-flex items-center gap-2 rounded-full border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 px-4 py-2 text-sm text-[var(--color-accent)]">
                  AURA is thinking...
                </p>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {sendError && (
            <p
              className="mx-4 mb-2 rounded-2xl border border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10 px-4 py-2 text-sm text-[var(--color-warm)] md:mx-6"
              role="alert"
            >
              {sendError}
            </p>
          )}

          <form onSubmit={handleSubmit} className="flex items-end gap-3 border-t border-[var(--color-line-soft)] px-4 py-4 md:px-6">
            <textarea
              className="input-field min-h-11 flex-1 resize-none"
              placeholder="Ask AURA anything about ASL-Quest..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={sending}
              aria-label="Message AURA"
            />
            <PrimaryButton type="submit" disabled={sending || !input.trim()} ariaLabel="Send message to AURA">
              Send
            </PrimaryButton>
          </form>
        </Card>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  voted,
  onFeedback,
}: {
  message: ChatbotMessage;
  voted: boolean;
  onFeedback: (helpful: boolean) => void;
}) {
  const isUser = message.role === "user";
  const statusChip =
    message.scope === "off_topic" ? (
      <StatusChip tone="neutral">Off-topic</StatusChip>
    ) : message.source === "knowledge_base" ? (
      // The LLM provider wasn't reachable for this reply (not configured, rate
      // limited, timed out, or otherwise erroring) -- chatbot_knowledge.
      // fallback_answer answered from the static offline knowledge base
      // instead, so this is a real answer, not a failure.
      <StatusChip tone="accent">Offline knowledge base</StatusChip>
    ) : null;

  return (
    <div className={`flex items-start gap-2 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && <AuraAvatar size="sm" />}
      <div className={`max-w-[85%] md:max-w-[70%] ${isUser ? "" : "min-w-0 flex-1 sm:flex-none"}`}>
        {statusChip && <div className="mb-1.5 flex gap-2">{statusChip}</div>}
        <div
          className={
            isUser
              ? "rounded-[var(--radius-panel)] rounded-br-sm bg-[var(--color-accent)] px-4 py-2.5 text-sm text-[var(--color-ink)]"
              : "rounded-[var(--radius-panel)] rounded-bl-sm border border-[var(--color-line)] bg-[var(--color-panel-soft)]/70 px-4 py-2.5 text-sm leading-relaxed text-[#eef4f0]"
          }
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
        <div className={`mt-1 flex items-center gap-2 text-xs text-[var(--color-muted)] ${isUser ? "justify-end" : "justify-between"}`}>
          <span>{formatTime(message.created_at)}</span>
          {!isUser && (
            <span className="flex items-center gap-1">
              <button
                type="button"
                className={`rounded-full px-2 py-0.5 transition hover:text-[var(--color-success)] disabled:cursor-not-allowed disabled:opacity-50 ${voted ? "text-[var(--color-success)]" : ""}`}
                onClick={() => onFeedback(true)}
                disabled={voted}
                aria-label="Mark this AURA response as helpful"
              >
                👍
              </button>
              <button
                type="button"
                className={`rounded-full px-2 py-0.5 transition hover:text-[var(--color-warm)] disabled:cursor-not-allowed disabled:opacity-50 ${voted ? "text-[var(--color-warm)]" : ""}`}
                onClick={() => onFeedback(false)}
                disabled={voted}
                aria-label="Mark this AURA response as not helpful"
              >
                👎
              </button>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
