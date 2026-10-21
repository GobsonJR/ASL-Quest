import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Icon } from "../Icon";
import { SignoCharacter } from "./SignoCharacter";
import { useSigno } from "./SignoContext";
import type { SignoChatMessage, SignoMood } from "./types";

const MOOD_LABEL: Record<SignoMood, string> = {
  idle: "Ready when you are",
  curious: "Curious about this one",
  listening: "Cheering you on",
  thinking: "Thinking…",
  happy: "Happy for you",
  encouraging: "You've got this",
  celebrating: "Celebrating!",
  sleepy: "A little sleepy",
  explaining: "Here to help",
};

function originLabel(message: SignoChatMessage): string | null {
  if (message.origin === "ai") {
    return message.source === "knowledge_base" ? "ASL Quest knowledge base" : "AURA assistant";
  }
  if (message.origin === "local") return message.failed ? null : "SIGNO tip";
  return null;
}

const MAX_LENGTH = 1000;

export function SignoPanel({ id, onClose }: { id: string; onClose: () => void }) {
  const {
    mood, pulse, reaction, accessories, prefs, messages, busy, suggestions, canHint,
    setPrefs, setTyping, react, ask, retry, hint, celebrateProgress,
  } = useSigno();
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useEffect(() => {
    inputRef.current?.focus();
    return () => setTyping(false);
  }, [setTyping]);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages, busy]);

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.stopPropagation();
      onClose();
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setTyping(false);
    void ask(text);
  }

  function choose(question: string) {
    if (question === "Celebrate my progress!") celebrateProgress();
    else void ask(question);
  }

  return (
    <section
      id={id}
      role="dialog"
      aria-modal="false"
      aria-labelledby={titleId}
      onKeyDown={onKeyDown}
      className="signo-panel-enter flex max-h-[min(34rem,calc(100dvh-9rem))] flex-col overflow-hidden rounded-[var(--radius-panel)] border-2 border-[var(--color-border)] bg-[var(--color-surface)] shadow-[0_4px_0_0_var(--color-border),0_18px_40px_-18px_rgba(38,50,56,0.35)]"
    >
      <header className="flex items-center gap-2 border-b-2 border-[var(--color-border-soft)] bg-[var(--color-primary-soft)] px-3 py-2.5">
        <button type="button" onClick={react} className="shrink-0 rounded-2xl" aria-label="Tap SIGNO for a reaction">
          <SignoCharacter
            mood={mood}
            size={52}
            accessories={accessories}
            reaction={reaction}
            pulse={pulse}
            effects={prefs.effects}
          />
        </button>
        <div className="min-w-0 flex-1">
          <h2 id={titleId} className="font-display text-lg font-bold leading-tight text-[var(--color-ink)]">
            SIGNO
          </h2>
          <p className="truncate text-xs font-semibold text-[var(--color-ink-soft)]">{MOOD_LABEL[mood]}</p>
        </div>
        <button
          type="button"
          onClick={() => setPrefs({ effects: !prefs.effects })}
          aria-pressed={prefs.effects}
          aria-label="Animations"
          title={prefs.effects ? "Animations on" : "Animations off"}
          className={`grid h-10 w-10 place-items-center rounded-xl border-2 ${
            prefs.effects
              ? "border-[var(--color-primary)] bg-[var(--color-surface)] text-[var(--color-primary-dim)]"
              : "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-muted)]"
          }`}
        >
          <Icon name="sparkles" className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={() => setPrefs({ minimized: true })}
          aria-label="Minimize SIGNO"
          title="Minimize SIGNO (hides tips)"
          className="grid h-10 w-10 place-items-center rounded-xl border-2 border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-ink-soft)]"
        >
          <span aria-hidden="true" className="block h-0.5 w-4 rounded bg-current" />
        </button>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close SIGNO"
          className="grid h-10 w-10 place-items-center rounded-xl border-2 border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-ink-soft)]"
        >
          <Icon name="x" className="h-5 w-5" />
        </button>
      </header>

      <div
        ref={logRef}
        role="log"
        aria-label="Conversation with SIGNO"
        aria-live="polite"
        className="min-h-[7rem] flex-1 space-y-3 overflow-y-auto px-3 py-3"
      >
        {messages.length === 0 && (
          <p className="text-sm leading-relaxed text-[var(--color-ink-soft)]">
            Hi, I'm SIGNO! Ask me about ASL or how to use ASL Quest. I can't see your camera or your hands, so I can
            only help with what you tell me.
          </p>
        )}
        {messages.map((m) => {
          const label = originLabel(m);
          if (m.origin === "user") {
            return (
              <div key={m.id} className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-[var(--color-primary)] px-3.5 py-2 text-sm font-semibold text-white">
                  {m.text}
                </p>
              </div>
            );
          }
          return (
            <div key={m.id} className="flex flex-col items-start gap-1">
              {label && (
                <span className="px-1 text-[0.68rem] font-bold uppercase tracking-wide text-[var(--color-muted)]">
                  {label}
                </span>
              )}
              <p
                className={`max-w-[92%] whitespace-pre-wrap rounded-2xl rounded-bl-md px-3.5 py-2 text-sm leading-relaxed ${
                  m.failed
                    ? "bg-[var(--color-danger-soft)] font-semibold text-[var(--color-danger)]"
                    : "border-2 border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-ink)]"
                }`}
                role={m.failed ? "alert" : undefined}
              >
                {m.text}
              </p>
              {m.failed && (
                <button
                  type="button"
                  onClick={retry}
                  disabled={busy}
                  className="min-h-9 rounded-lg px-2 text-sm font-bold text-[var(--color-primary-dim)] hover:underline disabled:opacity-50"
                >
                  Try again
                </button>
              )}
            </div>
          );
        })}
        {busy && (
          <p role="status" className="px-1 text-sm font-semibold text-[var(--color-muted)]">
            SIGNO is thinking…
          </p>
        )}
      </div>

      <div className="space-y-2 border-t-2 border-[var(--color-border-soft)] px-3 pb-3 pt-2">
        <div className="flex flex-wrap gap-1.5" aria-label="Suggested questions" role="group">
          {canHint && (
            <button
              type="button"
              onClick={hint}
              className="min-h-9 rounded-full border-2 border-[var(--color-primary)] bg-[var(--color-primary-soft)] px-3 text-xs font-bold text-[var(--color-primary-dim)]"
            >
              Give me a hint
            </button>
          )}
          {suggestions.slice(0, messages.length === 0 ? 3 : 2).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => choose(s)}
              disabled={busy}
              className="min-h-9 max-w-full rounded-full border-2 border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-left text-xs font-semibold leading-snug text-[var(--color-ink-soft)] hover:border-[var(--color-primary-soft)] disabled:opacity-50"
            >
              {s.length > 70 ? `${s.slice(0, 67)}…` : s}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="flex gap-2">
          <label htmlFor={`${id}-input`} className="sr-only">
            Ask SIGNO a question
          </label>
          <input
            id={`${id}-input`}
            ref={inputRef}
            value={draft}
            maxLength={MAX_LENGTH}
            onChange={(event) => setDraft(event.target.value)}
            onFocus={() => setTyping(true)}
            onBlur={() => setTyping(false)}
            placeholder="Ask about ASL…"
            autoComplete="off"
            className="input-field min-w-0 flex-1"
          />
          <button
            type="submit"
            disabled={busy || draft.trim().length === 0}
            className="min-h-11 rounded-[var(--radius-control)] bg-[var(--color-primary)] px-4 text-sm font-extrabold uppercase tracking-wide text-white shadow-[0_3px_0_0_var(--color-primary-dim)] disabled:bg-[var(--color-locked-soft)] disabled:text-[var(--color-locked)] disabled:shadow-none"
          >
            Send
          </button>
        </form>
        <p className="text-[0.68rem] leading-snug text-[var(--color-muted)]">
          Answers come from ASL Quest's AURA assistant. SIGNO can't see your camera or judge your signing.
        </p>
      </div>
    </section>
  );
}
