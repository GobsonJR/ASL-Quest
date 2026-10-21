import type { SignoBubble } from "./types";

/** Short contextual message above the launcher. Purely visual (aria-hidden): screen readers get
 * important messages once, through SIGNO's single live region, so animated bubbles never
 * cause repeated announcements. */
export function SignoSpeechBubble({ bubble, onDismiss }: { bubble: SignoBubble; onDismiss: () => void }) {
  const tone =
    bubble.kind === "celebrate"
      ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]"
      : "border-[var(--color-border)] bg-[var(--color-surface)]";
  return (
    <div
      key={bubble.id}
      aria-hidden="true"
      onClick={onDismiss}
      className={`signo-bubble-enter relative max-w-[min(17rem,calc(100vw-5.5rem))] cursor-pointer rounded-2xl border-2 px-3.5 py-2.5 text-sm font-semibold leading-snug text-[var(--color-ink)] shadow-[0_2px_0_0_var(--color-border)] ${tone}`}
    >
      {bubble.text}
      <span
        className={`absolute -bottom-[7px] right-7 h-3 w-3 rotate-45 border-b-2 border-r-2 ${tone}`}
        aria-hidden="true"
      />
    </div>
  );
}
