import { Card, SectionLabel, StatusChip } from "./AppShell";
import { LetterMasteryBar } from "./LetterMasteryBar";
import { REFERENCE_LICENSE_NOTE, referenceImageAlt, referenceImageSrc } from "../pages/alphabetFeedback";

/** The "before/during detection" reference card: a large, dynamically-updating
 * ASL handshape illustration for the current target letter, plus a live status
 * chip. Deliberately the visual anchor of the right-hand column -- see
 * PracticePage, which swaps it out for CelebrationOverlay once the learner
 * gets the letter right. `correct` is optional so existing callers/tests that
 * don't track per-letter mastery here still render exactly as before, minus
 * the mastery bar. */
export function AlphabetReferencePanel({
  letter,
  statusLabel,
  statusTone = "accent",
  correct,
}: {
  letter: string;
  statusLabel?: string;
  statusTone?: "neutral" | "success" | "warning" | "accent";
  correct?: number;
}) {
  return (
    <Card className="relative space-y-4 overflow-hidden text-center" data-testid="alphabet-reference-panel">
      <div className="relative space-y-4">
        <SectionLabel>Your Target</SectionLabel>

        <div className="mx-auto w-full max-w-[220px] overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] p-3">
          <img
            key={letter.toUpperCase()}
            src={referenceImageSrc(letter)}
            alt={referenceImageAlt(letter)}
            className="h-auto w-full"
            width={200}
            height={263}
            loading="eager"
          />
        </div>

        <div>
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">
            Show me {letter.toUpperCase()}
          </p>
          <p className="mt-1 font-display text-5xl font-bold text-[var(--color-primary)]">{letter.toUpperCase()}</p>
          <p className="mt-2 text-sm leading-relaxed text-[var(--color-ink-soft)]">
            Hold this handshape in front of the camera — match the hand position above.
          </p>
        </div>

        {statusLabel && (
          <div className="flex justify-center" aria-live="polite">
            <StatusChip tone={statusTone}>{statusLabel}</StatusChip>
          </div>
        )}

        {typeof correct === "number" && (
          <div className="mx-auto max-w-[220px] text-left">
            <LetterMasteryBar correct={correct} />
          </div>
        )}

        <p className="text-[0.65rem] leading-snug text-[var(--color-muted)]">{REFERENCE_LICENSE_NOTE}</p>
      </div>
    </Card>
  );
}
