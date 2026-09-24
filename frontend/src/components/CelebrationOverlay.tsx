import { BADGES } from "../game/constants";
import type { PracticeOutcome } from "../game/types";
import { LetterMasteryBar } from "./LetterMasteryBar";
import { PrimaryButton, SecondaryButton } from "./AppShell";

function badgeTitle(id: string): string {
  return BADGES.find((badge) => badge.id === id)?.title ?? id.replace(/_/g, " ");
}

/** The A-Z success state: replaces the reference/status column the instant a
 * correct sign is confirmed (see PracticePage), so the learner lands on an
 * unmissable celebration instead of technical prediction details. Locking
 * further camera frames while this is shown is handled by the caller
 * (CameraPractice stops its loop; PracticePage sets active=false) -- this
 * component only renders the result, it doesn't gate anything itself. */
export function CelebrationOverlay({
  letter,
  outcome,
  onNext,
  onPracticeAgain,
  nextLabel = "Next Letter →",
  streak,
  confidence,
  encouragement,
  correct,
  lettersCompleted,
}: {
  letter: string;
  outcome: PracticeOutcome;
  onNext: () => void;
  onPracticeAgain?: () => void;
  nextLabel?: string;
  streak?: number;
  confidence?: number | null;
  encouragement?: string;
  /** This letter's total correct count so far (including this attempt) --
   * drives the mastery bar below. Optional so callers that don't track
   * per-letter mastery still render the celebration without it. */
  correct?: number;
  /** How many of the 26 letters have reached mastery (lettersMasteredCount). */
  lettersCompleted?: number;
}) {
  const upperLetter = letter.toUpperCase();
  return (
    <div
      className="rounded-[var(--radius-panel)] border border-[var(--color-success)]/30 bg-[var(--color-panel)] p-6 text-center animate-pop"
      role="status"
      aria-live="polite"
    >
      <p className="text-3xl" aria-hidden="true">
        🎉
      </p>
      <p className="mt-1 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--color-success)]">
        Yes! You got it!
      </p>
      <h2 className="mt-2 font-display text-3xl md:text-4xl">{upperLetter} is correct!</h2>
      {encouragement && <p className="mt-2 text-sm text-[var(--color-mist)]">{encouragement}</p>}
      {outcome.message && <p className="mt-1 text-sm text-[var(--color-mist)]">{outcome.message}</p>}

      <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
        <span className="rounded-full bg-[var(--color-accent)] px-3 py-1 text-sm font-semibold text-[var(--color-ink)]">
          +{outcome.xpGained} XP
        </span>
        {typeof streak === "number" && streak > 0 && (
          <span className="rounded-full border border-[var(--color-warm)]/30 px-3 py-1 text-sm font-medium text-[var(--color-warm)]">
            🔥 {streak}-day streak
          </span>
        )}
        {outcome.leveledUp && (
          <span className="rounded-full border border-[var(--color-warm)]/30 px-3 py-1 text-sm font-medium text-[var(--color-warm)]">
            Level {outcome.newLevel}
          </span>
        )}
        {outcome.dailyCompleted && (
          <span className="rounded-full border border-[var(--color-line)] px-3 py-1 text-sm text-[var(--color-mist)]">
            Daily goal complete
          </span>
        )}
      </div>

      {outcome.badgesUnlocked.length > 0 && (
        <div className="mt-4 text-sm font-medium text-[var(--color-warm)]">
          New badge{outcome.badgesUnlocked.length > 1 ? "s" : ""}: {outcome.badgesUnlocked.map(badgeTitle).join(", ")}
        </div>
      )}

      {typeof correct === "number" && (
        <div className="mx-auto mt-5 max-w-[220px] text-left">
          <LetterMasteryBar correct={correct} />
          {typeof lettersCompleted === "number" && (
            <p className="mt-2 text-center text-xs text-[var(--color-muted)]">
              {lettersCompleted} / 26 letters mastered
            </p>
          )}
        </div>
      )}

      {/* Technical/confidence detail stays present but visually secondary --
          never the dominant success message (see task requirement). */}
      {typeof confidence === "number" && (
        <p className="mt-4 text-xs text-[var(--color-muted)]">Detection confidence: {Math.round(confidence * 100)}%</p>
      )}

      <p className="mt-5 text-sm font-medium text-[var(--color-success)]">✓ {upperLetter} completed</p>

      <div className="mt-4 flex flex-wrap justify-center gap-3">
        {onPracticeAgain && (
          <SecondaryButton onClick={onPracticeAgain} ariaLabel={`Practice ${upperLetter} again`}>
            Practice again
          </SecondaryButton>
        )}
        <PrimaryButton onClick={onNext} ariaLabel="Move to the next letter">
          {nextLabel}
        </PrimaryButton>
      </div>
    </div>
  );
}

export function LevelUpOverlay({
  level,
  onDismiss,
}: {
  level: number;
  onDismiss: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/70 px-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="level-up-title"
    >
      <div className="w-full max-w-sm animate-pop rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel)] p-8 text-center">
        <p className="text-sm font-medium text-[var(--color-warm)]">Level up</p>
        <h2 id="level-up-title" className="mt-2 font-display text-4xl">
          Level {level}
        </h2>
        <p className="mt-3 text-sm text-[var(--color-mist)]">Keep the streak going.</p>
        <button
          type="button"
          className="mt-6 inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] bg-[var(--color-accent)] px-5 py-2 text-sm font-semibold text-[var(--color-ink)]"
          onClick={onDismiss}
        >
          Keep learning
        </button>
      </div>
    </div>
  );
}

export function WordCompleteOverlay({
  word,
  letters,
  xp,
  accuracyLabel,
  onAgain,
  onNext,
  onBack,
  challengeSummary,
}: {
  word: string;
  letters: string[];
  xp: number;
  accuracyLabel: string;
  onAgain: () => void;
  onNext: () => void;
  onBack: () => void;
  challengeSummary?: string;
}) {
  return (
    <div className="rounded-[var(--radius-panel)] border border-[var(--color-success)]/30 bg-[var(--color-panel)] p-6 text-center animate-pop">
      <p className="text-sm font-medium text-[var(--color-success)]">{word} completed!</p>
      <h2 className="mt-2 font-display text-3xl md:text-4xl">{word}</h2>
      <p className="mt-3 flex flex-wrap justify-center gap-3 text-sm text-[var(--color-mist)]" aria-label="Completed letters">
        {letters.map((letter, index) => (
          <span key={`${letter}-${index}`}>
            {letter} ✓
          </span>
        ))}
      </p>
      <p className="mt-3 text-sm text-[var(--color-mist)]">{accuracyLabel}</p>
      {challengeSummary && <p className="mt-1 text-sm text-[var(--color-muted)]">{challengeSummary}</p>}
      <p className="mt-4 text-lg font-semibold text-[var(--color-accent)]">+{xp} XP</p>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        <button
          type="button"
          className="inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-[var(--color-ink)]"
          onClick={onAgain}
        >
          Practice again
        </button>
        <button
          type="button"
          className="inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] border border-[var(--color-line)] px-4 py-2 text-sm font-medium text-[var(--color-mist)]"
          onClick={onNext}
        >
          Next word
        </button>
        <button
          type="button"
          className="inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] px-4 py-2 text-sm text-[var(--color-muted)]"
          onClick={onBack}
        >
          Back to words
        </button>
      </div>
    </div>
  );
}
