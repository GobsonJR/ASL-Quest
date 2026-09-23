import { Card, SectionLabel, StatusChip } from "./AppShell";
import { REFERENCE_LICENSE_NOTE, referenceImageAlt, referenceImageSrc } from "../pages/alphabetFeedback";

/** The "before/during detection" reference card: a large, dynamically-updating
 * ASL handshape illustration for the current target letter, plus a live status
 * chip. Deliberately the visual anchor of the right-hand column -- see
 * PracticePage, which swaps it out for CelebrationOverlay once the learner
 * gets the letter right. */
export function AlphabetReferencePanel({
  letter,
  statusLabel,
  statusTone = "accent",
}: {
  letter: string;
  statusLabel?: string;
  statusTone?: "neutral" | "success" | "warning" | "accent";
}) {
  return (
    <Card className="space-y-4 text-center" data-testid="alphabet-reference-panel">
      <SectionLabel>Reference</SectionLabel>

      <div className="mx-auto w-full max-w-[220px] overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-white p-3">
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
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">Show this sign</p>
        <p className="mt-1 font-display text-5xl text-[var(--color-accent)]">{letter.toUpperCase()}</p>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-mist)]">
          Hold this handshape in front of the camera.
        </p>
      </div>

      {statusLabel && (
        <div className="flex justify-center" aria-live="polite">
          <StatusChip tone={statusTone}>{statusLabel}</StatusChip>
        </div>
      )}

      <p className="text-[0.65rem] leading-snug text-[var(--color-muted)]">{REFERENCE_LICENSE_NOTE}</p>
    </Card>
  );
}
