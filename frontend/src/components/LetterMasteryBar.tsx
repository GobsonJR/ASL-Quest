import { getLetterStatus, getMasteryPercent } from "../game/gamification";
import { MasteryBadge } from "./AppShell";

export function LetterMasteryBar({
  correct,
  showBadge = true,
  showPercent = true,
}: {
  correct: number;
  showBadge?: boolean;
  showPercent?: boolean;
}) {
  const status = getLetterStatus(correct);
  const percent = getMasteryPercent(correct);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        {showBadge && <MasteryBadge status={status} />}
        {showPercent && <span className="text-xs font-semibold text-[var(--color-ink-soft)]">{Math.round(percent)}%</span>}
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-[var(--color-surface-sunken)]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-native)] transition-all duration-700"
          style={{ width: `${percent}%` }}
          role="progressbar"
          aria-valuenow={Math.round(percent)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}
