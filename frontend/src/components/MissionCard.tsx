import { Card, StatusChip } from "./AppShell";

/** Frames the current letter as a short challenge rather than a raw
 * inference run -- sits above the target/camera cards on PracticePage.
 * Purely presentational: attemptNumber/streak are passed in, nothing here
 * touches game state. */
export function MissionCard({
  challengeLabel,
  attemptNumber,
  streak,
}: {
  challengeLabel: string;
  attemptNumber: number;
  streak: number;
}) {
  return (
    <Card className="flex flex-wrap items-center justify-between gap-3 py-4">
      <div>
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">{challengeLabel}</p>
        <p className="mt-1 text-base font-semibold text-[#eef4f0]">Get this sign correct</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatusChip tone="neutral">Attempt {attemptNumber}</StatusChip>
        {streak > 0 && <StatusChip tone="warning">🔥 {streak}-day streak</StatusChip>}
      </div>
    </Card>
  );
}
