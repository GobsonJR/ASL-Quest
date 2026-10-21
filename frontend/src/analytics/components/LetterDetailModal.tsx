import { Card, PrimaryButton, SectionLabel } from "../../components/AppShell";
import type { LetterDetail } from "../api";
import { LineChart } from "./charts";

export function LetterDetailModal({
  detail,
  loading,
  onClose,
  onPractice,
}: {
  detail: LetterDetail | null;
  loading: boolean;
  onClose: () => void;
  onPractice: (letter: string) => void;
}) {
  if (!detail && !loading) return null;

  return (
    <div className="fixed inset-0 z-50 grid place-items-end bg-black/70 px-4 py-6 backdrop-blur-sm md:place-items-center" role="dialog" aria-modal="true">
      <Card className="max-h-[85vh] w-full max-w-2xl overflow-y-auto">
        {loading || !detail ? (
          <p className="text-[var(--color-ink-soft)]">Loading letter analytics...</p>
        ) : (
          <>
            <div className="flex items-start justify-between gap-4">
              <div>
                <SectionLabel>Letter Detail</SectionLabel>
                <h2 className="mt-2 font-display text-4xl text-[var(--color-primary)]">{detail.letter}</h2>
                <p className="mt-1 text-sm text-[var(--color-ink-soft)]">{detail.status}</p>
              </div>
              <button type="button" className="text-sm text-[var(--color-ink-soft)]" onClick={onClose}>
                Close
              </button>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ["Mastery", `${detail.mastery_percent}%`],
                ["Accuracy", detail.accuracy != null ? `${detail.accuracy}%` : "—"],
                ["Attempts", String(detail.attempts)],
                ["Avg response", detail.average_response_sec != null ? `${detail.average_response_sec}s` : "—"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
                  <p className="text-xs text-[var(--color-muted)]">{label}</p>
                  <p className="mt-1 text-xl font-bold">{value}</p>
                </div>
              ))}
            </div>

            <div className="mt-6">
              <LineChart
                title="Accuracy trend (30 days)"
                points={detail.accuracy_trend}
                valueSuffix="%"
                emptyMessage="Keep practicing this letter to unlock trends."
              />
            </div>

            <div className="mt-6">
              <SectionLabel>Recent attempts</SectionLabel>
              <div className="mt-3 space-y-2">
                {detail.history.length === 0 && (
                  <p className="text-sm text-[var(--color-ink-soft)]">No attempts recorded for this letter yet.</p>
                )}
                {detail.history.map((item) => (
                  <div key={item.id} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                    <span>{new Date(item.date).toLocaleString()}</span>
                    <span>{item.prediction ?? "—"}</span>
                    <span className={item.correct ? "text-[var(--color-success)]" : "text-[var(--color-streak)]"}>
                      {item.correct ? "Correct" : "Incorrect"}
                    </span>
                    <span>{item.response_time_sec != null ? `${item.response_time_sec}s` : "—"}</span>
                    <span>{item.xp_earned > 0 ? `+${item.xp_earned} XP` : "0 XP"}</span>
                  </div>
                ))}
              </div>
            </div>

            <PrimaryButton className="mt-6" onClick={() => onPractice(detail.letter)}>
              Practice {detail.letter}
            </PrimaryButton>
          </>
        )}
      </Card>
    </div>
  );
}
