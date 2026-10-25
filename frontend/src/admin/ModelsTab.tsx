import { useEffect, useState } from "react";
import { Card, SectionLabel, StatPill, StatusChip } from "../components/AppShell";
import { getAdminChatFeedback, listAdminModels, listAdminPredictions, type AdminChatFeedback, type AdminModelRow, type AdminPrediction } from "./api";
import { ErrorNote, fmtDate } from "./ui";

export function ModelsTab() {
  const [models, setModels] = useState<AdminModelRow[] | null>(null);
  const [predictions, setPredictions] = useState<AdminPrediction[] | null>(null);
  const [chat, setChat] = useState<AdminChatFeedback | null>(null);
  const [negativeOnly, setNegativeOnly] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([listAdminModels(), listAdminPredictions()])
      .then(([m, p]) => {
        setModels(m.items);
        setPredictions(p.items);
      })
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(() => {
    getAdminChatFeedback(negativeOnly).then(setChat).catch((e: Error) => setError(e.message));
  }, [negativeOnly]);

  return (
    <div className="space-y-6">
      <ErrorNote message={error} />
      <Card className="space-y-3">
        <SectionLabel>Recognition models</SectionLabel>
        <p className="text-xs text-[var(--color-muted)]">Read-only. Accuracy is over predictions the app could grade against the expected sign.</p>
        {models && models.length === 0 && <p className="text-sm text-[var(--color-ink-soft)]">No model versions are registered and no predictions have been logged yet.</p>}
        <div className="overflow-x-auto">
          {models && models.length > 0 && (
            <table className="min-w-full text-left text-sm">
              <thead className="text-[var(--color-muted)]"><tr><th className="px-2 py-2">Model</th><th className="px-2 py-2">Predictions</th><th className="px-2 py-2">Accuracy</th><th className="px-2 py-2">Latency</th><th className="px-2 py-2">Registered</th></tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={`${m.model_type}-${m.version}`} className="border-t border-[var(--color-border-soft)]">
                    <td className="px-2 py-2 font-semibold">{m.model_type} {m.version}{m.num_classes ? <span className="font-normal text-[var(--color-muted)]"> · {m.num_classes} classes</span> : null}</td>
                    <td className="px-2 py-2">{m.predictions}</td>
                    <td className="px-2 py-2">{m.accuracy != null ? `${m.accuracy}% (${m.graded})` : "—"}</td>
                    <td className="px-2 py-2">{m.avg_latency_ms != null ? `${m.avg_latency_ms} ms` : "—"}</td>
                    <td className="px-2 py-2"><StatusChip tone={m.registered ? "success" : "neutral"}>{m.registered ? fmtDate(m.created_at) : "not registered"}</StatusChip></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {predictions && predictions.length > 0 && (
          <details>
            <summary className="cursor-pointer text-sm font-semibold">Latest {predictions.length} predictions</summary>
            <ul className="mt-2 space-y-1 text-xs">
              {predictions.map((p, i) => (
                <li key={i} className="flex justify-between rounded-lg border border-[var(--color-border-soft)] px-2 py-1">
                  <span>User #{p.user_id} · expected {p.expected ?? "?"} → got {p.predicted}{p.confidence != null ? ` (${Math.round(p.confidence * 100)}%)` : ""}</span>
                  <span>{p.correct == null ? "ungraded" : p.correct ? "✓" : "✗"} · {fmtDate(p.created_at)}</span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </Card>

      <Card className="space-y-3">
        <SectionLabel>Chatbot feedback</SectionLabel>
        {chat && (
          <div className="grid gap-3 sm:grid-cols-4">
            <StatPill icon="💬" label="Conversations" value={chat.summary.conversations} />
            <StatPill icon="✉️" label="Messages" value={chat.summary.messages} />
            <StatPill icon="👍" label="Helpful" value={chat.summary.positive} />
            <StatPill icon="👎" label="Not helpful" value={chat.summary.negative} />
          </div>
        )}
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={negativeOnly} onChange={(e) => setNegativeOnly(e.target.checked)} /> Only answers rated “not helpful”
        </label>
        <p className="text-xs text-[var(--color-muted)]">Only messages a learner explicitly rated are shown; full conversations stay private.</p>
        {chat && chat.items.length === 0 && <p className="text-sm text-[var(--color-ink-soft)]">Nothing to review.</p>}
        <ul className="space-y-2">
          {chat?.items.map((item, i) => (
            <li key={i} className="space-y-1 rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
              <p><span className="font-semibold">Asked:</span> {item.question ?? "—"}</p>
              <p><span className="font-semibold">Answer:</span> {item.answer}</p>
              <p className="text-xs text-[var(--color-muted)]">{item.rating && item.rating > 0 ? "👍" : "👎"}{item.comment ? ` “${item.comment}”` : ""} · {fmtDate(item.created_at)}</p>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
