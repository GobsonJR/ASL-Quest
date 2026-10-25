import { useCallback, useEffect, useState } from "react";
import { Card, PrimaryButton, ProgressBar, SecondaryButton, SectionLabel } from "../components/AppShell";
import { deleteAdminDatasetSession, listAdminDataset, type AdminDatasetSession } from "./api";
import { ConfirmDialog, ErrorNote, fmtDate } from "./ui";

type Data = Awaited<ReturnType<typeof listAdminDataset>>;

export function DatasetTab({ onOpenCollection }: { onOpenCollection: () => void }) {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<AdminDatasetSession | null>(null);

  const load = useCallback(() => {
    listAdminDataset().then(setData).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  if (!data) return <div className="space-y-3"><ErrorNote message={error} />{!error && <p className="text-[var(--color-ink-soft)]">Loading dataset…</p>}</div>;

  const letters = Object.keys(data.totals);
  const total = Object.values(data.totals).reduce((a, b) => a + b, 0);
  const missing = letters.filter((l) => data.totals[l] === 0);

  return (
    <div className="space-y-6">
      <Card className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <SectionLabel>Labelled webcam dataset</SectionLabel>
            <p className="text-sm text-[var(--color-ink-soft)]">
              {total} frames in {data.sessions.length} session{data.sessions.length === 1 ? "" : "s"}
              {missing.length > 0 ? ` · no frames yet for: ${missing.join(" ")}` : " · every letter has data"}.
            </p>
          </div>
          <PrimaryButton onClick={onOpenCollection}>Open the collection workspace</PrimaryButton>
        </div>
        <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {letters.map((l) => (
            <div key={l} className="flex items-center gap-2 text-xs">
              <span className="w-4 font-bold">{l}</span>
              <div className="flex-1"><ProgressBar percent={Math.min(100, (data.totals[l] / data.target_per_letter) * 100)} showPercent={false} size="sm" /></div>
              <span className="w-16 text-right tabular-nums text-[var(--color-muted)]">{data.totals[l]} / {data.target_per_letter}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="space-y-3">
        <SectionLabel>Sessions</SectionLabel>
        <p className="text-xs text-[var(--color-muted)]">Each session is one room/lighting/background and is kept whole when data is split for training and testing. Frames contain faces and rooms, so deleting is permanent.</p>
        <ErrorNote message={error} />
        {data.sessions.length === 0 && <p className="text-sm text-[var(--color-ink-soft)]">No sessions recorded on this server yet.</p>}
        <ul className="space-y-2">
          {data.sessions.map((s) => (
            <li key={s.session_id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
              <span>
                <span className="font-semibold">{s.room || s.session_id}</span> · {s.lighting} · {s.background} · {s.glasses ? "glasses" : "no glasses"}
                <span className="block font-mono text-xs text-[var(--color-muted)]">{s.session_id} · {fmtDate(s.created_at)}</span>
              </span>
              <span className="flex items-center gap-3">
                <span className="text-[var(--color-muted)]">{s.total} frames</span>
                <SecondaryButton onClick={() => setPending(s)}>Delete</SecondaryButton>
              </span>
            </li>
          ))}
        </ul>
      </Card>

      {pending && (
        <ConfirmDialog
          title="Delete this session?"
          confirmLabel="Delete session"
          requireText={pending.session_id}
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            await deleteAdminDatasetSession(pending.session_id);
            setPending(null);
            load();
          }}
        >
          <p>Permanently removes all {pending.total} frames of “{pending.room || pending.session_id}” from disk. Recorded in the audit log.</p>
        </ConfirmDialog>
      )}
    </div>
  );
}
