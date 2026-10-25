import { useState } from "react";
import { Card, PrimaryButton, SectionLabel } from "../components/AppShell";
import { exportDataset } from "./api";

/** Downloads the labelled dataset as a zip. Export is collection-only: it never trains, evaluates or promotes a model. */
export function ExportCard({ sessionId, total }: { sessionId?: string; total: number }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await exportDataset(sessionId);
      const url = URL.createObjectURL(result.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `asl_quest_dataset${sessionId ? `_${sessionId}` : ""}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setMessage(
        `Exported ${result.files} file${result.files === 1 ? "" : "s"}.` +
          (result.skippedSessions > 0 ? ` ${result.skippedSessions} older session${result.skippedSessions === 1 ? " was" : "s were"} left out because no contributor consent was recorded.` : "")
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="space-y-3" data-testid="export-card">
      <SectionLabel>Export the labelled dataset</SectionLabel>
      <p className="text-sm text-[var(--color-ink-soft)]">
        A zip with <code>images/&lt;LETTER&gt;/</code>, <code>native_clips/&lt;SIGN&gt;/</code>, <code>negatives/</code>, a <code>manifest.csv</code> and{" "}
        <code>manifest.jsonl</code> (label, session, capture time, hand, distance, contributor, consent) and a README. Only sessions with recorded consent are included.
      </p>
      <ul className="list-disc space-y-1 pl-5 text-xs text-[var(--color-muted)]">
        <li>Files can show identifiable people and their rooms: share them only as the contributors agreed, and delete them on request.</li>
        <li>Exporting does not train, evaluate or replace any model, and it never changes the existing training dataset.</li>
      </ul>
      <div className="flex flex-wrap items-center gap-3">
        <PrimaryButton disabled={busy || total === 0} onClick={() => void run()}>
          {busy ? "Preparing…" : sessionId ? "Export this session (.zip)" : "Export everything (.zip)"}
        </PrimaryButton>
        {total === 0 && <span className="text-xs text-[var(--color-muted)]">Nothing has been saved yet.</span>}
      </div>
      {message && <p role="status" className="text-sm font-semibold text-[var(--color-primary-dim)]">{message}</p>}
      {error && <p role="alert" className="text-sm font-semibold text-[var(--color-danger)]">{error}</p>}
    </Card>
  );
}
