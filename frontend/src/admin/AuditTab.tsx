import { useEffect, useState } from "react";
import { Card, SecondaryButton, SectionLabel } from "../components/AppShell";
import { listAdminAudit } from "./api";
import { ErrorNote, fmtDate } from "./ui";

type Data = Awaited<ReturnType<typeof listAdminAudit>>;

const ACTIONS = [
  "user.role", "user.activate", "user.deactivate", "user.reset_progress", "user.delete",
  "native_sign.update", "achievement.update", "dataset.delete_session",
];

export function AuditTab() {
  const [page, setPage] = useState(1);
  const [action, setAction] = useState("");
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listAdminAudit({ page, action: action || undefined }).then(setData).catch((e: Error) => setError(e.message));
  }, [page, action]);

  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionLabel>Audit log</SectionLabel>
        <select className="input-field !w-auto" aria-label="Filter by action" value={action} onChange={(e) => { setPage(1); setAction(e.target.value); }}>
          <option value="">All actions</option>
          {ACTIONS.map((a) => <option key={a}>{a}</option>)}
        </select>
      </div>
      <p className="text-xs text-[var(--color-muted)]">Every admin change is recorded here in the same step as the change itself. Entries can't be edited or removed from the app.</p>
      <ErrorNote message={error} />
      {data && data.items.length === 0 && <p className="text-sm text-[var(--color-ink-soft)]">No admin actions recorded yet.</p>}
      <ul className="space-y-2">
        {data?.items.map((e) => (
          <li key={e.id} className="rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
            <div className="flex flex-wrap justify-between gap-2">
              <span><span className="font-semibold">{e.admin}</span> · <code>{e.action}</code> · {e.target_type} {e.target_id}</span>
              <span className="text-xs text-[var(--color-muted)]">{fmtDate(e.created_at)}</span>
            </div>
            {Object.keys(e.detail).length > 0 && <pre className="mt-1 whitespace-pre-wrap break-words text-xs text-[var(--color-muted)]">{JSON.stringify(e.detail)}</pre>}
          </li>
        ))}
      </ul>
      {data && (
        <div className="flex items-center justify-between text-xs text-[var(--color-muted)]">
          <span>{data.total} entries</span>
          <span className="flex gap-2">
            <SecondaryButton disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Newer</SecondaryButton>
            <SecondaryButton disabled={page * data.page_size >= data.total} onClick={() => setPage((p) => p + 1)}>Older</SecondaryButton>
          </span>
        </div>
      )}
    </Card>
  );
}
