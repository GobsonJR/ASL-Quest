import { useState, type ReactNode } from "react";
import { PrimaryButton, SecondaryButton } from "../components/AppShell";

export const fmtDate = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : "—");

export function ErrorNote({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
      {message}
    </p>
  );
}

/** Modal confirmation. With `requireText` the person must type that exact text (used for irreversible deletes). */
export function ConfirmDialog({
  title,
  children,
  confirmLabel,
  requireText,
  onConfirm,
  onCancel,
}: {
  title: string;
  children: ReactNode;
  confirmLabel: string;
  requireText?: string;
  onConfirm: () => Promise<void>;
  onCancel: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ready = !requireText || typed === requireText;

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/50 px-4" role="dialog" aria-modal="true" aria-label={title}>
      <div className="w-full max-w-md space-y-4 rounded-2xl bg-[var(--color-surface)] p-6 shadow-xl">
        <h2 className="font-display text-xl font-bold">{title}</h2>
        <div className="space-y-2 text-sm text-[var(--color-ink-soft)]">{children}</div>
        {requireText && (
          <label className="block space-y-1 text-sm">
            <span>
              Type <strong>{requireText}</strong> to confirm
            </span>
            <input className="input-field" value={typed} onChange={(e) => setTyped(e.target.value)} autoFocus />
          </label>
        )}
        <ErrorNote message={error} />
        <div className="flex justify-end gap-3">
          <SecondaryButton onClick={onCancel} disabled={busy}>
            Cancel
          </SecondaryButton>
          <PrimaryButton disabled={!ready || busy} onClick={() => void run()}>
            {busy ? "Working…" : confirmLabel}
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
}
