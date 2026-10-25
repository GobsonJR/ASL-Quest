import { useCallback, useEffect, useState } from "react";
import { Card, SecondaryButton, SectionLabel } from "../components/AppShell";
import { ConfirmDialog } from "../admin/ui";
import {
  clipVideoUrl,
  deleteNativeClip,
  deleteSample,
  listNativeClips,
  listSamples,
  sampleImageUrl,
  type ClipRecord,
  type SampleRecord,
  type SessionInfo,
} from "./api";

type Item =
  | { kind: "image"; id: string; label: string; meta: string; url: string | null }
  | { kind: "clip"; id: string; label: string; meta: string; url: string | null };

/** Review gallery for what the current session has saved for ONE class (a letter or a native sign): thumbnails/players, who/when,
 * and a deliberate, confirmed delete. Files are admin-only, so they are fetched with the token and shown from blob URLs. */
export function SampleGallery({
  session,
  mode,
  label,
  onChanged,
}: {
  session: SessionInfo;
  mode: "letters" | "native";
  /** The letter (A-Z) or the sign gloss to show. */
  label: string | null;
  onChanged: (session: SessionInfo) => void;
}) {
  const [items, setItems] = useState<Item[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Item | null>(null);
  // the saved total changes whenever something is added or removed, which is when the list must be re-read
  const version = mode === "letters" ? (label ? session.counts[label] ?? 0 : 0) : label ? session.native_counts?.[label] ?? 0 : 0;

  const load = useCallback(async () => {
    if (!label) {
      setItems(null);
      return;
    }
    setError(null);
    try {
      if (mode === "letters") {
        const { samples } = await listSamples(session.session_id, label);
        setItems(await Promise.all(samples.map((s) => toImageItem(session.session_id, s))));
      } else {
        const { clips } = await listNativeClips(session.session_id, label);
        setItems(await Promise.all(clips.map((c) => toClipItem(session.session_id, c))));
      }
    } catch (err) {
      setError((err as Error).message);
      setItems([]);
    }
  }, [label, mode, session.session_id]);

  useEffect(() => {
    void load();
  }, [load, version]);

  // free the blob URLs when the list is replaced or the gallery closes
  useEffect(() => {
    return () => {
      items?.forEach((item) => item.url && URL.revokeObjectURL(item.url));
    };
  }, [items]);

  async function remove(item: Item) {
    const reply =
      item.kind === "image" ? await deleteSample(session.session_id, item.id) : await deleteNativeClip(session.session_id, item.id);
    onChanged(reply.session);
  }

  return (
    <Card className="space-y-3" data-testid="sample-gallery">
      <SectionLabel>{label ? `Saved in this session: ${label}` : "Saved in this session"}</SectionLabel>
      {!label && <p className="text-sm text-[var(--color-ink-soft)]">Choose a {mode === "letters" ? "letter" : "sign"} to review what has been saved for it.</p>}
      {error && <p role="alert" className="text-sm font-semibold text-[var(--color-danger)]">{error}</p>}
      {label && items === null && <p className="text-sm text-[var(--color-ink-soft)]">Loading…</p>}
      {label && items?.length === 0 && !error && (
        <p className="text-sm text-[var(--color-ink-soft)]">Nothing saved for {label} in this session yet.</p>
      )}
      {items && items.length > 0 && (
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3" aria-label={`Saved samples for ${label}`}>
          {items.map((item) => (
            <li key={item.id} className="space-y-1 rounded-xl border border-[var(--color-border-soft)] p-1.5">
              {item.url ? (
                item.kind === "image" ? (
                  <img src={item.url} alt={`${item.label}, ${item.meta}`} className="aspect-[4/3] w-full rounded-lg object-cover" />
                ) : (
                  <video src={item.url} controls playsInline preload="metadata" aria-label={`${item.label} clip, ${item.meta}`} className="aspect-[4/3] w-full rounded-lg bg-black" />
                )
              ) : (
                <div className="grid aspect-[4/3] w-full place-items-center rounded-lg bg-[var(--color-surface-soft)] text-xs text-[var(--color-muted)]">unavailable</div>
              )}
              <p className="truncate text-[0.7rem] text-[var(--color-muted)]">{item.meta}</p>
              <SecondaryButton onClick={() => setPending(item)} ariaLabel={`Delete this ${item.label} sample`}>Delete</SecondaryButton>
            </li>
          ))}
        </ul>
      )}

      {pending && (
        <ConfirmDialog
          title={`Delete this ${pending.label} ${pending.kind === "clip" ? "clip" : "image"}?`}
          confirmLabel="Delete permanently"
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            await remove(pending);
            setPending(null);
          }}
        >
          <p>This removes the file and its label from disk and is recorded in the audit log. It cannot be undone.</p>
        </ConfirmDialog>
      )}
    </Card>
  );
}

async function toImageItem(sessionId: string, sample: SampleRecord): Promise<Item> {
  let url: string | null = null;
  try {
    url = await sampleImageUrl(sessionId, sample.sample_id);
  } catch {
    /* shown as "unavailable" */
  }
  return {
    kind: "image",
    id: sample.sample_id,
    label: sample.letter,
    meta: `${new Date(sample.captured_at).toLocaleString()} · ${sample.hand} hand · ${sample.distance}${sample.mode === "hold" ? " · hold" : ""}`,
    url,
  };
}

async function toClipItem(sessionId: string, clip: ClipRecord): Promise<Item> {
  let url: string | null = null;
  try {
    url = await clipVideoUrl(sessionId, clip.sample_id);
  } catch {
    /* shown as "unavailable" */
  }
  return {
    kind: "clip",
    id: clip.sample_id,
    label: clip.gloss,
    meta: `${new Date(clip.captured_at).toLocaleString()}${clip.duration_ms ? ` · ${(clip.duration_ms / 1000).toFixed(1)} s` : ""}${clip.hand ? ` · ${clip.hand} hand` : ""}`,
    url,
  };
}
