import { useEffect, useState } from "react";
import { Card, PrimaryButton, SecondaryButton, SectionLabel, StatusChip } from "../components/AppShell";
import {
  listAdminAchievements,
  listAdminNativeSigns,
  patchAdminAchievement,
  patchAdminNativeSign,
  type AdminAchievement,
  type AdminNativeSign,
} from "./api";
import { ErrorNote } from "./ui";

function SignRow({ sign, onSaved }: { sign: AdminNativeSign; onSaved: (s: AdminNativeSign) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ display_name: sign.display_name, meaning: sign.meaning ?? "", description: sign.description ?? "" });
  const [error, setError] = useState<string | null>(null);

  async function save(body: Parameters<typeof patchAdminNativeSign>[1]) {
    setError(null);
    try {
      onSaved(await patchAdminNativeSign(sign.id, body));
      setEditing(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <li className="space-y-2 rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span>
          <span className="font-semibold">{sign.display_name}</span> <span className="text-xs text-[var(--color-muted)]">({sign.gloss}{sign.category ? ` · ${sign.category}` : ""})</span>
        </span>
        <span className="flex items-center gap-2">
          <StatusChip tone={sign.active ? "success" : "warning"}>{sign.active ? "visible to learners" : "hidden"}</StatusChip>
          <SecondaryButton onClick={() => void save({ active: !sign.active })}>{sign.active ? "Hide" : "Show"}</SecondaryButton>
          <SecondaryButton onClick={() => setEditing((v) => !v)}>{editing ? "Close" : "Edit"}</SecondaryButton>
        </span>
      </div>
      {editing && (
        <div className="space-y-2">
          <label className="block text-xs font-semibold">Display name
            <input className="input-field mt-1" value={draft.display_name} onChange={(e) => setDraft({ ...draft, display_name: e.target.value })} />
          </label>
          <label className="block text-xs font-semibold">Meaning
            <input className="input-field mt-1" value={draft.meaning} onChange={(e) => setDraft({ ...draft, meaning: e.target.value })} />
          </label>
          <label className="block text-xs font-semibold">How to sign it
            <textarea className="input-field mt-1" rows={3} value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} />
          </label>
          <PrimaryButton disabled={!draft.display_name.trim()} onClick={() => void save({ display_name: draft.display_name.trim(), meaning: draft.meaning, description: draft.description })}>Save changes</PrimaryButton>
        </div>
      )}
      <ErrorNote message={error} />
    </li>
  );
}

function AchievementRow({ item, onSaved }: { item: AdminAchievement; onSaved: (a: AdminAchievement) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ name: item.name, description: item.description });
  const [error, setError] = useState<string | null>(null);

  async function save() {
    setError(null);
    try {
      const saved = await patchAdminAchievement(item.id, { name: draft.name.trim(), description: draft.description.trim() });
      onSaved({ ...item, ...saved });
      setEditing(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <li className="space-y-2 rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span>
          <span className="font-semibold">{item.name}</span>
          <span className="block text-xs text-[var(--color-muted)]">{item.description} · unlocks when: {item.requirement} · earned by {item.unlocked_by}</span>
        </span>
        <SecondaryButton onClick={() => setEditing((v) => !v)}>{editing ? "Close" : "Edit text"}</SecondaryButton>
      </div>
      {editing && (
        <div className="space-y-2">
          <input className="input-field" aria-label="Achievement name" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          <textarea className="input-field" aria-label="Achievement description" rows={2} value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} />
          <PrimaryButton disabled={!draft.name.trim() || !draft.description.trim()} onClick={() => void save()}>Save</PrimaryButton>
        </div>
      )}
      <ErrorNote message={error} />
    </li>
  );
}

export function ContentTab() {
  const [signs, setSigns] = useState<AdminNativeSign[] | null>(null);
  const [achievements, setAchievements] = useState<AdminAchievement[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([listAdminNativeSigns(), listAdminAchievements()])
      .then(([s, a]) => {
        setSigns(s.items);
        setAchievements(a.items);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="space-y-6">
      <ErrorNote message={error} />
      <Card className="space-y-3">
        <SectionLabel>Native signs</SectionLabel>
        <p className="text-xs text-[var(--color-muted)]">Hiding a sign removes it from the learner-facing native signs list and practice. Nothing is deleted.</p>
        {signs && signs.length === 0 && <p className="text-sm text-[var(--color-ink-soft)]">No native signs are registered yet.</p>}
        <ul className="space-y-2">
          {signs?.map((s) => <SignRow key={s.id} sign={s} onSaved={(n) => setSigns((all) => all!.map((x) => (x.id === n.id ? n : x)))} />)}
        </ul>
      </Card>
      <Card className="space-y-3">
        <SectionLabel>Achievements</SectionLabel>
        <p className="text-xs text-[var(--color-muted)]">Name and description can be edited. Unlock rules are defined in code and aren't editable here.</p>
        <ul className="space-y-2">
          {achievements?.map((a) => <AchievementRow key={a.id} item={a} onSaved={(n) => setAchievements((all) => all!.map((x) => (x.id === n.id ? n : x)))} />)}
        </ul>
      </Card>
      <Card>
        <SectionLabel>Spelling words</SectionLabel>
        <p className="mt-2 text-sm text-[var(--color-ink-soft)]">
          The spelling-word catalog is defined in code (backend/words/catalog.py), so it can't be edited from here without changing what learners see inconsistently. Word statistics are on the Overview tab.
        </p>
      </Card>
    </div>
  );
}
