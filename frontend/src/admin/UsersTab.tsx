import { useCallback, useEffect, useState } from "react";
import { Card, PrimaryButton, SecondaryButton, SectionLabel, StatPill, StatusChip } from "../components/AppShell";
import { useAuth } from "../auth/AuthContext";
import {
  deleteAdminUser,
  getAdminUser,
  listAdminUsers,
  patchAdminUser,
  resetAdminUserProgress,
  type AdminUserDetail,
  type AdminUsersPage,
} from "./api";
import { CreateUserDialog } from "./CreateUserDialog";
import { ConfirmDialog, ErrorNote, fmtDate } from "./ui";

type Pending = { kind: "reset" | "delete" | "role" | "active"; user: AdminUserDetail };

export function UsersTab() {
  const { user: me } = useAuth();
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [role, setRole] = useState("");
  const [data, setData] = useState<AdminUsersPage | null>(null);
  const [selected, setSelected] = useState<AdminUserDetail | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const load = useCallback(() => {
    listAdminUsers({ page, search: query, role: role || undefined })
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [page, query, role]);
  useEffect(load, [load]);

  async function open(id: number) {
    setError(null);
    try {
      setSelected(await getAdminUser(id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function apply(p: Pending) {
    const u = p.user;
    if (p.kind === "reset") await resetAdminUserProgress(u.id);
    else if (p.kind === "delete") await deleteAdminUser(u.id, u.username);
    else if (p.kind === "role") await patchAdminUser(u.id, { role: u.role === "admin" ? "student" : "admin" });
    else await patchAdminUser(u.id, { is_active: !u.is_active });
    setPending(null);
    setNotice(`Done: ${p.kind} for ${u.username}. Recorded in the audit log.`);
    if (p.kind === "delete") setSelected(null);
    else setSelected(await getAdminUser(u.id));
    load();
  }

  const isSelf = selected?.id === me?.id;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <SectionLabel>Users</SectionLabel>
          <SecondaryButton onClick={() => setCreating(true)}>Add user</SecondaryButton>
        </div>
        <form
          className="flex flex-wrap gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            setPage(1);
            setQuery(search.trim());
          }}
        >
          <input className="input-field min-w-48 flex-1" placeholder="Search username or email" aria-label="Search users" value={search} onChange={(e) => setSearch(e.target.value)} />
          <select className="input-field !w-auto" aria-label="Filter by role" value={role} onChange={(e) => { setPage(1); setRole(e.target.value); }}>
            <option value="">All roles</option>
            <option value="student">student</option>
            <option value="admin">admin</option>
          </select>
          <PrimaryButton type="submit">Search</PrimaryButton>
        </form>
        <ErrorNote message={error} />
        {notice && <p role="status" className="text-sm font-semibold text-[var(--color-primary-dim)]">{notice}</p>}
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-[var(--color-muted)]">
              <tr>
                <th className="px-2 py-2">User</th>
                <th className="px-2 py-2">Role</th>
                <th className="px-2 py-2">Sessions</th>
                <th className="px-2 py-2">XP</th>
                <th className="px-2 py-2">Last active</th>
              </tr>
            </thead>
            <tbody>
              {data?.items.map((u) => (
                <tr key={u.id} className={`cursor-pointer border-t border-[var(--color-border-soft)] hover:bg-[var(--color-surface-soft)] ${selected?.id === u.id ? "bg-[var(--color-primary-soft)]" : ""}`} onClick={() => void open(u.id)}>
                  <td className="px-2 py-2">
                    <button type="button" className="text-left font-semibold" onClick={() => void open(u.id)}>{u.username}</button>
                    <span className="block text-xs text-[var(--color-muted)]">{u.email}</span>
                  </td>
                  <td className="px-2 py-2">{u.role}</td>
                  <td className="px-2 py-2">{u.practice_sessions}</td>
                  <td className="px-2 py-2">{u.xp}</td>
                  <td className="px-2 py-2">{u.last_activity ? new Date(u.last_activity).toLocaleDateString() : "—"}</td>
                </tr>
              ))}
              {data && data.items.length === 0 && (
                <tr><td colSpan={5} className="px-2 py-6 text-center text-[var(--color-muted)]">No users match.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {data && (
          <div className="flex items-center justify-between text-xs text-[var(--color-muted)]">
            <span>{data.total} users · page {data.page} of {Math.max(1, Math.ceil(data.total / data.page_size))}</span>
            <span className="flex gap-2">
              <SecondaryButton disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</SecondaryButton>
              <SecondaryButton disabled={page * data.page_size >= data.total} onClick={() => setPage((p) => p + 1)}>Next</SecondaryButton>
            </span>
          </div>
        )}
      </Card>

      <Card className="space-y-4" data-testid="user-detail">
        <SectionLabel>User detail</SectionLabel>
        {!selected ? (
          <p className="text-sm text-[var(--color-ink-soft)]">Select a user to see their progress and manage the account.</p>
        ) : (
          <>
            <div>
              <p className="font-display text-2xl font-bold">{selected.username}</p>
              <p className="text-sm text-[var(--color-muted)]">{selected.email} · joined {fmtDate(selected.created_at)}</p>
              <p className="mt-1 flex gap-2">
                <StatusChip tone={selected.role === "admin" ? "accent" : "neutral"}>{selected.role}</StatusChip>
                <StatusChip tone={selected.is_active ? "success" : "warning"}>{selected.is_active ? "active" : "deactivated"}</StatusChip>
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <StatPill icon="⭐" label="XP / level" value={`${selected.xp} / L${selected.level}`} />
              <StatPill icon="🔥" label="Streak (best)" value={`${selected.current_streak} (${selected.best_streak})`} />
              <StatPill icon="🎯" label="Letter accuracy" value={selected.accuracy != null ? `${selected.accuracy}%` : "—"} />
              <StatPill icon="🎮" label="Practice sessions" value={selected.practice_sessions} />
              <StatPill icon="🏅" label="Achievements" value={selected.achievements} />
              <StatPill icon="📝" label="Word / native sessions" value={`${selected.word_sessions} / ${selected.native_sessions}`} />
            </div>
            {selected.recent_activity.length > 0 && (
              <div className="text-xs text-[var(--color-muted)]">
                Recent: {selected.recent_activity.map((a) => `${a.letter}${a.correct ? "✓" : "✗"}`).join("  ")}
              </div>
            )}
            {isSelf && <p className="text-xs text-[var(--color-muted)]">This is you: role, deactivation and deletion are disabled for your own account.</p>}
            <div className="flex flex-wrap gap-3">
              <SecondaryButton disabled={isSelf} onClick={() => setPending({ kind: "role", user: selected })}>
                {selected.role === "admin" ? "Make student" : "Make admin"}
              </SecondaryButton>
              <SecondaryButton disabled={isSelf} onClick={() => setPending({ kind: "active", user: selected })}>
                {selected.is_active ? "Deactivate" : "Reactivate"}
              </SecondaryButton>
              <SecondaryButton onClick={() => setPending({ kind: "reset", user: selected })}>Reset progress</SecondaryButton>
              <SecondaryButton disabled={isSelf} onClick={() => setPending({ kind: "delete", user: selected })}>Delete user</SecondaryButton>
            </div>
          </>
        )}
      </Card>

      {creating && (
        <CreateUserDialog
          onClose={() => setCreating(false)}
          onCreated={(created) => {
            setCreating(false);
            setNotice(`Created ${created.role === "admin" ? "administrator" : "learner"} account for ${created.username}. Recorded in the audit log.`);
            load();
          }}
        />
      )}

      {pending && (
        <ConfirmDialog
          title={
            { reset: "Reset progress?", delete: "Delete this user?", role: "Change role?", active: pending.user.is_active ? "Deactivate account?" : "Reactivate account?" }[pending.kind]
          }
          confirmLabel={{ reset: "Reset progress", delete: "Delete permanently", role: "Change role", active: pending.user.is_active ? "Deactivate" : "Reactivate" }[pending.kind]}
          requireText={pending.kind === "delete" || pending.kind === "reset" ? pending.user.username : undefined}
          onCancel={() => setPending(null)}
          onConfirm={() => apply(pending)}
        >
          {pending.kind === "reset" && <p>Wipes XP, level, streaks, letter/word/native progress, achievements and practice history for <strong>{pending.user.username}</strong>. The account stays. This can't be undone.</p>}
          {pending.kind === "delete" && <p>Permanently deletes <strong>{pending.user.username}</strong> and all their data, including chatbot conversations. This can't be undone.</p>}
          {pending.kind === "role" && <p><strong>{pending.user.username}</strong> will become {pending.user.role === "admin" ? "a student" : "an admin with full access to this area"}.</p>}
          {pending.kind === "active" && <p>{pending.user.is_active ? "They will be signed out of every request and unable to log in." : "They will be able to log in again."}</p>}
          <p>This action is recorded in the audit log.</p>
        </ConfirmDialog>
      )}
    </div>
  );
}
