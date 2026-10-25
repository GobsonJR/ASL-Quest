import { useState } from "react";
import { Dialog } from "../components/Dialog";
import { PrimaryButton, SecondaryButton } from "../components/AppShell";
import { createAdminUser, type CreatedUser } from "./api";
import { ErrorNote } from "./ui";

/** An administrator creates an account with an explicit role. The role is sent to the admin-only endpoint, which checks the
 * caller's own role on the server; ordinary sign-up has no role field at all and always makes a regular learner. */
export function CreateUserDialog({ onCreated, onClose }: { onCreated: (user: CreatedUser) => void; onClose: () => void }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"student" | "admin">("student");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ready = username.trim().length >= 3 && /\S+@\S+\.\S+/.test(email) && password.length >= 8;

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      onCreated(await createAdminUser({ username: username.trim(), email: email.trim(), password, role }));
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <Dialog labelledBy="create-user-title" onClose={onClose} className="w-full max-w-md">
      <form
        className="space-y-4 rounded-2xl bg-[var(--color-surface)] p-6 shadow-xl"
        onSubmit={(event) => {
          event.preventDefault();
          if (ready && !busy) void submit();
        }}
      >
        <h2 id="create-user-title" className="font-display text-xl font-bold">Add a user</h2>
        <label className="block space-y-1 text-sm">
          <span className="font-semibold">Username (3+ characters)</span>
          <input className="input-field" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="off" />
        </label>
        <label className="block space-y-1 text-sm">
          <span className="font-semibold">Email</span>
          <input className="input-field" type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="off" />
        </label>
        <label className="block space-y-1 text-sm">
          <span className="font-semibold">Temporary password (8+ characters)</span>
          <input className="input-field" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" />
        </label>
        <fieldset className="space-y-1 text-sm">
          <legend className="font-semibold">Role</legend>
          <label className="flex items-center gap-2"><input type="radio" name="new-role" checked={role === "student"} onChange={() => setRole("student")} /> Learner</label>
          <label className="flex items-center gap-2"><input type="radio" name="new-role" checked={role === "admin"} onChange={() => setRole("admin")} /> Administrator (full access to this workspace)</label>
        </fieldset>
        <p className="text-xs text-[var(--color-muted)]">Recorded in the audit log. The password is never stored in it.</p>
        <ErrorNote message={error} />
        <div className="flex justify-end gap-3">
          <SecondaryButton onClick={onClose} disabled={busy}>Cancel</SecondaryButton>
          <PrimaryButton type="submit" disabled={!ready || busy}>{busy ? "Creating…" : "Create user"}</PrimaryButton>
        </div>
      </form>
    </Dialog>
  );
}
