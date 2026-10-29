import { type FormEvent, useEffect, useState } from "react";
import {
  changePassword,
  deleteAccount,
  exportUserData,
  fetchPreferences,
  logoutUser,
  updatePreferences,
} from "../auth/api";
import { useAuth } from "../auth/AuthContext";
import {
  Divider,
  PageHeader,
  PageLayout,
  Panel,
  PrimaryButton,
  SecondaryButton,
  SectionHeader,
} from "../components/AppShell";
import { useToast } from "../components/Toast";
import { isSoundEnabled, playCelebrationSound, setSoundEnabled } from "../utils/soundEffects";
import { generateProgressPdf } from "../utils/exportPdf";

export function SettingsPage() {
  const { logout } = useAuth();
  const { push } = useToast();
  const [reducedMotion, setReducedMotion] = useState(false);
  const [soundEnabled, setSoundEnabledState] = useState(() => isSoundEnabled());
  const [exporting, setExporting] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState("");

  useEffect(() => {
    fetchPreferences()
      .then((prefs) => setReducedMotion(Boolean(prefs.reduced_motion)))
      .catch(() => undefined);
  }, []);

  async function savePreferences() {
    await updatePreferences({ reduced_motion: reducedMotion });
    document.documentElement.classList.toggle("reduce-motion", reducedMotion);
    push("Preferences saved", "success");
  }

  async function handlePasswordChange(event: FormEvent) {
    event.preventDefault();
    try {
      await changePassword(currentPassword, newPassword);
      push("Password updated", "success");
      setCurrentPassword("");
      setNewPassword("");
    } catch (error) {
      push(error instanceof Error ? error.message : "Password change failed", "error");
    }
  }

  async function handleExport() {
    setExporting(true);
    try {
      const data = await exportUserData();
      generateProgressPdf(data);
      push("Learning progress PDF downloaded!", "success");
    } catch (error) {
      console.error("Export error:", error);
      push("Export failed", "error");
    } finally {
      setExporting(false);
    }
  }

  async function handleJsonExport() {
    try {
      const data = await exportUserData();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "asl-quest-raw-data.json";
      link.click();
      URL.revokeObjectURL(url);
      push("Raw JSON data downloaded", "info");
    } catch {
      push("Export failed", "error");
    }
  }

  async function handleDelete(event: FormEvent) {
    event.preventDefault();
    try {
      await deleteAccount(deletePassword, deleteConfirm);
      logoutUser();
      logout();
      push("Account deleted", "info");
    } catch (error) {
      push(error instanceof Error ? error.message : "Account deletion failed", "error");
    }
  }

  return (
    <PageLayout>
      <PageHeader
        eyebrow="Settings"
        title="Account & preferences"
        description="Manage how ASL Quest works for you."
      />

      <section className="space-y-4">
        <SectionHeader title="Preferences" />
        <label className="flex items-center gap-3 text-sm text-[var(--color-ink-soft)]">
          <input type="checkbox" checked={reducedMotion} onChange={(event) => setReducedMotion(event.target.checked)} />
          Reduce motion
        </label>
        <PrimaryButton onClick={savePreferences}>Save preferences</PrimaryButton>
      </section>

      <Divider />

      <section className="space-y-4">
        <SectionHeader title="Security" />
        <form className="max-w-md space-y-3" onSubmit={handlePasswordChange}>
          <input className="input-field" type="password" placeholder="Current password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} required />
          <input className="input-field" type="password" placeholder="New password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required minLength={8} />
          <PrimaryButton type="submit">Change password</PrimaryButton>
        </form>
        <SecondaryButton onClick={() => { logout(); push("Logged out", "info"); }}>Logout</SecondaryButton>
      </section>

      <Divider />

      <section className="space-y-4">
        <SectionHeader
          title="Sound & Celebrations"
          description="Audio feedback when you get signs right and build streaks (Duolingo style)."
        />
        <div className="space-y-3">
          <label className="flex items-center gap-3 text-sm text-[var(--color-ink-soft)] cursor-pointer">
            <input
              type="checkbox"
              checked={soundEnabled}
              onChange={(event) => {
                const checked = event.target.checked;
                setSoundEnabledState(checked);
                setSoundEnabled(checked);
                push(checked ? "Celebration sound enabled" : "Celebration sound muted", "info");
              }}
            />
            <span>Play cheerful chime and streak celebration sound effects</span>
          </label>
          {soundEnabled && (
            <div className="pt-1">
              <SecondaryButton
                type="button"
                onClick={() => {
                  playCelebrationSound({ streak: 5 });
                  push("Playing preview celebration chime!", "info");
                }}
              >
                🔊 Test celebration sound
              </SecondaryButton>
            </div>
          )}
        </div>
      </section>

      <Divider />

      <section className="space-y-4">
        <SectionHeader
          title="Your data"
          description="Download your learning progress report in a clean, human-readable PDF with executive summary and mastery breakdown."
        />
        <div className="flex flex-wrap items-center gap-3">
          <PrimaryButton onClick={handleExport} disabled={exporting}>
            {exporting ? "Generating PDF..." : "📄 Export my progress (PDF)"}
          </PrimaryButton>
          <button
            type="button"
            onClick={handleJsonExport}
            className="text-xs text-[var(--color-muted)] hover:underline"
          >
            Download raw JSON
          </button>
        </div>
      </section>

      <Divider />

      <Panel className="border-[var(--color-danger)]/25 bg-[var(--color-danger)]/[0.03]">
        <SectionHeader title="Danger zone" description="Permanently delete your account and all learning data." />
        <form className="mt-4 max-w-md space-y-3" onSubmit={handleDelete}>
          <input className="input-field" type="password" placeholder="Password" value={deletePassword} onChange={(e) => setDeletePassword(e.target.value)} required />
          <input className="input-field" placeholder="Type DELETE to confirm" value={deleteConfirm} onChange={(e) => setDeleteConfirm(e.target.value)} required />
          <button
            type="submit"
            className="inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] bg-[var(--color-danger)] px-4 py-2 text-sm font-semibold text-white"
          >
            Delete account
          </button>
        </form>
      </Panel>
    </PageLayout>
  );
}
