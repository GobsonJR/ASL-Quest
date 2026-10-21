import { Card, PrimaryButton, SecondaryButton } from "./AppShell";
import { Dialog } from "./Dialog";

export function MigrationPrompt({
  onImport,
  onDismiss,
  loading,
}: {
  onImport: () => void;
  onDismiss: () => void;
  loading?: boolean;
}) {
  return (
    <Dialog labelledBy="migration-title" onClose={onDismiss} className="w-full max-w-lg">
      <Card className="w-full space-y-4">
        <h2 id="migration-title" className="font-display text-2xl">Import local progress?</h2>
        <p className="text-sm text-[var(--color-ink-soft)]">
          We found ASL Quest progress saved in this browser. Your account is empty, so you can import it now.
          If you skip this, your server account will stay fresh and local progress remains as a cache only.
        </p>
        <div className="flex flex-wrap gap-3">
          <PrimaryButton onClick={onImport} disabled={loading}>
            {loading ? "Importing..." : "Import progress"}
          </PrimaryButton>
          <SecondaryButton onClick={onDismiss}>Keep server progress</SecondaryButton>
        </div>
      </Card>
    </Dialog>
  );
}
