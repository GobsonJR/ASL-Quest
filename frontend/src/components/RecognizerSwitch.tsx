import type { ExperimentalLandmarkInfo, RecognizerChoice } from "../api";

/** Lets a developer compare the production ResNet with the experimental landmark recognizer on live camera frames.
 * Renders nothing unless the backend reports an experimental model (ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL), so ordinary
 * deployments never see it. ResNet is always the starting choice and there is always a one-click way back to it. */
export function RecognizerSwitch({
  info,
  value,
  onChange,
}: {
  info: ExperimentalLandmarkInfo | null | undefined;
  value: RecognizerChoice;
  onChange: (next: RecognizerChoice) => void;
}) {
  if (!info?.loaded) return null;
  const experimental = value === "landmarks";
  return (
    <section
      aria-label="Recognizer comparison (experimental)"
      className={`space-y-2 rounded-2xl border-2 px-4 py-3 text-sm ${experimental ? "border-[var(--color-warning,#d97706)] bg-[var(--color-warning-soft,#fff4d6)]" : "border-[var(--color-border-soft)] bg-[var(--color-surface-soft)]"}`}
    >
      <p className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">Recognizer · developer comparison</p>
      <div role="radiogroup" aria-label="Recognizer" className="flex flex-wrap gap-x-6 gap-y-1">
        <label className="flex items-center gap-2">
          <input type="radio" name="recognizer" checked={!experimental} onChange={() => onChange("resnet")} />
          <span>
            <span className="font-semibold">ResNet</span> (default, production)
          </span>
        </label>
        <label className="flex items-center gap-2">
          <input type="radio" name="recognizer" checked={experimental} onChange={() => onChange("landmarks")} />
          <span>
            <span className="font-semibold">Hand landmarks</span> (experimental · {info.model_file})
          </span>
        </label>
      </div>
      {experimental && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3">
          <p>
            <strong>Experimental recognizer active.</strong> Based on a small preliminary test only; it is not validated for general use. Practice results
            still count toward your progress.
          </p>
          <button
            type="button"
            className="rounded-xl border-2 border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 font-semibold"
            onClick={() => onChange("resnet")}
          >
            Back to ResNet
          </button>
        </div>
      )}
    </section>
  );
}
