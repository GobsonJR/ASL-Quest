import { useEffect, useState } from "react";
import {
  Card,
  EmptyState,
  PageHeader,
  PageLayout,
  PrimaryButton,
  ProgressBar,
  SecondaryButton,
  StatusChip,
} from "../components/AppShell";
import { useToast } from "../components/Toast";
import { useGame } from "../game/GameContext";
import {
  getNativeSignProgress,
  getNativeSigns,
  practiceNativeSign,
  type NativePracticeResult,
  type NativeSign,
  type NativeSignProgressEntry,
} from "./api";
import { NativeVideoCapture } from "./NativeVideoCapture";

type Phase = "capture" | "processing" | "result";

export function NativeSignPracticePage() {
  const { practiceNativeSignId, navigate } = useGame();
  const { push } = useToast();

  const [sign, setSign] = useState<NativeSign | null>(null);
  const [progress, setProgress] = useState<NativeSignProgressEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("capture");
  const [result, setResult] = useState<NativePracticeResult | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    if (practiceNativeSignId == null) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    Promise.all([getNativeSigns(), getNativeSignProgress(practiceNativeSignId)])
      .then(([signsResponse, progressResponse]) => {
        if (cancelled) return;
        const match = signsResponse.items.find((item) => item.id === practiceNativeSignId) ?? null;
        setSign(match);
        setProgress(progressResponse);
      })
      .catch((err) => {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : "Unable to load this native sign.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [practiceNativeSignId]);

  async function handleClip(clip: Blob, mimeType: string) {
    if (!sign) return;
    setPhase("processing");
    setPredictError(null);
    try {
      const extension = mimeType.includes("mp4") ? "mp4" : "webm";
      const outcome = await practiceNativeSign(clip, sign.id, `practice.${extension}`, 5);
      setResult(outcome);
      setPhase("result");
      const refreshed = await getNativeSignProgress(sign.id).catch(() => null);
      if (refreshed) setProgress(refreshed);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction failed. Please try again.";
      setPredictError(message);
      push(message, "error");
      setPhase("capture");
    }
  }

  function tryAgain() {
    setResult(null);
    setPredictError(null);
    setPhase("capture");
  }

  if (practiceNativeSignId == null) {
    return (
      <PageLayout>
        <EmptyState
          icon="🤟"
          title="No sign selected"
          description="Choose a sign from the Native Signs catalog to start practicing."
          action={<PrimaryButton onClick={() => navigate("native")}>Back to Native Signs</PrimaryButton>}
        />
      </PageLayout>
    );
  }

  if (loading) {
    return (
      <PageLayout>
        <div className="grid min-h-[40vh] place-items-center text-[var(--color-mist)]">Loading sign...</div>
      </PageLayout>
    );
  }

  if (loadError || !sign) {
    return (
      <PageLayout>
        <EmptyState
          icon="⚠️"
          title="Unable to load this sign"
          description={loadError ?? "This native sign could not be found."}
          action={<PrimaryButton onClick={() => navigate("native")}>Back to Native Signs</PrimaryButton>}
        />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <SecondaryButton onClick={() => navigate("native")} ariaLabel="Back to Native Signs">
        ← Back to Native Signs
      </SecondaryButton>

      <div className="glass-panel relative overflow-hidden rounded-[var(--radius-panel)] p-6 md:p-8">
        <div
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(600px 260px at 15% -10%, rgba(215,243,106,0.16) 0%, transparent 55%), radial-gradient(500px 220px at 100% 10%, rgba(243,193,107,0.10) 0%, transparent 50%)",
          }}
          aria-hidden="true"
        />
        <div className="relative">
          <div className="flex flex-wrap items-center gap-2">
            <StatusChip tone="accent">NATIVE SIGN PRACTICE</StatusChip>
            {sign.category && <StatusChip tone="neutral">{sign.category}</StatusChip>}
            {sign.difficulty && <StatusChip tone="neutral">{sign.difficulty}</StatusChip>}
          </div>
          <PageHeader title={sign.display_name} description={sign.meaning ?? sign.description ?? undefined} compact />
          {sign.example_text && (
            <p className="mt-2 text-sm italic text-[var(--color-muted)]">&ldquo;{sign.example_text}&rdquo;</p>
          )}

          {progress && (
            <div className="mt-5 max-w-sm">
              <ProgressBar percent={progress.mastery} label="Mastery" size="sm" />
              <p className="mt-1 text-xs text-[var(--color-muted)]">
                {progress.correct} / {progress.attempts} correct
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <Card className="space-y-4">
          <p className="text-sm text-[var(--color-mist)]">
            Perform the sign naturally in front of the camera. Recording is brief and automatic.
          </p>

          {phase === "capture" && (
            <NativeVideoCapture disabled={false} onClipReady={handleClip} />
          )}

          {phase === "processing" && (
            <div className="grid min-h-[220px] place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel-soft)]/40">
              <div className="text-center">
                <p className="animate-pulse-glow inline-block rounded-full border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 px-4 py-2 text-sm text-[var(--color-accent)]">
                  Analyzing your sign...
                </p>
              </div>
            </div>
          )}

          {phase === "result" && (
            <div className="flex flex-wrap gap-3">
              <PrimaryButton onClick={tryAgain}>Try again</PrimaryButton>
              <SecondaryButton onClick={() => navigate("native")}>Back to Native Signs</SecondaryButton>
            </div>
          )}

          {predictError && (
            <p className="rounded-2xl border border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10 px-4 py-2 text-sm text-[var(--color-warm)]">
              {predictError}
            </p>
          )}
        </Card>

        <Card className="flex flex-col">
          {phase !== "result" || !result ? (
            <div className="flex flex-1 items-center justify-center text-center text-sm text-[var(--color-mist)]">
              Your result will appear here after recording.
            </div>
          ) : (
            <div className="animate-pop space-y-4">
              <div
                className={`rounded-[var(--radius-panel)] border p-5 text-center ${
                  result.correct
                    ? "border-[var(--color-success)]/30 bg-[var(--color-success)]/[0.06]"
                    : "border-[var(--color-warm)]/30 bg-[var(--color-warm)]/[0.06]"
                }`}
              >
                <p
                  className={`text-sm font-medium ${
                    result.correct ? "text-[var(--color-success)]" : "text-[var(--color-warm)]"
                  }`}
                >
                  {result.correct ? "Correct!" : "Not quite"}
                </p>
                <h3 className="mt-2 font-display text-3xl text-[#eef4f0]">{result.prediction}</h3>
                <p className="mt-1 text-xs uppercase tracking-[0.1em] text-[var(--color-muted)]">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
                  <p className="text-xs text-[var(--color-muted)]">Expected</p>
                  <p className="font-semibold text-[#eef4f0]">{result.expected_sign.gloss}</p>
                </div>
                <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
                  <p className="text-xs text-[var(--color-muted)]">Predicted</p>
                  <p className="font-semibold text-[#eef4f0]">{result.prediction}</p>
                </div>
              </div>

              {result.top_k.length > 1 && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.1em] text-[var(--color-muted)]">
                    Top predictions
                  </p>
                  <ul className="mt-2 space-y-1.5">
                    {result.top_k.map((item) => (
                      <li key={item.gloss} className="flex items-center gap-3">
                        <span className="w-20 shrink-0 text-sm text-[var(--color-mist)]">{item.gloss}</span>
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-ink)]">
                          <div
                            className="h-full rounded-full bg-[var(--color-accent)]"
                            style={{ width: `${item.confidence * 100}%` }}
                          />
                        </div>
                        <span className="w-10 shrink-0 text-right text-xs text-[var(--color-muted)]">
                          {Math.round(item.confidence * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <p className="text-xs text-[var(--color-muted)]">Response time: {result.response_time}ms</p>
            </div>
          )}
        </Card>
      </div>
    </PageLayout>
  );
}
