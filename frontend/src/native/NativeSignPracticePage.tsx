import { useEffect, useRef, useState, type RefObject } from "react";
import {
  Card,
  EmptyState,
  PageHeader,
  PageLayout,
  PrimaryButton,
  ProgressBar,
  SecondaryButton,
  SectionHeader,
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

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Converts a watch-page URL into its embeddable form for the given approved source type.
 * Returns null when the URL should instead be shown as a link-out card -- either because
 * the source isn't a recognized embeddable provider (e.g. "external_url", a plain
 * reference page), or because the URL itself doesn't contain a parseable video id. Never
 * falls back to a bare <video src> for an arbitrary URL: an "external_url" reference here
 * is a web PAGE (e.g. a dictionary entry), not a media file, and would just fail to play. */
function referenceEmbedUrl(url: string, sourceType: string | null): string | null {
  if (sourceType === "youtube") {
    const match = url.match(/(?:v=|youtu\.be\/|embed\/)([\w-]{11})/);
    return match ? `https://www.youtube.com/embed/${match[1]}` : null;
  }
  if (sourceType === "vimeo") {
    const match = url.match(/vimeo\.com\/(?:video\/)?(\d+)(?:\/([a-f0-9]+))?/);
    if (!match) return null;
    return match[2] ? `https://player.vimeo.com/video/${match[1]}?h=${match[2]}` : `https://player.vimeo.com/video/${match[1]}`;
  }
  return null;
}

// A short, learner-facing reminder shown above the reference video/link -- same spirit as
// LearnPage.tsx's SIGN_HINTS for A-Z, but for native vocabulary. Presentational only, not
// backed by a "tip" column (native_signs has no such field, and adding one isn't needed
// for a handful of static strings scoped to this page).
const REFERENCE_TIPS: Record<string, string> = {
  HELLO: "Watch the hand's starting position closely, then match the same brief wave.",
  THANKYOU: "Notice where the fingertips start (at the chin) and where the hand ends up.",
  PLEASE: "Watch the circular motion on the chest -- speed and size matter less than the shape.",
  YES: "The whole sign is a small, repeated motion from the wrist, not the whole arm.",
  NO: "Watch how the fingers snap together -- it's quick, not a slow pinch.",
  MOTHER: "Watch exactly where the thumb taps -- that single contact point is the sign.",
  WATER: "Watch the handshape stay fixed while it taps near the mouth.",
  EAT1: "Watch the fingertips tap the mouth -- once for the verb EAT, not a longer motion.",
  HELP: "Watch which hand moves and which hand stays still underneath it.",
  BOOK: "Watch the two flat hands open like covers -- that opening motion is the whole sign.",
};

export function NativeSignPracticePage() {
  const { practiceNativeSignId, navigate, startNativeSignPractice, applyNativeSignResult } = useGame();
  const { push } = useToast();

  const [sign, setSign] = useState<NativeSign | null>(null);
  const [signs, setSigns] = useState<NativeSign[]>([]);
  const [progress, setProgress] = useState<NativeSignProgressEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("capture");
  const [result, setResult] = useState<NativePracticeResult | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  const referenceRef = useRef<HTMLDivElement>(null);
  const [referencePulse, setReferencePulse] = useState(false);
  // Gates the capture step behind an explicit acknowledgment that the learner watched the
  // reference first (LEARN -> WATCH -> COPY, not straight to recording). NativeVideoCapture
  // itself is untouched and always mounted -- this only toggles its existing `disabled`
  // prop, so the record/predict/result mechanics stay exactly as they were.
  const [watched, setWatched] = useState(false);

  useEffect(() => {
    if (practiceNativeSignId == null) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setPhase("capture");
    setResult(null);
    setPredictError(null);
    setWatched(false);
    Promise.all([getNativeSigns(), getNativeSignProgress(practiceNativeSignId)])
      .then(([signsResponse, progressResponse]) => {
        if (cancelled) return;
        setSigns(signsResponse.items);
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
      applyNativeSignResult(Boolean(outcome.correct), outcome.xp_earned, outcome.new_achievements);
      if (outcome.xp_earned > 0) push(`+${outcome.xp_earned} XP`, "success");
      if (outcome.new_achievements.length > 0) {
        push(`Achievement unlocked: ${outcome.new_achievements.map((item) => item.name).join(", ")}`, "info");
      }
      const refreshed = await getNativeSignProgress(sign.id).catch(() => null);
      if (refreshed) setProgress(refreshed);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction failed. Please try again.";
      setPredictError(message);
      push(message, "error");
      setPhase("capture");
    }
  }

  function practiceAgain() {
    setResult(null);
    setPredictError(null);
    setPhase("capture");
  }

  function watchReferenceAgain() {
    practiceAgain();
    referenceRef.current?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
    setReferencePulse(true);
    window.setTimeout(() => setReferencePulse(false), 1500);
  }

  const activeSigns = signs.filter((item) => item.active);
  const nextSign =
    sign && activeSigns.length > 1
      ? activeSigns[(activeSigns.findIndex((item) => item.id === sign.id) + 1) % activeSigns.length]
      : null;

  function goToNextSign() {
    if (nextSign) startNativeSignPractice(nextSign.id);
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
          <PageHeader title={sign.display_name} description={sign.meaning ?? undefined} compact />
          {sign.description && (
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--color-mist)]">{sign.description}</p>
          )}
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

      <StepTracker phase={phase} watched={watched} />

      <div className="grid gap-6 md:grid-cols-2">
        <ReferencePanel
          key={sign.id}
          sign={sign}
          panelRef={referenceRef}
          pulse={referencePulse}
          watched={watched}
          onReady={() => setWatched(true)}
        />

        <Card className="flex min-w-0 flex-col space-y-4">
          <SectionHeader title="Your practice" description="Now perform the sign." />

          {phase === "capture" && <NativeVideoCapture disabled={!watched} onClipReady={handleClip} />}
          {phase === "capture" && !watched && (
            <p className="text-xs text-[var(--color-muted)]" role="status">
              Watch the reference and select &ldquo;I&rsquo;m ready&rdquo; to unlock recording.
            </p>
          )}

          {phase === "processing" && (
            <div
              className="grid min-h-[220px] place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel-soft)]/40"
              role="status"
              aria-live="polite"
            >
              <div className="text-center">
                <p className="animate-pulse-glow inline-block rounded-full border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 px-4 py-2 text-sm text-[var(--color-accent)]">
                  Analyzing your sign...
                </p>
              </div>
            </div>
          )}

          {phase === "result" && result && (
            <div className="grid min-h-[220px] place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel-soft)]/40 px-6 py-10 text-center">
              <p
                className={`text-sm font-semibold ${
                  result.correct ? "text-[var(--color-success)]" : "text-[var(--color-warm)]"
                }`}
              >
                Clip recorded — see your result below.
              </p>
            </div>
          )}

          {predictError && (
            <p
              className="rounded-2xl border border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10 px-4 py-2 text-sm text-[var(--color-warm)]"
              role="alert"
            >
              {predictError}
            </p>
          )}
        </Card>
      </div>

      {phase === "result" && result && (
        <Card className="animate-pop space-y-5">
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
            {result.correct ? (
              <p className="mt-3 text-lg font-semibold text-[var(--color-accent)]">+{result.xp_earned} XP</p>
            ) : (
              <p className="mt-3 text-sm text-[var(--color-muted)]">No correctness XP — try again</p>
            )}
            {result.new_achievements.length > 0 && (
              <p className="mt-1 text-sm font-medium text-[var(--color-warm)]">
                New badge{result.new_achievements.length > 1 ? "s" : ""}:{" "}
                {result.new_achievements.map((item) => item.name).join(", ")}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
              <p className="text-xs text-[var(--color-muted)]">Expected</p>
              <p className="font-semibold text-[#eef4f0]">{result.expected_sign.gloss}</p>
            </div>
            <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
              <p className="text-xs text-[var(--color-muted)]">Predicted</p>
              <p className="font-semibold text-[#eef4f0]">{result.prediction}</p>
            </div>
            <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
              <p className="text-xs text-[var(--color-muted)]">Response time</p>
              <p className="font-semibold text-[#eef4f0]">{result.response_time}ms</p>
            </div>
            <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
              <p className="text-xs text-[var(--color-muted)]">Mastery now</p>
              <p className="font-semibold text-[#eef4f0]">{progress ? `${Math.round(progress.mastery)}%` : "—"}</p>
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

          <div className="border-t border-[var(--color-line-soft)] pt-5 text-center">
            <p className="text-base font-semibold text-[#eef4f0]">{result.correct ? "Great job!" : "Try again"}</p>
            <div className="mt-3 flex flex-wrap justify-center gap-3">
              {result.correct ? (
                <>
                  <PrimaryButton onClick={practiceAgain}>Practice again</PrimaryButton>
                  <SecondaryButton onClick={goToNextSign} className={nextSign ? "" : "hidden"}>
                    Next sign
                  </SecondaryButton>
                </>
              ) : (
                <>
                  <SecondaryButton onClick={watchReferenceAgain}>Watch reference</SecondaryButton>
                  <PrimaryButton onClick={practiceAgain}>Practice again</PrimaryButton>
                </>
              )}
              <SecondaryButton onClick={() => navigate("native")}>Back to Native Signs</SecondaryButton>
            </div>
          </div>
        </Card>
      )}
    </PageLayout>
  );
}

function StepTracker({ phase, watched }: { phase: Phase; watched: boolean }) {
  const steps: Array<{ key: "watch" | "copy" | "check"; label: string }> = [
    { key: "watch", label: "Watch" },
    { key: "copy", label: "Copy" },
    { key: "check", label: "Check" },
  ];

  function statusFor(key: "watch" | "copy" | "check"): "done" | "active" | "upcoming" {
    if (key === "watch") return watched ? "done" : "active";
    if (key === "copy") return watched && phase === "capture" ? "active" : watched ? "done" : "upcoming";
    return phase === "processing" ? "active" : phase === "result" ? "done" : "upcoming";
  }

  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Lesson progress: watch, copy, check">
      {steps.map((step, index) => {
        const status = statusFor(step.key);
        return (
          <div key={step.key} className="flex items-center gap-2">
            <span
              className={`rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-[0.08em] transition ${
                status === "active"
                  ? "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/15 text-[var(--color-accent)]"
                  : status === "done"
                    ? "border-[var(--color-success)]/30 bg-[var(--color-success)]/10 text-[var(--color-success)]"
                    : "border-[var(--color-line)] text-[var(--color-muted)]"
              }`}
            >
              {status === "done" ? "✓ " : ""}
              {step.label}
            </span>
            {index < steps.length - 1 && (
              <span className="animate-motion-arrow text-[var(--color-muted)]" aria-hidden="true">
                →
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ReferencePanel({
  sign,
  panelRef,
  pulse,
  watched,
  onReady,
}: {
  sign: NativeSign;
  panelRef: RefObject<HTMLDivElement | null>;
  pulse: boolean;
  watched: boolean;
  onReady: () => void;
}) {
  const reference = sign.reference;
  const embedUrl =
    reference.available && reference.video_url ? referenceEmbedUrl(reference.video_url, reference.source_type) : null;
  const tip = REFERENCE_TIPS[sign.gloss];

  const [embedState, setEmbedState] = useState<"loading" | "loaded" | "error">("loading");

  // React's synthetic event system only wires up "load" for <iframe> -- there is no
  // working onError for this tag (confirmed against react-dom's own source: only
  // "image" registers both "error" and "load"; "iframe"/"object" register "load"
  // only). A truly broken embed (blocked/unreachable/private) therefore never tells
  // us it failed -- the only observable signal is that "loaded" never arrives, so a
  // timeout is the actual failure detector here, not an onError handler.
  useEffect(() => {
    if (!embedUrl) return;
    setEmbedState("loading");
    const timer = window.setTimeout(() => {
      setEmbedState((current) => (current === "loaded" ? current : "error"));
    }, 8000);
    return () => window.clearTimeout(timer);
  }, [embedUrl]);

  return (
    <Card
      className={`flex min-w-0 flex-col space-y-4 transition-shadow ${
        pulse ? "animate-pulse-glow ring-2 ring-[var(--color-accent)]/50" : ""
      }`}
      style={{ scrollMarginTop: "5rem" }}
    >
      <div ref={panelRef} />
      <SectionHeader title="Reference" description="Learn the sign before you copy it." />
      <p className="-mt-2 font-display text-2xl text-[#eef4f0]">{sign.display_name}</p>

      {reference.available && reference.video_url ? (
        <div className="space-y-2">
          <div className="relative aspect-[4/3] w-full overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-black">
            {embedUrl ? (
              <>
                {embedState !== "loaded" && (
                  <div
                    className="absolute inset-0 grid place-items-center bg-[var(--color-ink)]"
                    // Hide the purely decorative loading spinner from the a11y tree, but
                    // the error state's message + "Open source instead" link are real
                    // content a screen-reader user needs -- they must stay accessible.
                    aria-hidden={embedState === "loading"}
                  >
                    {embedState === "loading" ? (
                      <span className="animate-pulse-glow rounded-full border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 px-4 py-2 text-xs text-[var(--color-accent)]">
                        Loading reference...
                      </span>
                    ) : (
                      <div className="px-6 text-center">
                        <p className="text-sm text-[var(--color-mist)]">This reference couldn&rsquo;t load.</p>
                        <a
                          href={reference.video_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-2 inline-block text-xs font-medium text-[var(--color-accent)] hover:underline"
                        >
                          Open source instead ↗
                        </a>
                      </div>
                    )}
                  </div>
                )}
                <iframe
                  className="h-full w-full"
                  src={embedUrl}
                  title={`Reference demonstration of ${sign.display_name}`}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  onLoad={() => setEmbedState("loaded")}
                />
              </>
            ) : (
              <a
                href={reference.video_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex h-full w-full flex-col items-center justify-center gap-3 bg-gradient-to-br from-[var(--color-panel-soft)] to-[var(--color-panel)] px-6 text-center transition hover:from-[var(--color-panel)] hover:to-[var(--color-ink)]"
              >
                <span className="text-5xl" aria-hidden="true">
                  ▶
                </span>
                <p className="text-sm font-semibold text-[#eef4f0]">Watch &ldquo;{sign.display_name}&rdquo;</p>
                <StatusChip tone="accent">Open source ↗</StatusChip>
              </a>
            )}
          </div>
          <p className="text-xs font-medium uppercase tracking-[0.1em] text-[var(--color-muted)]">Watch carefully</p>
          {tip && <p className="text-sm leading-relaxed text-[var(--color-mist)]">{tip}</p>}
          {reference.license_note && <p className="text-xs text-[var(--color-muted)]">{reference.license_note}</p>}
          {/* For the link-out case the card above IS the open-source link -- a second
              copy of the same link right below it would just be noise. Only the
              embedded-iframe case needs a distinct secondary "view the original" link. */}
          {embedUrl && (
            <a
              href={reference.video_url ?? undefined}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block text-xs font-medium text-[var(--color-accent)] hover:underline"
            >
              Open source ↗
            </a>
          )}
        </div>
      ) : (
        <div className="aspect-[4/3] w-full overflow-hidden rounded-[var(--radius-panel)] border border-dashed border-[var(--color-line)] bg-gradient-to-br from-[var(--color-panel-soft)] to-[var(--color-panel)]">
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <span className="animate-hand-wave text-5xl" aria-hidden="true">
              🤟
            </span>
            <StatusChip tone="neutral">Reference video coming soon</StatusChip>
            <p className="max-w-xs text-sm leading-relaxed text-[var(--color-mist)]">
              No verified reference video is connected for &ldquo;{sign.display_name}&rdquo; yet. Once an approved
              source is added to the catalog, it will play here automatically.
            </p>
            <div className="flex items-center gap-1 text-[var(--color-accent)]" aria-hidden="true">
              <span className="animate-motion-arrow">›</span>
              <span className="animate-motion-arrow" style={{ animationDelay: "0.25s" }}>
                ›
              </span>
              <span className="animate-motion-arrow" style={{ animationDelay: "0.5s" }}>
                ›
              </span>
            </div>
            <p className="text-xs text-[var(--color-muted)]">
              For now, use the gloss, meaning, and description above to learn the sign, then copy it on the right.
            </p>
          </div>
        </div>
      )}

      <div className="mt-auto pt-1">
        {watched ? (
          <StatusChip tone="success">✓ Watched</StatusChip>
        ) : (
          <PrimaryButton className="w-full" onClick={onReady}>
            I&rsquo;m ready — Continue
          </PrimaryButton>
        )}
      </div>
    </Card>
  );
}
