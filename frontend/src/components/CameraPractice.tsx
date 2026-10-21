import { useEffect, useRef, useState } from "react";
import { predictImage, type PredictResponse, type RecognizerChoice } from "../api";
import { describeCameraError } from "../shared/cameraError";
import {
  MIN_FRAME_INTERVAL_MS,
  StableLetterTracker,
  classifyFrame,
  statusText,
  troubleshootingTip,
  type LivePhase,
} from "../shared/recognition";
import { EmptyState, StatusChip } from "./AppShell";
import { Glyph } from "./Icon";

// Development-only: request server diagnostics and log state changes to the console.
const DIAGNOSTICS = import.meta.env.DEV;

type CameraPracticeProps = {
  targetLetter: string;
  active: boolean;
  ready: boolean;
  backendError: string | null;
  onCorrect: (elapsedMs: number, confidence: number) => void;
  onIncorrect: (predictedLetter: string) => void;
  onStatusChange?: (status: string) => void;
  incorrectHint?: string;
  /** Bump this (e.g. on "Practice again") to reset the current attempt --
   * votes, handled-correct/incorrect flags, status -- without tearing down
   * and re-requesting the camera stream, exactly like a targetLetter change
   * already does. */
  attemptKey?: number;
  /** Which engine answers: "resnet" (default, production) or the experimental "landmarks" model. Changing it restarts the
   * current attempt (votes are cleared) so answers from two engines are never mixed. */
  recognizer?: RecognizerChoice;
  /** Minimum time between recognition requests. Exposed mainly so tests can run quickly. */
  frameIntervalMs?: number;
};

type Reading = {
  phase: LivePhase;
  framesInPhase: number;
  handDetected: boolean | null;
  label: string | null;
  confidence: number | null;
  lowConfidence: boolean;
  roundTripMs: number | null;
  response: PredictResponse | null;
};

const INITIAL_READING: Reading = {
  phase: "starting",
  framesInPhase: 0,
  handDetected: null,
  label: null,
  confidence: null,
  lowConfidence: false,
  roundTripMs: null,
  response: null,
};

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const pct = (n: number, total: number) => (total ? `${Math.round((100 * n) / total)}%` : "—");

/** True if the on-screen preview is flipped left-right by CSS (this app does not flip it). */
function isPreviewMirrored(video: HTMLVideoElement): boolean {
  const transform = window.getComputedStyle(video).transform;
  const match = /matrix\(([-\d.]+),/.exec(transform);
  return match ? Number(match[1]) < 0 : false;
}

export function CameraPractice({
  targetLetter,
  active,
  ready,
  backendError,
  onCorrect,
  onIncorrect,
  onStatusChange,
  incorrectHint,
  attemptKey = 0,
  recognizer = "resnet",
  frameIntervalMs = MIN_FRAME_INTERVAL_MS,
}: CameraPracticeProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  // Only the loop whose id matches may keep running: stopping and immediately restarting the
  // camera can never leave two loops sending frames at once.
  const loopIdRef = useRef(0);
  // Bumped whenever the attempt/target/camera session changes. A response that was requested under
  // an older epoch is discarded, so a late answer can't overwrite a newer attempt.
  const epochRef = useRef(0);
  const tracker = useRef(new StableLetterTracker());
  const challengeStartedAt = useRef<number>(Date.now());
  const handledCorrect = useRef(false);
  const handledIncorrect = useRef(false);
  const lastErrorRef = useRef<string | null>(null);
  const lastPhaseLogged = useRef<LivePhase | null>(null);
  const lastFrameRef = useRef<Blob | null>(null);
  // Development counters: real-world rates of no-hand / uncertain / recognized frames since this page opened.
  const statsRef = useRef({ frames: 0, no_hand: 0, low_confidence: 0, recognized: 0, detector_unavailable: 0 });
  const [statsView, setStatsView] = useState(statsRef.current);
  // The loop is long-lived, so it reads the latest props/handlers through refs, not stale closures.
  const latest = useRef({ targetLetter, onCorrect, onIncorrect, onStatusChange, incorrectHint, frameIntervalMs, recognizer });
  latest.current = { targetLetter, onCorrect, onIncorrect, onStatusChange, incorrectHint, frameIntervalMs, recognizer };

  const [status, setStatus] = useState("Show your hand");
  const [statusTone, setStatusTone] = useState<"neutral" | "success" | "warning" | "accent">("accent");
  const [error, setError] = useState<string | null>(backendError);
  const [reading, setReading] = useState<Reading>(INITIAL_READING);
  const [videoSize, setVideoSize] = useState<string>("—");
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => setError(backendError), [backendError]);

  useEffect(() => {
    handledCorrect.current = false;
    handledIncorrect.current = false;
    challengeStartedAt.current = Date.now();
    epochRef.current += 1;
    tracker.current.reset();
    updateStatus("Show your hand", "accent");
    if (active && ready && streamRef.current && !runningRef.current) {
      runningRef.current = true;
      setRunning(true);
      void runPredictionLoop();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetLetter, attemptKey, recognizer]);

  useEffect(() => {
    // `cancelled` guards against a StrictMode dev-mode double-invoke (or any
    // other fast unmount/remount): the effect runs, cleans up, and runs again
    // before the first startCamera()'s getUserMedia/play() promises settle.
    // Without this, the first (now-stale, and by then interrupted/rejected)
    // invocation's catch block can call setError(...) *after* the second
    // invocation already succeeded, stomping good state with a false
    // "Camera unavailable" -- a real pre-existing race, not a StrictMode-only
    // curiosity, since npm run dev (StrictMode-enabled) is how this app is
    // actually developed and tested day to day.
    let cancelled = false;
    if (active && ready) {
      void startCamera(() => cancelled);
    } else {
      stopCamera();
    }
    return () => {
      cancelled = true;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ready]);

  function updateStatus(next: string, tone: typeof statusTone = "neutral") {
    setStatus(next);
    setStatusTone(tone);
    latest.current.onStatusChange?.(next);
  }

  async function startCamera(isCancelled: () => boolean) {
    if (!ready) {
      setError(backendError ?? "Practice is unavailable right now.");
      return;
    }
    setLoading(true);
    setReading(INITIAL_READING);
    updateStatus(statusText("starting"), "neutral");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 960, height: 720 },
        audio: false,
      });
      if (isCancelled()) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      if (isCancelled()) return;
      runningRef.current = true;
      setRunning(true);
      setError(null);
      lastErrorRef.current = null;
      updateStatus("Show your hand", "accent");
      void runPredictionLoop();
    } catch (err) {
      // A cancelled attempt's play()/getUserMedia can reject (e.g.
      // AbortError when cleanup already tore down the video/stream) purely
      // because it was interrupted, not because the camera is actually
      // unavailable -- don't show a false error for that.
      if (isCancelled()) return;
      setError(describeCameraError(err));
      updateStatus("Camera permission needed", "warning");
    } finally {
      if (!isCancelled()) setLoading(false);
    }
  }

  function stopCamera() {
    runningRef.current = false;
    loopIdRef.current += 1; // retire any loop that is still mid-request
    epochRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    tracker.current.reset();
    drawBox(null, 0, 0);
    setRunning(false);
  }

  function drawBox(
    box: [number, number, number, number] | null | undefined,
    width: number,
    height: number,
    uncertain = false
  ) {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.width = width;
    overlay.height = height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!box) return;
    const [x1, y1, x2, y2] = box;
    // Green = confident reading; amber = hand found but the letter is uncertain.
    ctx.strokeStyle = uncertain ? "#ff9600" : "#58cc02";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  /** Development only: download the last frame exactly as it was sent to the server, named with the
   * letter being practiced (frame_C_<time>.jpg), so it can be analysed offline. */
  function saveLastFrame() {
    const blob = lastFrameRef.current;
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `frame_${latest.current.targetLetter}_${Date.now()}.jpg`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function setTransientError(message: string) {
    if (lastErrorRef.current === message) return;
    lastErrorRef.current = message;
    setError(message);
  }

  async function runPredictionLoop() {
    const myLoop = ++loopIdRef.current;
    const alive = () => runningRef.current && loopIdRef.current === myLoop;

    while (alive()) {
      const frameStart = performance.now();
      const video = videoRef.current;
      const canvas = canvasRef.current;
      // Only process frames from a camera that is actually delivering video.
      if (!video || !canvas || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
        await sleep(16);
        continue;
      }
      if (DIAGNOSTICS) setVideoSize(`${video.videoWidth}×${video.videoHeight}`);
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) break;
      ctx.drawImage(video, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((value) => resolve(value), "image/jpeg", 0.72)
      );
      if (!blob || !alive()) continue;
      if (DIAGNOSTICS) lastFrameRef.current = blob; // the exact JPEG sent to the server, for "Save frame"

      const requestEpoch = epochRef.current;
      try {
        const sentAt = performance.now();
        const prediction = await predictImage(blob, "practice.jpg", {
          requireHand: true,
          // ResNet is the default and sends no extra parameter at all, so its requests are exactly what they were before.
          ...(latest.current.recognizer === "landmarks" ? { recognizer: "landmarks" as const } : {}),
          debug: DIAGNOSTICS,
          ...(DIAGNOSTICS
            ? {
                // Development only: the server saves this request's exact inputs when it runs with
                // ASL_QUEST_DEV_DIAGNOSTICS=1 (rate-limited and capped there).
                saveDiagnostics: true,
                target: latest.current.targetLetter,
                previewMirrored: isPreviewMirrored(video),
                videoSize: `${video.videoWidth}x${video.videoHeight}`,
              }
            : {}),
        });
        if (!alive() || handledCorrect.current) continue;
        // The attempt changed while this request was in flight: its answer belongs to the past.
        if (requestEpoch !== epochRef.current) continue;

        setError(null);
        lastErrorRef.current = null;

        const frameState = classifyFrame(prediction);
        // The hand box lets the tracker drop votes when the hand has moved to a clearly different place/size.
        const result = tracker.current.update(frameState, prediction.prediction, prediction.hand_box ?? null);
        if (DIAGNOSTICS) {
          const stats = statsRef.current;
          stats.frames += 1;
          stats[frameState] += 1;
          setStatsView({ ...stats });
        }
        const uncertain = frameState === "low_confidence";
        drawBox(prediction.hand_box ?? null, canvas.width, canvas.height, uncertain);
        setReading({
          phase: result.phase,
          framesInPhase: result.framesInPhase,
          handDetected: frameState === "no_hand" ? false : frameState === "detector_unavailable" ? null : true,
          label: prediction.prediction,
          confidence: prediction.prediction ? prediction.confidence : null,
          lowConfidence: uncertain,
          roundTripMs: Math.round(performance.now() - sentAt),
          response: DIAGNOSTICS ? prediction : null,
        });
        if (DIAGNOSTICS && lastPhaseLogged.current !== result.phase) {
          lastPhaseLogged.current = result.phase;
          console.debug("[asl-camera]", result.phase, {
            video: `${video.videoWidth}x${video.videoHeight}`,
            handDetected: prediction.hand_detected,
            box: prediction.hand_box,
            label: prediction.prediction,
            confidence: prediction.confidence,
            serverDiagnostics: prediction.diagnostics,
          });
        }

        if (result.phase !== "stable") {
          const tone = result.phase === "low_confidence" || result.phase === "no_hand" ? "warning" : "accent";
          updateStatus(prediction.message && frameState === "detector_unavailable" ? prediction.message : statusText(result.phase), tone);
        } else if (result.stable === latest.current.targetLetter) {
          handledCorrect.current = true;
          runningRef.current = false;
          updateStatus("Perfect sign!", "success");
          latest.current.onCorrect(Date.now() - challengeStartedAt.current, prediction.confidence);
          continue;
        } else if (result.stable) {
          // Friendly, specific status ("Almost -- you're showing S") instead of raw "Predicted: S" --
          // the fuller "you're showing X, we want Y" callout lives in PracticePage.
          updateStatus(latest.current.incorrectHint ?? `Almost — you're showing ${result.stable}`, "warning");
          if (!handledIncorrect.current) {
            handledIncorrect.current = true;
            latest.current.onIncorrect(result.stable);
          }
        }
      } catch (err) {
        if (!alive()) continue;
        tracker.current.reset();
        const message = err instanceof Error ? err.message : "Something went wrong. Try again.";
        setTransientError(message);
        setReading((prev) => ({ ...prev, phase: "error", framesInPhase: prev.phase === "error" ? prev.framesInPhase + 1 : 1 }));
        updateStatus(statusText("error"), "warning");
        await sleep(500); // back off instead of hammering a failing backend
      }

      // Throttle: send at most ~1/frameInterval frames per second.
      const spent = performance.now() - frameStart;
      if (spent < latest.current.frameIntervalMs) await sleep(latest.current.frameIntervalMs - spent);
    }
  }

  if (error && !running && !loading) {
    return <EmptyState icon="📷" title="Camera unavailable" description={error} />;
  }

  const tip = troubleshootingTip(reading.phase, reading.framesInPhase);
  const handChip =
    reading.handDetected === null
      ? { tone: "neutral" as const, text: reading.phase === "detector_unavailable" ? "Hand detector off" : "Hand: —" }
      : reading.handDetected
        ? { tone: "success" as const, text: "Hand detected" }
        : { tone: "warning" as const, text: "No hand" };
  const letterChip = !reading.label
    ? { tone: "neutral" as const, text: "Reading: —" }
    : reading.lowConfidence
      ? { tone: "warning" as const, text: `Unsure: ${reading.label} ${Math.round((reading.confidence ?? 0) * 100)}%` }
      : { tone: "accent" as const, text: `Reading: ${reading.label} ${Math.round((reading.confidence ?? 0) * 100)}%` };

  return (
    <div className="space-y-4">
      <div className="relative overflow-hidden rounded-[1.75rem] border-4 border-[var(--color-primary-soft)] bg-black shadow-[0_8px_28px_-8px_rgba(88,204,2,0.35)]">
        <video
          ref={videoRef}
          className="aspect-[4/3] w-full object-cover"
          playsInline
          muted
          aria-label="Webcam preview for sign practice"
        />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />

        {!running && (
          <div className="absolute inset-0 grid place-items-center bg-black/60 px-6 text-center backdrop-blur-[2px]">
            <div>
              <p className="text-lg font-semibold text-white">
                {loading ? "Starting camera..." : active ? "Camera paused" : "Practice paused"}
              </p>
              {!loading && active && (
                <p className="mt-2 text-sm text-white/70">Allow camera access to begin signing.</p>
              )}
            </div>
          </div>
        )}

        <div className="absolute left-3 top-3">
          <StatusChip tone={running ? "success" : "neutral"}>{running ? "● Live" : "Idle"}</StatusChip>
        </div>

        <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
      </div>

      {/* Hand detection and letter reading are reported separately: a weak letter guess is never shown as "no hand". */}
      <div className="flex flex-wrap items-center gap-2" aria-label="Recognition status" data-testid="recognition-status">
        <StatusChip tone={handChip.tone}>{handChip.text}</StatusChip>
        <StatusChip tone={letterChip.tone}>{letterChip.text}</StatusChip>
      </div>

      <div className="flex items-center justify-between gap-3 rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
        <p className="text-sm font-medium text-[var(--color-ink-soft)]">Status</p>
        <StatusChip tone={statusTone}>{status}</StatusChip>
      </div>

      {tip && (
        <p
          role="note"
          className="flex gap-2 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-ink-soft)]"
        >
          <Glyph value="💡" />
          <span>{tip}</span>
        </p>
      )}

      {error && running && (
        <p
          role="alert"
          className="rounded-2xl border border-[var(--color-streak)]/25 bg-[var(--color-streak-soft)] px-4 py-2 text-sm text-[var(--color-streak)]"
        >
          {error}
        </p>
      )}

      <details
        open={DIAGNOSTICS || undefined}
        className="rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3 text-xs text-[var(--color-muted)]"
      >
        <summary className="cursor-pointer font-semibold text-[var(--color-ink-soft)]">Technical details</summary>
        <div className="mt-2 space-y-1">
          <p>Detected: {reading.label ?? "—"}</p>
          <p>Confidence: {reading.confidence == null ? "—" : `${(reading.confidence * 100).toFixed(1)}%`}</p>
          <p>
            Hand detected:{" "}
            {reading.handDetected == null ? "—" : reading.handDetected ? "Yes" : "No"}
          </p>
          {DIAGNOSTICS && (
            <>
              <p>Phase: {reading.phase}</p>
              <p>
                Frames {statsView.frames}: no hand {pct(statsView.no_hand, statsView.frames)}, uncertain{" "}
                {pct(statsView.low_confidence, statsView.frames)}, recognized {pct(statsView.recognized, statsView.frames)}
              </p>
              <p>Video: {videoSize}</p>
              <p>Round trip: {reading.roundTripMs == null ? "—" : `${reading.roundTripMs} ms`}</p>
              <p>Inference: {reading.response?.diagnostics?.inference_ms ?? "—"} ms</p>
              <p>
                Input: {reading.response?.diagnostics?.tensor_shape?.join("×") ?? "—"}{" "}
                {reading.response?.diagnostics?.tensor_dtype ?? ""} on {reading.response?.diagnostics?.device ?? "—"}
              </p>
              <p>Class index: {reading.response?.diagnostics?.class_index ?? "—"}</p>
              <p>Hand box: {reading.response?.hand_box ? reading.response.hand_box.join(", ") : "—"}</p>
              <p>Frame (server): {reading.response?.frame_size?.join("×") ?? "—"}</p>
              <p>Detector min confidence: {reading.response?.diagnostics?.hand_detection_confidence ?? "—"}</p>
              <p>
                Capture:{" "}
                {reading.response?.diagnostics?.capture_enabled
                  ? (reading.response.diagnostics.saved_to ?? "on (rate-limited)")
                  : "off (start the backend with ASL_QUEST_DEV_DIAGNOSTICS=1)"}
              </p>
              <button
                type="button"
                onClick={saveLastFrame}
                className="mt-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 font-semibold text-[var(--color-ink-soft)]"
              >
                Save last frame (for scripts/diagnose_frame.py)
              </button>
            </>
          )}
        </div>
      </details>
    </div>
  );
}
