import { useEffect, useRef, useState } from "react";
import { predictImage, type PredictResponse } from "../api";
import { EmptyState, StatusChip } from "./AppShell";

const VOTE_WINDOW = 7;

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
};

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
}: CameraPracticeProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const votes = useRef<string[]>([]);
  const challengeStartedAt = useRef<number>(Date.now());
  const handledCorrect = useRef(false);
  const handledIncorrect = useRef(false);
  const lastErrorRef = useRef<string | null>(null);

  const [status, setStatus] = useState("Show your hand");
  const [statusTone, setStatusTone] = useState<"neutral" | "success" | "warning" | "accent">("accent");
  const [error, setError] = useState<string | null>(backendError);
  const [debug, setDebug] = useState<PredictResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => setError(backendError), [backendError]);

  useEffect(() => {
    handledCorrect.current = false;
    handledIncorrect.current = false;
    challengeStartedAt.current = Date.now();
    votes.current = [];
    updateStatus("Show your hand", "accent");
    if (active && ready && streamRef.current && !runningRef.current) {
      runningRef.current = true;
      setRunning(true);
      void runPredictionLoop();
    }
  }, [targetLetter, attemptKey]);

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
  }, [active, ready]);

  function updateStatus(next: string, tone: typeof statusTone = "neutral") {
    setStatus(next);
    setStatusTone(tone);
    onStatusChange?.(next);
  }

  async function startCamera(isCancelled: () => boolean) {
    if (!ready) {
      setError(backendError ?? "Practice is unavailable right now.");
      return;
    }
    setLoading(true);
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
    } catch {
      // A cancelled attempt's play()/getUserMedia can reject (e.g.
      // AbortError when cleanup already tore down the video/stream) purely
      // because it was interrupted, not because the camera is actually
      // unavailable -- don't show a false error for that.
      if (isCancelled()) return;
      setError("Your camera isn't available right now. Check permissions and try again.");
      updateStatus("Camera permission needed", "warning");
    } finally {
      if (!isCancelled()) setLoading(false);
    }
  }

  function stopCamera() {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    votes.current = [];
    setRunning(false);
  }

  function drawBox(box: [number, number, number, number] | null | undefined, width: number, height: number) {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.width = width;
    overlay.height = height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!box) return;
    const [x1, y1, x2, y2] = box;
    ctx.strokeStyle = "#d7f36a";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  function setTransientError(message: string) {
    if (lastErrorRef.current === message) return;
    lastErrorRef.current = message;
    setError(message);
  }

  async function runPredictionLoop() {
    while (runningRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        await new Promise((resolve) => window.setTimeout(resolve, 16));
        continue;
      }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) break;
      ctx.drawImage(video, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((value) => resolve(value), "image/jpeg", 0.72)
      );
      if (!blob || !runningRef.current) continue;

      try {
        const prediction = await predictImage(blob, "practice.jpg", { requireHand: true });
        if (!runningRef.current || handledCorrect.current) continue;
        setDebug(prediction);
        drawBox(prediction.hand_box ?? null, canvas.width, canvas.height);
        setError(null);
        lastErrorRef.current = null;

        if (!prediction.prediction) {
          votes.current = [];
          updateStatus(prediction.message ?? "Show your hand in frame", "accent");
          continue;
        }

        if (prediction.low_confidence) {
          votes.current = [];
          updateStatus("Hold your sign steady", "warning");
          continue;
        }

        votes.current = [...votes.current, prediction.prediction].slice(-VOTE_WINDOW);
        const stable = majority(votes.current);
        if (!stable) {
          updateStatus("Keep signing...", "accent");
          continue;
        }

        if (stable === targetLetter) {
          handledCorrect.current = true;
          runningRef.current = false;
          updateStatus("Perfect sign!", "success");
          onCorrect(Date.now() - challengeStartedAt.current, prediction.confidence);
          continue;
        }

        // Friendly, specific status ("Almost — you're showing S") instead of
        // raw "Predicted: S" -- the fuller "you're showing X, we want Y"
        // callout lives in PracticePage (built from the same stable letter
        // via onIncorrect), this chip just stays short.
        updateStatus(incorrectHint ?? `Almost — you're showing ${stable}`, "warning");
        if (!handledIncorrect.current) {
          handledIncorrect.current = true;
          onIncorrect(stable);
        }
        votes.current = [];
      } catch (err) {
        votes.current = [];
        setTransientError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      }
    }
  }

  if (error && !running && !loading) {
    return (
      <EmptyState
        icon="📷"
        title="Camera unavailable"
        description={error}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative overflow-hidden rounded-3xl border border-[var(--color-line)] bg-black shadow-[inset_0_0_40px_rgba(0,0,0,0.5)]">
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
              <p className="text-lg font-semibold text-[var(--color-mist)]">
                {loading ? "Starting camera..." : active ? "Camera paused" : "Practice paused"}
              </p>
              {!loading && active && (
                <p className="mt-2 text-sm text-[var(--color-muted)]">Allow camera access to begin signing.</p>
              )}
            </div>
          </div>
        )}

        <div className="absolute left-3 top-3">
          <StatusChip tone={running ? "success" : "neutral"}>{running ? "Live" : "Idle"}</StatusChip>
        </div>

        <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
      </div>

      <div className="flex items-center justify-between gap-3 rounded-2xl border border-[var(--color-line-soft)] bg-[var(--color-panel-soft)] px-4 py-3">
        <p className="text-sm text-[var(--color-mist)]">Status</p>
        <StatusChip tone={statusTone}>{status}</StatusChip>
      </div>

      {error && running && (
        <p className="rounded-2xl border border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10 px-4 py-2 text-sm text-[var(--color-warm)]">
          {error}
        </p>
      )}

      <details className="rounded-2xl border border-[var(--color-line-soft)] bg-[var(--color-panel-soft)] px-4 py-3 text-xs text-[var(--color-muted)]">
        <summary className="cursor-pointer font-semibold text-[var(--color-mist)]">Technical details</summary>
        <div className="mt-2 space-y-1">
          <p>Detected: {debug?.prediction ?? "—"}</p>
          <p>Confidence: {debug ? `${(debug.confidence * 100).toFixed(1)}%` : "—"}</p>
          <p>Hand detected: {debug?.hand_detected == null ? "—" : debug.hand_detected ? "Yes" : "No"}</p>
        </div>
      </details>
    </div>
  );
}

function majority(labels: string[]): string | null {
  if (!labels.length) return null;
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  const [label, count] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  return count >= Math.max(2, Math.floor(labels.length / 2) + 1) ? label : null;
}
