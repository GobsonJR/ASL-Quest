import { useEffect, useRef, useState } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";
import { predictLandmarks } from "../api";
import { landmarkPayload, RollingStat, type Point3 } from "../debug/landmarkStats";
import { describeCameraError } from "../shared/cameraError";
import { createVideoHandLandmarker } from "../shared/handLandmarker";
import {
  StableLetterTracker,
  statusText,
  troubleshootingTip,
  type FrameState,
  type HandBox,
  type LivePhase,
} from "../shared/recognition";
import { EmptyState, StatusChip } from "./AppShell";
import { Glyph } from "./Icon";

/** Minimum time between landmark requests (about 10/s), and the tracker's notion of one "frame". */
export const LANDMARK_REQUEST_INTERVAL_MS = 100;

type LandmarkPracticeProps = {
  targetLetter: string;
  active: boolean;
  ready: boolean;
  backendError: string | null;
  onCorrect: (elapsedMs: number, confidence: number) => void;
  onIncorrect: (predictedLetter: string) => void;
  onStatusChange?: (status: string) => void;
  incorrectHint?: string;
  /** Bump to reset the current attempt without restarting the camera, exactly like CameraPractice. */
  attemptKey?: number;
  /** Minimum time between recognition requests. Exposed mainly so tests can run quickly. */
  requestIntervalMs?: number;
};

type Reading = {
  phase: LivePhase;
  framesInPhase: number;
  handDetected: boolean | null;
  label: string | null;
  confidence: number | null;
  lowConfidence: boolean;
};

const INITIAL_READING: Reading = { phase: "starting", framesInPhase: 0, handDetected: null, label: null, confidence: null, lowConfidence: false };

/** Experimental recognizer: MediaPipe runs in the browser (one persistent VIDEO-mode landmarker), and only the 21 landmarks go
 * to /predict_landmarks. Same props and callbacks as CameraPractice; XP is only ever awarded by the parent's onCorrect, which
 * this component calls at most once per attempt, after the StableLetterTracker has confirmed a confident, steady letter. */
export function LandmarkPractice({
  targetLetter,
  active,
  ready,
  backendError,
  onCorrect,
  onIncorrect,
  onStatusChange,
  incorrectHint,
  attemptKey = 0,
  requestIntervalMs = LANDMARK_REQUEST_INTERVAL_MS,
}: LandmarkPracticeProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const tracker = useRef(new StableLetterTracker());
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  // Bumped on every attempt/target/camera change: an answer requested under an older epoch is discarded.
  const epochRef = useRef(0);
  const challengeStartedAt = useRef(Date.now());
  const handledCorrect = useRef(false);
  const handledIncorrect = useRef(false);
  const latest = useRef({ targetLetter, onCorrect, onIncorrect, onStatusChange, incorrectHint, requestIntervalMs });
  latest.current = { targetLetter, onCorrect, onIncorrect, onStatusChange, incorrectHint, requestIntervalMs };

  const [landmarkerState, setLandmarkerState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(backendError);
  const [status, setStatus] = useState("Show your hand");
  const [statusTone, setStatusTone] = useState<"neutral" | "success" | "warning" | "accent">("accent");
  const [reading, setReading] = useState<Reading>(INITIAL_READING);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [restartKey, setRestartKey] = useState(0);
  const [timing, setTiming] = useState<{ detect: number | null; api: number | null }>({ detect: null, api: null });

  useEffect(() => setError(backendError), [backendError]);

  function updateStatus(next: string, tone: typeof statusTone = "neutral") {
    setStatus(next);
    setStatusTone(tone);
    latest.current.onStatusChange?.(next);
  }

  // New attempt (target letter or "practice again"): fresh votes and flags; an in-flight answer becomes stale.
  useEffect(() => {
    // After a confirmed sign the detection loop stops itself. If the parent keeps `active` true (e.g. the next letter of a
    // word), a new attempt must start it again, so bump the restart key.
    if (handledCorrect.current) setRestartKey((key) => key + 1);
    handledCorrect.current = false;
    handledIncorrect.current = false;
    challengeStartedAt.current = Date.now();
    epochRef.current += 1;
    tracker.current.reset();
    updateStatus("Show your hand", "accent");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetLetter, attemptKey]);

  // The landmarker lives as long as this component (not just one attempt), and is always closed on unmount.
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setLandmarkerState("loading");
    createVideoHandLandmarker()
      .then((landmarker) => {
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setLandmarkerState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setLandmarkerState("error");
        setError(`Could not load the hand tracker (${err instanceof Error ? err.message : String(err)}). Switch back to ResNet to keep practicing.`);
      });
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, [ready]);

  // Camera + detection loop, only while the attempt is active and the tracker is loaded.
  useEffect(() => {
    if (!active || !ready || landmarkerState !== "ready") return;
    const landmarker = landmarkerRef.current;
    const video = videoRef.current;
    if (!landmarker || !video) return;

    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | null = null;
    const abort = new AbortController();
    const detectStat = new RollingStat(60);
    const apiStat = new RollingStat(60);

    let lastVideoTime = -1;
    let lastStamp = 0;
    let latestHand: { points: Point3[]; handedness: string | null; box: HandBox; capturedAt: number } | null = null;
    let inFlight = false;
    let lastSent = 0;
    let lastSentCapture = -1;
    let lastNoHandTick = 0;
    let lastUi = 0;

    const draw = (points: Point3[] | null) => {
      const overlay = overlayRef.current;
      if (!overlay) return;
      if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
      if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
      const ctx = overlay.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!points) return;
      const w = overlay.width;
      const h = overlay.height;
      ctx.lineWidth = Math.max(2, w / 320);
      ctx.strokeStyle = "#58cc02";
      for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(points[start].x * w, points[start].y * h);
        ctx.lineTo(points[end].x * w, points[end].y * h);
        ctx.stroke();
      }
      ctx.fillStyle = "#ff4d4d";
      for (const p of points) {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, Math.max(3, w / 190), 0, Math.PI * 2);
        ctx.fill();
      }
    };

    // Feed one frame-state into the tracker and update the UI/XP callbacks (the same decisions CameraPractice makes).
    const applyFrame = (frameState: FrameState, label: string | null, confidence: number | null, box: HandBox | null) => {
      const result = tracker.current.update(frameState, label, box);
      setReading({
        phase: result.phase,
        framesInPhase: result.framesInPhase,
        handDetected: frameState === "no_hand" ? false : true,
        label,
        confidence,
        lowConfidence: frameState === "low_confidence",
      });
      if (result.phase !== "stable") {
        updateStatus(statusText(result.phase), result.phase === "low_confidence" || result.phase === "no_hand" ? "warning" : "accent");
      } else if (result.stable === latest.current.targetLetter) {
        if (handledCorrect.current) return;
        handledCorrect.current = true; // never award twice for the same attempt, however many frames still arrive
        cancelled = true; // stop detecting; the parent now switches `active` off
        updateStatus("Perfect sign!", "success");
        latest.current.onCorrect(Date.now() - challengeStartedAt.current, confidence ?? 0);
      } else if (result.stable) {
        updateStatus(latest.current.incorrectHint ?? `Almost — you're showing ${result.stable}`, "warning");
        if (!handledIncorrect.current) {
          handledIncorrect.current = true;
          latest.current.onIncorrect(result.stable);
        }
      }
    };

    const sendLatest = () => {
      // One request at a time, always the newest landmarks; stale frames are dropped, never queued.
      if (inFlight || !latestHand || latestHand.capturedAt === lastSentCapture) return;
      const snapshot = latestHand;
      const epoch = epochRef.current;
      inFlight = true;
      lastSent = performance.now();
      lastSentCapture = snapshot.capturedAt;
      const t0 = performance.now();
      predictLandmarks(landmarkPayload(snapshot.points, snapshot.handedness), abort.signal)
        .then((prediction) => {
          apiStat.add(performance.now() - t0);
          // Discard answers from an older attempt, after cleanup, or for a hand that has since left the frame.
          if (cancelled || epoch !== epochRef.current || !latestHand) return;
          setError(null);
          if (!prediction.prediction) {
            applyFrame("no_hand", null, null, null);
            return;
          }
          applyFrame(prediction.low_confidence ? "low_confidence" : "recognized", prediction.prediction, prediction.confidence, snapshot.box);
        })
        .catch((err) => {
          if (abort.signal.aborted || cancelled) return;
          tracker.current.reset();
          setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
          setReading((prev) => ({ ...prev, phase: "error", framesInPhase: prev.phase === "error" ? prev.framesInPhase + 1 : 1 }));
          updateStatus(statusText("error"), "warning");
        })
        .finally(() => {
          inFlight = false;
        });
    };

    const loop = () => {
      if (cancelled) return;
      raf = requestAnimationFrame(loop);
      if (video.readyState < 2 || !video.videoWidth || video.currentTime === lastVideoTime) return; // only genuinely new frames
      lastVideoTime = video.currentTime;
      const now = performance.now();
      const stamp = Math.max(now, lastStamp + 1); // MediaPipe needs strictly increasing timestamps
      lastStamp = stamp;
      let result;
      try {
        const t0 = performance.now();
        result = landmarker.detectForVideo(video, stamp);
        detectStat.add(performance.now() - t0);
      } catch (err) {
        setError(`Hand tracking failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }

      const hand = result.landmarks[0];
      if (hand) {
        const points = hand.map((p) => ({ x: p.x, y: p.y, z: p.z }));
        const xs = points.map((p) => p.x * video.videoWidth);
        const ys = points.map((p) => p.y * video.videoHeight);
        latestHand = {
          points,
          handedness: result.handedness[0]?.[0]?.categoryName ?? null,
          box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
          capturedAt: now,
        };
        draw(points);
        if (now - lastSent >= latest.current.requestIntervalMs) sendLatest();
      } else {
        latestHand = null;
        draw(null);
        // Count a missed hand once per request interval, so the tracker's "frames" mean the same thing as when a hand is present.
        if (now - lastNoHandTick >= latest.current.requestIntervalMs) {
          lastNoHandTick = now;
          applyFrame("no_hand", null, null, null);
        }
      }

      if (now - lastUi > 500) {
        lastUi = now;
        setTiming({ detect: detectStat.median, api: apiStat.median });
      }
    };

    const start = async () => {
      setLoading(true);
      setReading(INITIAL_READING);
      updateStatus(statusText("starting"), "neutral");
      try {
        const media = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 960, height: 720 },
          audio: false,
        });
        if (cancelled) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        stream = media;
        video.srcObject = media;
        await video.play();
        if (cancelled) return;
        setRunning(true);
        setError(null);
        updateStatus("Show your hand", "accent");
        raf = requestAnimationFrame(loop);
      } catch (err) {
        if (cancelled) return; // an interrupted start (unmount, StrictMode) is not a camera problem
        setError(describeCameraError(err));
        updateStatus("Camera permission needed", "warning");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      abort.abort(); // drop any pending request
      epochRef.current += 1;
      stream?.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
      tracker.current.reset();
      draw(null);
      setRunning(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ready, landmarkerState, restartKey]);

  if (error && !running && !loading) {
    return <EmptyState icon="📷" title="Hand tracking unavailable" description={error} />;
  }

  const tip = troubleshootingTip(reading.phase, reading.framesInPhase);
  const preparing = landmarkerState === "loading";
  const handChip =
    reading.handDetected === null
      ? { tone: "neutral" as const, text: "Hand: —" }
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
        <video ref={videoRef} className="aspect-[4/3] w-full object-cover" playsInline muted aria-label="Webcam preview for sign practice" />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />

        {!running && (
          <div className="absolute inset-0 grid place-items-center bg-black/60 px-6 text-center backdrop-blur-[2px]">
            <div>
              <p className="text-lg font-semibold text-white">
                {preparing ? "Loading hand tracker..." : loading ? "Starting camera..." : active ? "Camera paused" : "Practice paused"}
              </p>
              {!loading && !preparing && active && <p className="mt-2 text-sm text-white/70">Allow camera access to begin signing.</p>}
            </div>
          </div>
        )}

        <div className="absolute left-3 top-3">
          <StatusChip tone={running ? "success" : "neutral"}>{running ? "● Live" : "Idle"}</StatusChip>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2" aria-label="Recognition status" data-testid="recognition-status">
        <StatusChip tone={handChip.tone}>{handChip.text}</StatusChip>
        <StatusChip tone={letterChip.tone}>{letterChip.text}</StatusChip>
      </div>

      <div className="flex items-center justify-between gap-3 rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
        <p className="text-sm font-medium text-[var(--color-ink-soft)]">Status</p>
        <StatusChip tone={statusTone}>{status}</StatusChip>
      </div>

      {tip && (
        <p role="note" className="flex gap-2 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-ink-soft)]">
          <Glyph value="💡" />
          <span>{tip}</span>
        </p>
      )}

      {error && running && (
        <p role="alert" className="rounded-2xl border border-[var(--color-streak)]/25 bg-[var(--color-streak-soft)] px-4 py-2 text-sm text-[var(--color-streak)]">
          {error}
        </p>
      )}

      <details className="rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3 text-xs text-[var(--color-muted)]">
        <summary className="cursor-pointer font-semibold text-[var(--color-ink-soft)]">Technical details</summary>
        <div className="mt-2 space-y-1">
          <p>Recognizer: hand landmarks (experimental), tracked in your browser</p>
          <p>Detected: {reading.label ?? "—"}</p>
          <p>Confidence: {reading.confidence == null ? "—" : `${(reading.confidence * 100).toFixed(1)}%`}</p>
          <p>Hand detection (median): {timing.detect == null ? "—" : `${timing.detect.toFixed(1)} ms`}</p>
          <p>Prediction request (median): {timing.api == null ? "—" : `${timing.api.toFixed(1)} ms`}</p>
        </div>
      </details>
    </div>
  );
}
