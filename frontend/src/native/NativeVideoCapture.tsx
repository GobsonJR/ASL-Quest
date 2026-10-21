import { useEffect, useRef, useState } from "react";
import { PrimaryButton, StatusChip } from "../components/AppShell";
import { describeCameraError } from "../shared/cameraError";

// Native signs (e.g. BOOK) are multi-part motions; 3s was too tight to even get both hands
// in frame. Give a get-ready countdown with a live preview, then a longer window. The user
// can still press "Stop now" as soon as they finish.
const RECORD_SECONDS = 10;
const COUNTDOWN_SECONDS = 3;
const CANDIDATE_MIME_TYPES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4",
];

function pickSupportedMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const type of CANDIDATE_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return null;
}

type CaptureState = "idle" | "countdown" | "recording" | "clip_ready";

export function NativeVideoCapture({
  disabled = false,
  onClipReady,
}: {
  disabled?: boolean;
  onClipReady: (clip: Blob, mimeType: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<number | null>(null);

  const [state, setState] = useState<CaptureState>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(RECORD_SECONDS);
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);

  useEffect(() => {
    setUnsupported(pickSupportedMimeType() === null);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  async function startRecording() {
    setCameraError(null);
    const mimeType = pickSupportedMimeType();
    if (mimeType === null) {
      setUnsupported(true);
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    } catch (err) {
      setCameraError(describeCameraError(err));
      return;
    }

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    // Live preview + 3-2-1 so the learner can get into position before recording starts.
    setState("countdown");
    for (let remaining = COUNTDOWN_SECONDS; remaining > 0; remaining--) {
      setCountdown(remaining);
      await new Promise<void>((resolve) => {
        stopTimerRef.current = window.setTimeout(resolve, 1000);
      });
      if (streamRef.current !== stream) return; // cancelled via reset/unmount
    }

    chunksRef.current = [];
    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const clip = new Blob(chunksRef.current, { type: mimeType });
      stopCamera();
      setState("clip_ready");
      onClipReady(clip, mimeType);
    };
    // Without this, a mid-recording failure (device unplugged, track ended,
    // encoder error) leaves the UI stuck showing "Recording..." forever --
    // onstop never fires on its own after a recorder-level error.
    recorder.onerror = (event) => {
      if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
      stopCamera();
      setState("idle");
      const mediaError = (event as unknown as { error?: unknown }).error;
      setCameraError(describeCameraError(mediaError));
    };
    recorderRef.current = recorder;

    recorder.start();
    setState("recording");
    setSecondsLeft(RECORD_SECONDS);

    const tick = (remaining: number) => {
      if (remaining <= 0) {
        recorder.stop();
        return;
      }
      setSecondsLeft(remaining);
      stopTimerRef.current = window.setTimeout(() => tick(remaining - 1), 1000);
    };
    stopTimerRef.current = window.setTimeout(() => tick(RECORD_SECONDS - 1), 1000);
  }

  function reset() {
    if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
    stopCamera();
    setState("idle");
    setCameraError(null);
  }

  if (unsupported) {
    return (
      <div className="rounded-[var(--radius-panel)] border border-dashed border-[var(--color-border)] px-6 py-10 text-center">
        <p className="text-sm text-[var(--color-ink-soft)]">
          Video recording isn't supported in this browser. Try a recent version of Chrome, Edge, or Firefox.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative overflow-hidden rounded-[1.75rem] border-4 border-[var(--color-native-soft)] bg-black shadow-[0_8px_28px_-8px_rgba(28,176,246,0.35)]">
        <video
          ref={videoRef}
          className="aspect-[4/3] w-full object-cover"
          playsInline
          muted
          aria-label="Webcam preview for native sign practice"
        />
        {state !== "recording" && state !== "countdown" && (
          <div className="absolute inset-0 grid place-items-center bg-black/60 px-6 text-center backdrop-blur-[2px]">
            <p className="text-sm text-white/85">
              {state === "clip_ready" ? "Clip captured" : "Camera preview appears when recording starts"}
            </p>
          </div>
        )}
        <div className="absolute left-3 top-3">
          <StatusChip tone={state === "recording" ? "warning" : "neutral"}>
            {state === "recording" ? `Recording · ${secondsLeft}s` : state === "countdown" ? "Get ready" : "Idle"}
          </StatusChip>
        </div>
        {state === "countdown" && (
          <div className="absolute inset-0 grid place-items-center" role="status" aria-live="polite">
            <span className="font-display text-8xl text-white drop-shadow-lg">{countdown}</span>
          </div>
        )}
      </div>

      {cameraError && (
        <p className="rounded-2xl border border-[var(--color-streak)]/30 bg-[var(--color-streak)]/10 px-4 py-2 text-sm text-[var(--color-streak)]">
          {cameraError}
        </p>
      )}

      <div className="flex flex-wrap gap-3">
        {state === "idle" && (
          <PrimaryButton onClick={startRecording} disabled={disabled}>
            Start recording ({RECORD_SECONDS}s)
          </PrimaryButton>
        )}
        {state === "countdown" && <PrimaryButton onClick={reset}>Cancel</PrimaryButton>}
        {state === "recording" && (
          <PrimaryButton onClick={() => recorderRef.current?.stop()}>Stop now</PrimaryButton>
        )}
        {state === "clip_ready" && (
          <PrimaryButton onClick={reset} disabled={disabled}>
            Record again
          </PrimaryButton>
        )}
      </div>
    </div>
  );
}
