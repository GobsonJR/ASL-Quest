import { useEffect, useRef, useState } from "react";
import { Card, PrimaryButton, SecondaryButton, SectionLabel, StatusChip } from "../components/AppShell";
import { CollectApiError, saveNativeClip, type NativeSignOption, type SessionInfo } from "./api";

const MAX_SECONDS = 10;
const MIME_TYPES = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];

function supportedMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? null;
}

type Take = { blob: Blob; url: string; durationMs: number };

/** Records a short clip of a native sign from the already-open camera stream. Nothing is recorded until the admin presses
 * Record, nothing is saved until they watch it back and confirm the sign, and the sign label is the one they chose. */
export function NativeClipRecorder({
  signs,
  session,
  getStream,
  cameraOn,
  hand,
  targetPerSign,
  onGlossChange,
  onSaved,
}: {
  signs: NativeSignOption[];
  session: SessionInfo;
  getStream: () => MediaStream | null;
  cameraOn: boolean;
  hand: string;
  targetPerSign: number;
  /** Tells the page which sign is selected, so the review gallery can show it. */
  onGlossChange?: (gloss: string | null) => void;
  onSaved: (session: SessionInfo) => void;
}) {
  const [gloss, setGloss] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [take, setTake] = useState<Take | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);
  const takeUrlRef = useRef<string | null>(null);

  const clearTake = () => {
    if (takeUrlRef.current) URL.revokeObjectURL(takeUrlRef.current);
    takeUrlRef.current = null;
    setTake(null);
  };

  // leaving the page (or switching sign mode) must stop an in-progress recording and free the preview
  useEffect(
    () => () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
      if (takeUrlRef.current) URL.revokeObjectURL(takeUrlRef.current);
    },
    []
  );

  const mimeType = supportedMimeType();

  function startRecording() {
    const stream = getStream();
    if (!gloss || !stream || !mimeType) return;
    setError(null);
    setMessage(null);
    clearTake();
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => event.data.size > 0 && chunks.push(event.data);
    recorder.onstop = () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      const blob = new Blob(chunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      takeUrlRef.current = url;
      setTake({ blob, url, durationMs: Date.now() - startedAtRef.current });
      setRecording(false);
    };
    recorder.onerror = () => {
      setRecording(false);
      setError("The recording failed. Check the camera and try again.");
    };
    recorderRef.current = recorder;
    startedAtRef.current = Date.now();
    recorder.start();
    setRecording(true);
    setSeconds(0);
    timerRef.current = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - startedAtRef.current) / 1000);
      setSeconds(elapsed);
      if (elapsed >= MAX_SECONDS) stopRecording();
    }, 250);
  }

  function stopRecording() {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  async function save() {
    if (!take || !gloss || busy) return;
    setBusy(true);
    setError(null);
    try {
      const reply = await saveNativeClip({ blob: take.blob, sessionId: session.session_id, gloss, hand, durationMs: take.durationMs });
      onSaved(reply.session);
      setMessage(`Saved a clip as ${gloss} in this session.`);
      clearTake();
    } catch (err) {
      const failure = err as CollectApiError;
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  function discard() {
    clearTake();
    setError(null);
    setMessage("Discarded: nothing was saved.");
  }

  const counts = session.native_counts ?? {};

  return (
    <Card className="space-y-3" data-testid="native-recorder">
      <SectionLabel>1 · Choose the native sign you are about to perform</SectionLabel>
      {signs.length === 0 ? (
        <p className="text-sm text-[var(--color-ink-soft)]">There are no active native signs in the catalog yet.</p>
      ) : (
        <div className="flex flex-wrap gap-2" role="group" aria-label="Native sign to record">
          {signs.map((sign) => (
            <button
              key={sign.gloss}
              type="button"
              aria-pressed={gloss === sign.gloss}
              aria-label={`Sign ${sign.display_name}, ${counts[sign.gloss] ?? 0} clips this session`}
              disabled={recording}
              onClick={() => {
                setGloss(sign.gloss);
                onGlossChange?.(sign.gloss);
              }}
              className={`min-h-10 rounded-xl border-2 px-3 text-sm font-bold ${
                gloss === sign.gloss ? "border-[var(--color-native)] bg-[var(--color-native-soft)]" : "border-[var(--color-border)]"
              }`}
            >
              {sign.display_name}
              <span className="ml-2 text-xs font-semibold text-[var(--color-muted)]">{counts[sign.gloss] ?? 0}/{targetPerSign}</span>
            </button>
          ))}
        </div>
      )}

      {!mimeType && (
        <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
          This browser cannot record video. Use a current Chrome, Edge or Firefox.
        </p>
      )}

      {!take && (
        <div className="flex flex-wrap items-center gap-3">
          {!recording ? (
            <PrimaryButton disabled={!gloss || !cameraOn || !mimeType} onClick={startRecording}>
              {gloss ? `Record ${gloss} (up to ${MAX_SECONDS} s)` : "Choose a sign first"}
            </PrimaryButton>
          ) : (
            <>
              <StatusChip tone="warning">● Recording · {seconds}s</StatusChip>
              <PrimaryButton onClick={stopRecording}>Stop</PrimaryButton>
            </>
          )}
        </div>
      )}

      {take && (
        <div className="space-y-3" data-testid="clip-review">
          <SectionLabel>2 · Watch it back, then confirm</SectionLabel>
          <video src={take.url} controls playsInline className="aspect-[4/3] w-full max-w-md rounded-xl bg-black" aria-label="Recorded clip preview" />
          <p className="text-xs text-[var(--color-muted)]">{(take.blob.size / 1024).toFixed(0)} KB · {(take.durationMs / 1000).toFixed(1)} s</p>
          <div className="flex flex-wrap gap-3">
            <PrimaryButton disabled={busy} onClick={() => void save()}>{`Yes, this is ${gloss} — save clip`}</PrimaryButton>
            <SecondaryButton onClick={() => { clearTake(); setError(null); }}>Retake</SecondaryButton>
            <SecondaryButton onClick={discard}>Discard</SecondaryButton>
          </div>
        </div>
      )}

      {message && <p role="status" className="text-sm font-semibold text-[var(--color-primary-dim)]">{message}</p>}
      {error && (
        <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
          {error}
        </p>
      )}
      <p className="text-xs text-[var(--color-muted)]">
        The label is exactly the sign you confirm here; the recognizer is never consulted. Clips can show a face and a room, so only record people who agreed.
      </p>
    </Card>
  );
}
