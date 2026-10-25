import { useCallback, useEffect, useRef, useState } from "react";
import { Card, EmptyState, PageHeader, PageLayout, PrimaryButton, ProgressBar, SecondaryButton, SectionLabel, StatusChip } from "../components/AppShell";
import { describeCameraError } from "../shared/cameraError";
import { ExportCard } from "./ExportCard";
import { NativeClipRecorder } from "./NativeClipRecorder";
import { SampleGallery } from "./SampleGallery";
import { referenceImageAlt, referenceImageSrc } from "../pages/alphabetFeedback";
import {
  CollectApiError,
  analyzeFrame,
  collectStatus,
  createSession,
  deleteSample,
  listSessions,
  saveNegative,
  saveSample,
  type Analysis,
  type NegativeKind,
  type CollectStatus,
  type SessionInfo,
  type SessionInput,
} from "./api";

/** Hold-sequence timing (an object so tests can shorten it). */
export const collectTiming = { holdFrames: 10, holdIntervalMs: 300 };
const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

type Frame = { blob: Blob; url: string; analysis: Analysis };
type Review = { kind: "single" | "hold"; frames: Frame[] };

const DEFAULT_SESSION: SessionInput = {
  room: "",
  lighting: "lamp",
  background: "shelves",
  background_notes: "",
  face_visible: true,
  glasses: false,
  person: "",
  camera: "",
  notes: "",
  consent_confirmed: false,
};

/** Admin workspace for collecting LABELLED webcam data: alphabet frames and native-sign clips.
 * The contributor's consent is recorded first, the camera is switched on only by an explicit click, you choose the class,
 * capture, review exactly what would be saved, and then confirm the label before anything is stored. No model prediction is
 * ever shown or used as a label, and nothing here trains, evaluates or promotes a model. */
export function CollectPage() {
  const [status, setStatus] = useState<CollectStatus | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  // per-letter counts the session already had when it was opened (so resumed sessions are not counted twice)
  const [opened, setOpened] = useState<Record<string, number>>({});
  const [openedNative, setOpenedNative] = useState<Record<string, number>>({});
  const begin = (s: SessionInfo) => {
    setOpened(s.counts);
    setOpenedNative(s.native_counts ?? {});
    setSession(s);
  };
  // leaving a session always goes back to the chooser: nothing is captured until a session is picked again
  const changeSession = () => {
    setCameraWanted(false);
    setReview(null);
    setLetter(null);
    setLastSaved([]);
    setMessage(null);
    setSaveError(null);
    setSession(null);
  };
  const [existing, setExisting] = useState<SessionInfo[]>([]);
  const [form, setForm] = useState<SessionInput>(DEFAULT_SESSION);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  // The browser's camera permission prompt only ever appears after the admin clicks "Turn on camera".
  const [cameraWanted, setCameraWanted] = useState(false);
  const [mode, setMode] = useState<"letters" | "native">("letters");
  const [nativeGloss, setNativeGloss] = useState<string | null>(null);

  const [letter, setLetter] = useState<string | null>(null);
  const [hand, setHand] = useState("right");
  const [distance, setDistance] = useState("medium");
  const [review, setReview] = useState<Review | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<{ message: string; reasons: string[] } | null>(null);
  const [lastSaved, setLastSaved] = useState<string[]>([]);

  useEffect(() => {
    collectStatus()
      .then(async (s) => {
        setStatus(s);
        setExisting((await listSessions()).sessions);
      })
      .catch((err: unknown) => {
        const e = err as CollectApiError;
        setProblem(
          e.status === 404
            ? "Data collection is switched off on the backend. Start it with ASL_QUEST_DEV_COLLECT=1 (see the README)."
            : e.status === 401 || e.status === 403
              ? "Dataset collection is available to administrators only."
              : e.message
        );
      });
  }, []);

  // the chooser always shows up-to-date counts (returning from a session would otherwise list stale numbers)
  useEffect(() => {
    if (session || !status) return;
    listSessions().then((r) => setExisting(r.sessions)).catch(() => undefined);
  }, [session, status]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  }, []);

  useEffect(() => {
    if (!session || !cameraWanted) return;
    setCameraError(null);
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user", width: 960, height: 720 }, audio: false })
      .then(async (stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setCameraOn(true);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setCameraError(describeCameraError(err));
        setCameraWanted(false);
      });
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [session, cameraWanted, stopCamera]);

  // free preview URLs when a review is replaced or the page closes
  useEffect(() => {
    return () => review?.frames.forEach((f) => URL.revokeObjectURL(f.url));
  }, [review]);

  async function grabFrame(): Promise<Blob | null> {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    return new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92));
  }

  async function capture(kind: "single" | "hold") {
    if (!letter || busy) return;
    setBusy(true);
    setMessage(null);
    setSaveError(null);
    try {
      const frames: Frame[] = [];
      const count = kind === "hold" ? collectTiming.holdFrames : 1;
      for (let i = 0; i < count; i += 1) {
        const blob = await grabFrame();
        if (!blob) throw new CollectApiError("The camera isn't delivering frames yet.", 0);
        frames.push({ blob, url: URL.createObjectURL(blob), analysis: await analyzeFrame(blob) });
        if (i < count - 1) await sleep(collectTiming.holdIntervalMs);
      }
      setReview({ kind, frames });
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const videoSize = () => (videoRef.current ? `${videoRef.current.videoWidth}x${videoRef.current.videoHeight}` : undefined);

  async function confirmSave() {
    if (!review || !letter || !session || busy) return;
    setBusy(true);
    setSaveError(null);
    try {
      const usable = review.frames.filter((f) => f.analysis.usable);
      const sequenceId = review.kind === "hold" ? `h${Date.now()}` : undefined;
      const saved: string[] = [];
      let latest = session;
      for (const [index, frame] of usable.entries()) {
        const reply = await saveSample({
          blob: frame.blob,
          sessionId: session.session_id,
          letter,
          hand,
          distance,
          mode: review.kind,
          sequenceId,
          frameIndex: sequenceId ? index : undefined,
          videoSize: videoSize(),
        });
        saved.push(reply.sample.sample_id);
        latest = reply.session;
      }
      setSession(latest);
      setLastSaved(saved);
      setMessage(`Saved ${saved.length} frame${saved.length === 1 ? "" : "s"} as ${letter} in session "${session.room || session.session_id}" (${session.session_id}).`);
      setReview(null);
    } catch (err) {
      const e = err as CollectApiError;
      setSaveError({ message: e.message, reasons: e.reasons ?? [] });
    } finally {
      setBusy(false);
    }
  }

  async function captureNegative(kind: NegativeKind) {
    if (!session || busy) return;
    setBusy(true);
    setMessage(null);
    try {
      const blob = await grabFrame();
      if (!blob) throw new CollectApiError("The camera isn't delivering frames yet.", 0);
      const reply = await saveNegative(blob, session.session_id, kind, distance);
      setSession(reply.session);
      const what = kind === "no_hand" ? "no-hand" : "partial-hand";
      setMessage(
        `Saved a ${what} frame. MediaPipe ${reply.negative.detector_found_hand ? "DID find a hand in it (recorded as a false detection)" : "found no hand, as expected"}.`
      );
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function reject() {
    setReview(null);
    setSaveError(null);
    setMessage("Rejected: nothing was saved.");
  }

  async function undo() {
    if (!session || lastSaved.length === 0) return;
    try {
      let latest = session;
      for (const id of lastSaved) latest = (await deleteSample(session.session_id, id)).session;
      setSession(latest);
      setMessage(`Removed the last ${lastSaved.length} saved frame${lastSaved.length === 1 ? "" : "s"}.`);
      setLastSaved([]);
    } catch (err) {
      setMessage(`Couldn't undo: ${(err as Error).message}`);
    }
  }

  // keyboard: Space = capture, Enter = save, Escape = reject (ignored while typing in a field)
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tag === "BUTTON") return;
      if (event.key === " " && !review) {
        event.preventDefault();
        void capture("single");
      } else if (event.key === "Enter" && review) {
        void confirmSave();
      } else if (event.key === "Escape" && review) {
        reject();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (problem) {
    return (
      <PageLayout>
        <PageHeader eyebrow="Admin · Dataset collection" title="Collect labelled data" description="Build a labelled dataset of alphabet frames and native-sign clips." />
        <EmptyState icon="⚠️" title="Collection unavailable" description={problem} />
      </PageLayout>
    );
  }
  if (!status) return <p className="text-[var(--color-ink-soft)]">Checking the backend…</p>;

  // ---------------------------------------------------------------- session setup
  if (!session) {
    const set = <K extends keyof SessionInput>(key: K, value: SessionInput[K]) => setForm((f) => ({ ...f, [key]: value }));
    const typedRoom = form.room.trim().toLowerCase();
    const sameRoom = typedRoom ? existing.filter((s) => (s.room || "").trim().toLowerCase() === typedRoom) : [];
    return (
      <PageLayout>
        <PageHeader
          eyebrow="Admin · Dataset collection"
          title="Collect labelled data"
          description="Describe the room and record the contributor's consent once. Each room/lighting/background combination is one session, and whole sessions are kept together when the data is split for training and testing."
        />
        <Card className="max-w-2xl space-y-4">
          <SectionLabel>New recording session</SectionLabel>
          <label className="block space-y-1 text-sm">
            <span className="font-semibold">Room (e.g. “bedroom shelf wall”)</span>
            <input className="input-field" value={form.room} onChange={(e) => set("room", e.target.value)} />
          </label>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block space-y-1 text-sm">
              <span className="font-semibold">Lighting</span>
              <select className="input-field" value={form.lighting} onChange={(e) => set("lighting", e.target.value)}>
                {status.options.lighting.map((o) => <option key={o}>{o}</option>)}
              </select>
            </label>
            <label className="block space-y-1 text-sm">
              <span className="font-semibold">Background</span>
              <select className="input-field" value={form.background} onChange={(e) => set("background", e.target.value)}>
                {status.options.background.map((o) => <option key={o}>{o}</option>)}
              </select>
            </label>
          </div>
          <div className="flex flex-wrap gap-6 text-sm">
            <label className="flex items-center gap-2"><input type="checkbox" checked={form.face_visible} onChange={(e) => set("face_visible", e.target.checked)} /> Face visible in frame</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={form.glasses} onChange={(e) => set("glasses", e.target.checked)} /> Wearing glasses</label>
          </div>
          <label className="block space-y-1 text-sm">
            <span className="font-semibold">Person signing (e.g. “person-2”; keeps people separate between training and testing)</span>
            <input className="input-field" value={form.person ?? ""} onChange={(e) => set("person", e.target.value)} />
          </label>
          <label className="block space-y-1 text-sm">
            <span className="font-semibold">Notes (optional)</span>
            <input className="input-field" value={form.notes} onChange={(e) => set("notes", e.target.value)} />
          </label>
          <fieldset className="space-y-2 rounded-2xl border-2 border-[var(--color-border)] bg-[var(--color-surface-soft)] p-4 text-sm">
            <legend className="px-1 text-xs font-extrabold uppercase tracking-[0.12em] text-[var(--color-muted)]">Contributor consent</legend>
            <p className="text-[var(--color-ink-soft)]">
              Images and clips can show an identifiable person and their room. Record only people who agreed to be recorded and to have
              these files stored, reviewed and exported for building an ASL dataset. They can ask for their data to be deleted.
            </p>
            <label className="flex items-start gap-2 font-semibold">
              <input type="checkbox" className="mt-1" checked={form.consent_confirmed} onChange={(e) => set("consent_confirmed", e.target.checked)} />
              <span>The person being recorded has agreed to this.</span>
            </label>
          </fieldset>
          {sameRoom.length > 0 && (
            <p role="note" className="rounded-xl bg-[var(--color-warning-soft,#fff4d6)] px-3 py-2 text-sm">
              A session for “{form.room.trim()}” already exists ({sameRoom.map((s) => `${s.total} frames, ${s.session_id}`).join("; ")}).
              “Start session” creates a SEPARATE new one; use “Resume” below to add to the old one.
            </p>
          )}
          <div className="flex flex-wrap gap-3">
            <PrimaryButton
              disabled={!form.room.trim() || !(form.person ?? "").trim() || !form.consent_confirmed}
              onClick={async () => {
                try {
                  begin(await createSession({ ...form, room: form.room.trim(), person: (form.person ?? "").trim() }));
                } catch (err) {
                  setProblem((err as Error).message);
                }
              }}
            >
              Start session
            </PrimaryButton>
          </div>
        </Card>
        {existing.length > 0 && (
          <Card className="max-w-2xl space-y-2">
            <SectionLabel>Resume an existing session (adds to it)</SectionLabel>
            {existing.map((s) => (
              <button key={s.session_id} type="button" aria-label={`Resume session ${s.session_id}`} className="flex w-full items-center justify-between gap-3 rounded-xl border-2 border-[var(--color-border)] px-3 py-2 text-left text-sm hover:border-[var(--color-primary)]" onClick={() => begin(s)}>
                <span>
                  <span className="font-semibold">{s.room || s.session_id}</span> · {s.lighting} · {s.background} · {s.glasses ? "glasses" : "no glasses"}
                  <span className="block font-mono text-xs text-[var(--color-muted)]">{s.session_id}{s.created_at ? ` · created ${new Date(s.created_at).toLocaleString()}` : ""}</span>
                </span>
                <span className="shrink-0 text-[var(--color-muted)]">{s.total} frames</span>
              </button>
            ))}
          </Card>
        )}
      </PageLayout>
    );
  }

  // ---------------------------------------------------------------- collecting
  const target = status.target_per_letter;
  const sessionCount = (l: string) => session.counts[l] ?? 0;
  // all sessions: totals fetched when the page opened, plus what this session has added since it was opened
  const overall = (l: string) => (status.totals[l] ?? 0) + sessionCount(l) - (opened[l] ?? 0);
  const usableInReview = review?.frames.filter((f) => f.analysis.usable).length ?? 0;

  return (
    <PageLayout>
      <PageHeader
        eyebrow={`Session: ${session.room || session.session_id}`}
        title="Collect labelled data"
        description={`${session.lighting} light, ${session.background} background. Choose the letter, make the sign, capture, then confirm. This session so far: ${session.total} frames.`}
        compact
      />

      <div role="status" aria-label="Active session" className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border-2 border-[var(--color-primary)] bg-[var(--color-primary-soft)] px-4 py-3 text-sm">
        <div>
          <span className="font-extrabold">Saving into: {session.room || session.session_id}</span>
          <span className="block font-mono text-xs">{session.session_id}{session.created_at ? ` · created ${new Date(session.created_at).toLocaleString()}` : ""}</span>
          <span className="block text-xs">{session.lighting} · {session.background} · {session.face_visible ? "face visible" : "face hidden"} · {session.glasses ? "glasses" : "no glasses"} · {session.total} frames in this session</span>
        </div>
        <SecondaryButton onClick={changeSession}>Change session</SecondaryButton>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <div className="relative overflow-hidden rounded-[1.75rem] border-4 border-[var(--color-primary-soft)] bg-black">
            <video ref={videoRef} className="aspect-[4/3] w-full object-cover" playsInline muted aria-label="Webcam preview for data collection" />
            {!cameraOn && (
              <div className="absolute inset-0 grid place-items-center bg-black/70 px-6 text-center text-white">
                <div className="space-y-3">
                  <p>{cameraError ?? (cameraWanted ? "Starting camera…" : "The camera is off. Nothing is captured until you turn it on and press Capture.")}</p>
                  {!cameraWanted && <PrimaryButton onClick={() => setCameraWanted(true)}>Turn on camera</PrimaryButton>}
                </div>
              </div>
            )}
            {cameraOn && (
              <span role="status" className="absolute left-3 top-3 rounded-full bg-black/60 px-3 py-1 text-xs font-bold text-white">
                <span className="mr-1 text-[var(--color-danger)]">●</span>Camera on · nothing is saved until you confirm
              </span>
            )}
            <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
          </div>
          {cameraOn && (
            <div>
              <SecondaryButton onClick={() => setCameraWanted(false)}>Turn off camera</SecondaryButton>
            </div>
          )}

          <div role="tablist" aria-label="What to collect" className="flex gap-2">
            {(
              [
                ["letters", "Alphabet frames"],
                ["native", "Native-sign clips"],
              ] as const
            ).map(([id, text]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={mode === id}
                onClick={() => {
                  setMode(id);
                  setReview(null);
                }}
                className={`min-h-10 rounded-full border-2 px-4 text-sm font-bold ${mode === id ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]" : "border-[var(--color-border)]"}`}
              >
                {text}
              </button>
            ))}
          </div>

          {mode === "native" && (
            <NativeClipRecorder
              signs={status.native_signs}
              session={session}
              getStream={() => streamRef.current}
              cameraOn={cameraOn}
              hand={hand}
              targetPerSign={status.target_per_native_sign}
              onGlossChange={setNativeGloss}
              onSaved={setSession}
            />
          )}

          {mode === "letters" && (
          <>
          <Card className="space-y-3">
            <SectionLabel>1 · Choose the letter you are about to sign</SectionLabel>
            <div className="grid grid-cols-9 gap-1.5" role="group" aria-label="Letter to record">
              {status.letters.map((l) => (
                <button
                  key={l}
                  type="button"
                  aria-pressed={letter === l}
                  aria-label={`Letter ${l}, ${sessionCount(l)} frames this session`}
                  onClick={(e) => {
                    setLetter(l);
                    e.currentTarget.blur(); // a focused button would swallow the Space-to-capture shortcut
                  }}
                  className={`min-h-10 rounded-lg border-2 text-sm font-extrabold ${letter === l ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]" : "border-[var(--color-border)]"}`}
                >
                  {l}
                  <span className="block text-[0.6rem] font-semibold text-[var(--color-muted)]">{sessionCount(l)}</span>
                </button>
              ))}
            </div>
            <div className="flex flex-wrap gap-4 text-sm">
              <label className="flex items-center gap-2">Hand
                <select className="input-field !w-auto" value={hand} onChange={(e) => setHand(e.target.value)} aria-label="Hand used">
                  {status.options.hand.map((o) => <option key={o}>{o}</option>)}
                </select>
              </label>
              <label className="flex items-center gap-2">Distance
                <select className="input-field !w-auto" value={distance} onChange={(e) => setDistance(e.target.value)} aria-label="Distance from camera">
                  {status.options.distance.map((o) => <option key={o}>{o}</option>)}
                </select>
              </label>
            </div>
            <div className="flex flex-wrap gap-3">
              <PrimaryButton disabled={!letter || busy || !cameraOn || !!review} onClick={() => void capture("single")}>
                {letter ? `Capture ${letter} (Space)` : "Choose a letter first"}
              </PrimaryButton>
              <SecondaryButton disabled={!letter || busy || !cameraOn || !!review} onClick={() => void capture("hold")} ariaLabel="Capture a hold sequence of 10 frames">
                Capture hold (10 frames, 3 s)
              </SecondaryButton>
              {lastSaved.length > 0 && <SecondaryButton onClick={() => void undo()}>Undo last save</SecondaryButton>}
            </div>
            <p className="text-xs text-[var(--color-muted)]">
              Vary your hand position, distance, angle and the room between captures. Near-identical consecutive frames are rejected.
              Hold sequences are for measuring how stable a held sign is.
            </p>
          </Card>

          <Card className="space-y-3" data-testid="negatives-card">
            <SectionLabel>Frames with no usable hand (for the held-out evaluation)</SectionLabel>
            <p className="text-xs text-[var(--color-muted)]">
              The recognizer must NOT answer on these. No-hand: both hands out of view. Partial hand: a hand cut off by the frame edge. No letter is needed;
              MediaPipe's verdict is recorded next to your label, never used as it.
            </p>
            <div className="flex flex-wrap gap-3">
              <SecondaryButton disabled={busy || !cameraOn} onClick={() => void captureNegative("no_hand")}>Save NO-HAND frame</SecondaryButton>
              <SecondaryButton disabled={busy || !cameraOn} onClick={() => void captureNegative("partial_hand")}>Save PARTIAL-HAND frame</SecondaryButton>
            </div>
            <p className="text-xs text-[var(--color-muted)]">
              This session: {session.negatives?.no_hand ?? 0} no-hand, {session.negatives?.partial_hand ?? 0} partial-hand.
            </p>
          </Card>
          </>
          )}
        </div>

        <div className="space-y-4">
          {mode === "letters" && letter && (
            <Card className="flex items-center gap-4">
              <img src={referenceImageSrc(letter)} alt={referenceImageAlt(letter)} className="h-24 w-auto rounded-xl bg-[var(--color-surface-soft)] p-1" />
              <div>
                <p className="text-sm text-[var(--color-muted)]">Reference for the letter you chose</p>
                <p className="font-display text-4xl font-bold text-[var(--color-primary)]">{letter}</p>
                <p className="text-xs text-[var(--color-muted)]">{overall(letter)} saved overall (target {target})</p>
              </div>
            </Card>
          )}

          {mode === "letters" && (
          <Card className="space-y-3" data-testid="review-panel">
            <SectionLabel>2 · Review, then confirm</SectionLabel>
            {message && <p role="status" className="text-sm font-semibold text-[var(--color-primary-dim)]">{message}</p>}
            {!review ? (
              <p className="text-sm text-[var(--color-ink-soft)]">Capture a frame to see the exact crop the recognizer would receive and whether it is good enough.</p>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {review.frames.map((frame, i) => (
                    <figure key={frame.url} className={`rounded-xl border-2 p-1 ${frame.analysis.usable ? "border-[var(--color-primary)]" : "border-[var(--color-danger)]"}`}>
                      <img src={frame.analysis.crop_preview ?? frame.url} alt={`Frame ${i + 1}: ${frame.analysis.usable ? "usable" : "rejected"}`} className="aspect-square w-full rounded-lg object-cover" />
                      <figcaption className="mt-1 text-[0.7rem] leading-tight">
                        <StatusChip tone={frame.analysis.usable ? "success" : "warning"}>{frame.analysis.usable ? "usable" : "rejected"}</StatusChip>
                        {frame.analysis.reasons.map((r) => <span key={r} className="mt-1 block text-[var(--color-danger)]">{r}</span>)}
                        {frame.analysis.warnings.map((w) => <span key={w} className="mt-1 block text-[var(--color-muted)]">{w}</span>)}
                      </figcaption>
                    </figure>
                  ))}
                </div>
                {saveError && (
                  <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
                    {saveError.message}{saveError.reasons.length > 0 ? ` (${saveError.reasons.join("; ")})` : ""}
                  </p>
                )}
                <div className="flex flex-wrap gap-3">
                  <PrimaryButton disabled={busy || usableInReview === 0 || !letter} onClick={() => void confirmSave()}>
                    {`Yes, this is ${letter} — save ${usableInReview} frame${usableInReview === 1 ? "" : "s"} (Enter)`}
                  </PrimaryButton>
                  <SecondaryButton onClick={reject}>Reject (Esc)</SecondaryButton>
                </div>
                <p className="text-xs text-[var(--color-muted)]">
                  Only save if the hand really shows {letter}. The label is exactly what you confirm here; the recognizer is never consulted.
                </p>
              </>
            )}
          </Card>
          )}

          <SampleGallery
            session={session}
            mode={mode}
            label={mode === "letters" ? letter : nativeGloss}
            onChanged={(updated) => {
              setSession(updated);
              setLastSaved([]);
            }}
          />

          {status.native_signs.length > 0 && (
            <Card className="space-y-2" data-testid="native-progress">
              <SectionLabel>Native-sign clips (all sessions, target {status.target_per_native_sign} per sign)</SectionLabel>
              <div className="grid grid-cols-1 gap-y-1">
                {status.native_signs.map((sign) => {
                  const total = (status.native_totals[sign.gloss] ?? 0) + (session.native_counts?.[sign.gloss] ?? 0) - (openedNative[sign.gloss] ?? 0);
                  return (
                    <div key={sign.gloss} className="flex items-center gap-2 text-xs">
                      <span className="w-24 truncate font-bold">{sign.display_name}</span>
                      <div className="flex-1"><ProgressBar percent={Math.min(100, (total / status.target_per_native_sign) * 100)} showPercent={false} size="sm" /></div>
                      <span className="w-12 text-right tabular-nums text-[var(--color-muted)]">{total}</span>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          <ExportCard sessionId={session.session_id} total={session.total + (session.native_total ?? 0)} />

          <Card className="space-y-2">
            <SectionLabel>Progress (all sessions, target {target} per letter)</SectionLabel>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {status.letters.map((l) => (
                <div key={l} className="flex items-center gap-2 text-xs">
                  <span className="w-4 font-bold">{l}</span>
                  <div className="flex-1"><ProgressBar percent={Math.min(100, (overall(l) / target) * 100)} showPercent={false} size="sm" /></div>
                  <span className="w-12 text-right tabular-nums text-[var(--color-muted)]">{overall(l)}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </PageLayout>
  );
}
