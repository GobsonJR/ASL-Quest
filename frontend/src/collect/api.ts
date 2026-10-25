import { authHeaders } from "../auth/token";
import { API_BASE } from "../config";

// Client for the ADMIN-ONLY labelled-data API (backend/routers/collect.py). Every call carries the signed-in admin's bearer token;
// the server decides from its own account data whether that person is an admin (401/403 otherwise), and answers 404 unless the
// operator switched collection on with ASL_QUEST_DEV_COLLECT=1.

export type CollectOptions = {
  lighting: string[];
  background: string[];
  hand: string[];
  distance: string[];
};

export type NativeSignOption = { gloss: string; display_name: string };

export type CollectStatus = {
  enabled: true;
  letters: string[];
  native_signs: NativeSignOption[];
  options: CollectOptions;
  target_per_letter: number;
  target_per_native_sign: number;
  totals: Record<string, number>;
  native_totals: Record<string, number>;
  limits: { max_image_bytes: number; max_clip_bytes: number; image_formats: string[]; clip_formats: string[] };
  consent_required: boolean;
  thresholds: { min_sharpness: number; min_hand_px: number; brightness_range: [number, number] };
  hand_detector_available: boolean;
};

export type SessionInput = {
  room: string;
  lighting: string;
  background: string;
  background_notes: string;
  face_visible: boolean;
  glasses: boolean;
  /** Who is signing; held-out evaluation keeps people disjoint from training. */
  person?: string;
  camera: string;
  notes: string;
  /** The person being recorded has agreed. The server refuses to create a session without it. */
  consent_confirmed: boolean;
};

export type SessionInfo = Omit<SessionInput, "consent_confirmed"> & {
  session_id: string;
  created_at?: string;
  consent?: { confirmed: boolean; recorded_at?: string; recorded_by?: string };
  counts: Record<string, number>;
  total: number;
  negatives?: Record<string, number>;
  native_counts?: Record<string, number>;
  native_total?: number;
};

export type Analysis = {
  usable: boolean;
  reasons: string[];
  warnings: string[];
  metrics: Record<string, unknown>;
  hand_box: [number, number, number, number] | null;
  crop_preview: string | null;
};

export type SavedSample = { sample: { sample_id: string; letter: string }; session: SessionInfo };

export type SampleRecord = {
  sample_id: string;
  session_id: string;
  letter: string;
  captured_at: string;
  hand: string;
  distance: string;
  mode: string;
};

export type ClipRecord = {
  sample_id: string;
  session_id: string;
  gloss: string;
  captured_at: string;
  bytes: number;
  duration_ms: number | null;
  hand: string | null;
};

export class CollectApiError extends Error {
  status: number;
  reasons: string[];
  constructor(message: string, status: number, reasons: string[] = []) {
    super(message);
    this.status = status;
    this.reasons = reasons;
  }
}

async function failure(response: Response): Promise<CollectApiError> {
  let message = `Request failed (${response.status}).`;
  let reasons: string[] = [];
  try {
    const body = await response.json();
    const detail = body.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      message = detail.message ?? message;
      reasons = detail.reasons ?? [];
    } else if (typeof detail === "string") {
      message = detail;
    }
  } catch {
    /* keep the default message */
  }
  if (response.status === 401) message = "Your session has expired. Sign in again.";
  if (response.status === 403) message = "Administrator access is required.";
  return new CollectApiError(message, response.status, reasons);
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = authHeaders();
  if (token.Authorization) headers.set("Authorization", token.Authorization);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/collect${path}`, { ...init, headers });
  } catch {
    throw new CollectApiError("Can't reach the backend.", 0);
  }
  if (!response.ok) throw await failure(response);
  return response;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return (await send(path, init)).json() as Promise<T>;
}

export const collectStatus = () => request<CollectStatus>("/status");

export const listSessions = () => request<{ sessions: SessionInfo[]; totals: Record<string, number> }>("/sessions");

export const createSession = (input: SessionInput) =>
  request<SessionInfo>("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

export const getSession = (id: string) => request<SessionInfo>(`/sessions/${id}`);

export function analyzeFrame(blob: Blob): Promise<Analysis> {
  const body = new FormData();
  body.append("file", blob, "frame.jpg");
  return request<Analysis>("/analyze", { method: "POST", body });
}

export type SaveSampleInput = {
  blob: Blob;
  sessionId: string;
  /** The letter the person selected AND confirmed. It is never filled in from a model prediction. */
  letter: string;
  hand: string;
  distance: string;
  mode: "single" | "hold";
  sequenceId?: string;
  frameIndex?: number;
  videoSize?: string;
};

export function saveSample(input: SaveSampleInput): Promise<SavedSample> {
  const body = new FormData();
  body.append("file", input.blob, "frame.jpg");
  body.append("session_id", input.sessionId);
  body.append("letter", input.letter);
  body.append("confirmed", "true");
  body.append("hand", input.hand);
  body.append("distance", input.distance);
  body.append("mode", input.mode);
  if (input.sequenceId) body.append("sequence_id", input.sequenceId);
  if (input.frameIndex !== undefined) body.append("frame_index", String(input.frameIndex));
  if (input.videoSize) body.append("video_size", input.videoSize);
  return request<SavedSample>("/samples", { method: "POST", body });
}

export type NegativeKind = "no_hand" | "partial_hand";
export type SavedNegative = { negative: { sample_id: string; kind: NegativeKind; detector_found_hand: boolean }; session: SessionInfo };

/** A frame the person confirms has NO usable hand (or a hand cut by the frame edge): the recognizer must not answer on it. */
export function saveNegative(blob: Blob, sessionId: string, kind: NegativeKind, distance?: string): Promise<SavedNegative> {
  const body = new FormData();
  body.append("file", blob, "frame.jpg");
  body.append("session_id", sessionId);
  body.append("kind", kind);
  body.append("confirmed", "true");
  if (distance) body.append("distance", distance);
  return request<SavedNegative>("/negatives", { method: "POST", body });
}

export const deleteSample = (sessionId: string, sampleId: string) =>
  request<{ deleted: string; session: SessionInfo }>(`/sessions/${sessionId}/samples/${sampleId}`, { method: "DELETE" });

// ---------------------------------------------------------------- native-sign clips
export type SaveClipInput = {
  blob: Blob;
  sessionId: string;
  /** The sign the person selected AND confirmed (an active native-sign gloss). */
  gloss: string;
  hand?: string;
  durationMs?: number;
};

export function saveNativeClip(input: SaveClipInput): Promise<{ clip: ClipRecord; session: SessionInfo }> {
  const body = new FormData();
  const extension = input.blob.type.includes("mp4") ? "mp4" : "webm";
  body.append("file", input.blob, `clip.${extension}`);
  body.append("session_id", input.sessionId);
  body.append("gloss", input.gloss);
  body.append("confirmed", "true");
  if (input.hand) body.append("hand", input.hand);
  if (input.durationMs !== undefined) body.append("duration_ms", String(Math.round(input.durationMs)));
  return request("/native-clips", { method: "POST", body });
}

export const deleteNativeClip = (sessionId: string, clipId: string) =>
  request<{ deleted: string; session: SessionInfo }>(`/sessions/${sessionId}/native-clips/${clipId}`, { method: "DELETE" });

// ---------------------------------------------------------------- review gallery
export const listSamples = (sessionId: string, letter?: string) =>
  request<{ samples: SampleRecord[] }>(`/sessions/${sessionId}/samples${letter ? `?label=${encodeURIComponent(letter)}` : ""}`);

export const listNativeClips = (sessionId: string, gloss?: string) =>
  request<{ clips: ClipRecord[] }>(`/sessions/${sessionId}/native-clips${gloss ? `?gloss=${encodeURIComponent(gloss)}` : ""}`);

/** Files are admin-only, and an <img>/<video> tag cannot send the bearer token, so they are fetched with it and shown from a blob URL. */
export async function fetchBlobUrl(path: string): Promise<string> {
  const response = await send(path);
  return URL.createObjectURL(await response.blob());
}

export const sampleImageUrl = (sessionId: string, sampleId: string) => fetchBlobUrl(`/sessions/${sessionId}/samples/${sampleId}/image`);
export const clipVideoUrl = (sessionId: string, clipId: string) => fetchBlobUrl(`/sessions/${sessionId}/native-clips/${clipId}/video`);

// ---------------------------------------------------------------- export
export type ExportResult = { blob: Blob; files: number; skippedSessions: number };

/** The labelled dataset as a zip (images/, native_clips/, manifest.csv/.jsonl, README.txt). Sessions without consent are skipped. */
export async function exportDataset(sessionId?: string): Promise<ExportResult> {
  const response = await send(`/export${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`);
  return {
    blob: await response.blob(),
    files: Number(response.headers.get("X-Export-Files") ?? 0),
    skippedSessions: Number(response.headers.get("X-Export-Skipped-Sessions") ?? 0),
  };
}
