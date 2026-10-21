import { API_BASE } from "../config";

// Client for the development-only labelled-webcam-data API (backend/routers/collect.py). The backend answers
// 404 unless it was started with ASL_QUEST_DEV_COLLECT=1.

export type CollectOptions = {
  lighting: string[];
  background: string[];
  hand: string[];
  distance: string[];
};

export type CollectStatus = {
  enabled: true;
  letters: string[];
  options: CollectOptions;
  target_per_letter: number;
  totals: Record<string, number>;
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
  camera: string;
  notes: string;
};

export type SessionInfo = SessionInput & {
  session_id: string;
  created_at?: string;
  counts: Record<string, number>;
  total: number;
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

export class CollectApiError extends Error {
  status: number;
  reasons: string[];
  constructor(message: string, status: number, reasons: string[] = []) {
    super(message);
    this.status = status;
    this.reasons = reasons;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/collect${path}`, init);
  } catch {
    throw new CollectApiError("Can't reach the backend.", 0);
  }
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    let reasons: string[] = [];
    try {
      const body = await response.json();
      const detail = body.detail;
      if (detail && typeof detail === "object") {
        message = detail.message ?? message;
        reasons = detail.reasons ?? [];
      } else if (typeof detail === "string") {
        message = detail;
      }
    } catch {
      /* keep the default message */
    }
    throw new CollectApiError(message, response.status, reasons);
  }
  return response.json() as Promise<T>;
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

export const deleteSample = (sessionId: string, sampleId: string) =>
  request<{ deleted: string; session: SessionInfo }>(`/sessions/${sessionId}/samples/${sampleId}`, { method: "DELETE" });
