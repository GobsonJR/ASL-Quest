import { authHeaders } from "../auth/token";

const API = "/api";

export type NativeSignReference = {
  available: boolean;
  video_url: string | null;
  thumbnail_url: string | null;
  source_type: string | null;
  license_note: string | null;
};

export type NativeSign = {
  id: number;
  gloss: string;
  display_name: string;
  meaning: string | null;
  category: string | null;
  difficulty: string | null;
  description: string | null;
  example_text: string | null;
  dataset_available: boolean;
  model_available: boolean;
  active: boolean;
  reference: NativeSignReference;
};

export type NativeTopKItem = {
  gloss: string;
  confidence: number;
};

export type NativePrediction = {
  prediction: string;
  confidence: number;
  top_k: NativeTopKItem[];
  model_type: string;
  model_version: string;
  latency_ms: number;
  native_sign_id: number | null;
  correct: boolean | null;
};

export type NativeAchievement = { id: string; name: string };

export type NativePracticeResult = NativePrediction & {
  expected_sign: { id: number; gloss: string; display_name: string };
  session_id: number;
  response_time: number;
  xp_earned: number;
  new_achievements: NativeAchievement[];
};

export type NativeSignProgressEntry = {
  sign_id: number;
  gloss: string;
  display_name: string;
  attempts: number;
  correct: number;
  mastery: number;
  last_practiced: string | null;
};

export type NativeProgress = {
  total_signs: number;
  started_signs: number;
  mastered_signs: number;
  overall_mastery: number;
  signs: NativeSignProgressEntry[];
};

async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail ?? fallback;
  } catch {
    return fallback;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) throw new Error(await parseError(response, "Native sign request failed."));
  return response.json();
}

export function getNativeSigns(): Promise<{ items: NativeSign[] }> {
  return request<{ items: NativeSign[] }>("/native/signs");
}

export function getNativeProgress(): Promise<NativeProgress> {
  return request<NativeProgress>("/native/progress");
}

export function getNativeSignProgress(signId: number): Promise<NativeSignProgressEntry> {
  return request<NativeSignProgressEntry>(`/native/progress/${signId}`);
}

async function postNativeVideo<T>(
  file: Blob,
  filename: string,
  params: URLSearchParams
): Promise<T> {
  const body = new FormData();
  body.append("file", file, filename);

  const headers = new Headers();
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  // No Content-Type set here: the browser must set its own multipart boundary for FormData.

  const query = params.toString();
  const response = await fetch(`${API}/native/predict${query ? `?${query}` : ""}`, {
    method: "POST",
    headers,
    body,
  });
  if (!response.ok) throw new Error(await parseError(response, "Native sign prediction failed."));
  return response.json();
}

export function predictNativeSign(
  file: Blob,
  filename = "clip.mp4",
  options: { topK?: number; expectedLabel?: string } = {}
): Promise<NativePrediction> {
  const params = new URLSearchParams();
  if (options.topK != null) params.set("top_k", String(options.topK));
  if (options.expectedLabel != null) params.set("expected_label", options.expectedLabel);
  return postNativeVideo<NativePrediction>(file, filename, params);
}

/** The real practice flow: server resolves nativeSignId to the actual catalog gloss,
 * determines correctness itself, and records a session + updates progress. */
export function practiceNativeSign(
  file: Blob,
  nativeSignId: number,
  filename = "clip.mp4",
  topK = 5
): Promise<NativePracticeResult> {
  const params = new URLSearchParams();
  params.set("native_sign_id", String(nativeSignId));
  params.set("top_k", String(topK));
  return postNativeVideo<NativePracticeResult>(file, filename, params);
}
