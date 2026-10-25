import type { GameState } from "../game/types";
import { authHeaders, clearToken, setToken } from "./token";
import { API_BASE as API } from "../config";

export type UserProfile = {
  id: number;
  username: string;
  email: string;
  role?: string;
  created_at: string;
};

export type FullUserProfile = UserProfile & {
  level: number;
  xp: number;
  current_streak: number;
  best_streak: number;
  letters_mastered: number;
  total_practice_sessions: number;
  overall_accuracy: number | null;
  total_signs_practiced: number;
  total_days_active: number;
  most_practiced_letter: string | null;
  strongest_letter: string | null;
  weakest_letter: string | null;
  avatar_initial: string;
};

export type RecommendationsResponse = {
  has_data: boolean;
  recommended: Array<{ letter: string; reason: string }>;
  weakest: { letter: string; accuracy: number | null; attempts: number } | null;
  strongest: { letter: string; accuracy: number | null; attempts: number } | null;
  message: string | null;
};

export type AdminOverview = {
  total_users: number;
  active_users: number;
  total_practice_sessions: number;
  total_xp_awarded: number;
  average_accuracy: number | null;
  total_letter_attempts: number;
  total_word_practice_sessions?: number;
  words_completed?: number;
  average_word_accuracy?: number | null;
};

export type AdminUsersPage = {
  page: number;
  page_size: number;
  total: number;
  items: Array<{
    id: number;
    username: string;
    email: string;
    role: string;
    created_at: string;
    practice_sessions: number;
    xp: number;
    last_activity: string | null;
  }>;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
};

export type SyncStatus = "idle" | "syncing" | "synced" | "offline" | "error";

export async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === "string" && detail) return detail;
    // FastAPI validation errors (422) send a list of { msg, loc, ... } objects.
    if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
    return fallback;
  } catch {
    // Non-JSON body (e.g. a plain-text 404/502 from a proxy or host).
    return fallback;
  }
}

/** How long sign-in and registration wait for the backend. A free-tier Render instance can take 30-50 s to wake, so
 * rather than leave the form on "Signing in..." the request is abandoned and the user is told to retry. */
export const AUTH_TIMEOUT_MS = 25_000;

/** An API failure with its cause kept apart: `timeout` and `network` mean the server could not be reached (so a saved
 * session must not be discarded), `http` carries the response status (401 = invalid credentials/token). */
export class ApiError extends Error {
  constructor(message: string, readonly kind: "timeout" | "network" | "http" | "invalid-response", readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

const TIMEOUT_MESSAGE = "The ASL Quest server is taking too long to respond (it may be waking up). Please try again in a moment.";

/** fetch() that turns a network-level failure (backend asleep/unreachable, CORS blocked, offline)
 * into a readable message instead of the browser's bare "Failed to fetch". */
async function fetchApi(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API}${path}`, init);
  } catch {
    if (init.signal?.aborted) throw new ApiError(TIMEOUT_MESSAGE, "timeout");
    throw new ApiError("Can't reach the ASL Quest server. Check your connection and try again in a moment.", "network");
  }
}

/** With `timeoutMs` the whole exchange -- connecting, the response headers AND reading the body -- is abandoned after
 * that long and reported as a retryable timeout, so no stage of the request can leave a caller waiting forever. */
async function request<T>(path: string, init: RequestInit = {}, timeoutMs?: number): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) {
    headers.set("Authorization", tokenHeaders.Authorization);
  }
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetchApi(path, controller ? { ...init, headers, signal: controller.signal } : { ...init, headers });
    if (!response.ok) {
      const message = await parseError(response, `Request failed (${response.status}).`).catch(() => `Request failed (${response.status}).`);
      throw new ApiError(controller?.signal.aborted ? TIMEOUT_MESSAGE : message, controller?.signal.aborted ? "timeout" : "http", response.status);
    }
    try {
      return (await response.json()) as T;
    } catch {
      if (controller?.signal.aborted) throw new ApiError(TIMEOUT_MESSAGE, "timeout");
      throw new ApiError("The server sent an unexpected response. Please try again.", "invalid-response");
    }
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export async function registerUser(input: {
  username: string;
  email: string;
  password: string;
}): Promise<UserProfile> {
  const auth = await request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify(input),
  }, AUTH_TIMEOUT_MS);
  setToken(auth.access_token);
  try {
    return await fetchCurrentUser();
  } catch (err) {
    // The account now exists but this page never got its profile; retrying "Create account" would answer "already
    // registered", so say what happened and leave no half-open session behind.
    clearToken();
    if (err instanceof ApiError && err.kind !== "http") {
      throw new ApiError("Your account was created, but the server was too slow to finish signing you in. Please sign in with it in a moment.", err.kind);
    }
    throw err;
  }
}

export async function loginUser(login: string, password: string): Promise<UserProfile> {
  const auth = await request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ login, password }),
  }, AUTH_TIMEOUT_MS);
  setToken(auth.access_token);
  try {
    return await fetchCurrentUser();
  } catch (err) {
    clearToken(); // not signed in on this page; a retry starts from a clean login
    throw err;
  }
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  return request<UserProfile>("/auth/me", {}, AUTH_TIMEOUT_MS);
}

export function logoutUser(): void {
  clearToken();
}

export async function fetchProgress(): Promise<GameState> {
  return request<GameState>("/progress");
}

export async function saveProgress(state: GameState): Promise<GameState> {
  return request<GameState>("/progress", {
    method: "PUT",
    body: JSON.stringify(state),
  });
}

export async function migrateLocalProgress(state: GameState): Promise<GameState> {
  return request<GameState>("/progress/migrate-local", {
    method: "POST",
    body: JSON.stringify(state),
  });
}

export async function fetchMigrationOffer(): Promise<{ should_offer: boolean; reason: string | null }> {
  return request("/progress/migration-offer", {}, AUTH_TIMEOUT_MS);
}

export async function recordPracticeSession(input: {
  letter: string;
  prediction: string | null;
  correct: boolean;
  response_time?: number;
  xp_earned?: number;
  challenge_type?: string | null;
  confidence?: number | null;
  reason?: string | null;
}): Promise<void> {
  await request("/practice/session", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function restoreSession(): Promise<UserProfile | null> {
  if (!authHeaders().Authorization) return null;
  try {
    return await fetchCurrentUser();
  } catch (err) {
    // Only a 401 means the saved token is bad. A timeout, network error or 5xx means the server is unavailable, so the
    // token is kept and the next page load tries again instead of silently signing the user out.
    if (err instanceof ApiError && err.kind === "http" && err.status === 401) clearToken();
    return null;
  }
}

export async function fetchProfile(): Promise<FullUserProfile> {
  return request<FullUserProfile>("/users/me/profile");
}

export async function fetchPreferences(): Promise<{ reduced_motion?: boolean }> {
  return request("/users/me/preferences");
}

export async function updatePreferences(input: { reduced_motion?: boolean }): Promise<{ reduced_motion?: boolean }> {
  return request("/users/me/preferences", { method: "PUT", body: JSON.stringify(input) });
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}/users/me/change-password`, {
    method: "POST",
    headers,
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!response.ok) throw new Error(await parseError(response, "Password change failed."));
}

export async function deleteAccount(password: string, confirm: string): Promise<void> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}/users/me`, {
    method: "DELETE",
    headers,
    body: JSON.stringify({ password, confirm }),
  });
  if (!response.ok) throw new Error(await parseError(response, "Account deletion failed."));
}

export async function exportUserData(): Promise<Record<string, unknown>> {
  return request("/users/me/export");
}

export async function fetchRecommendations(): Promise<RecommendationsResponse> {
  return request("/recommendations");
}

export async function fetchAdminOverview(): Promise<AdminOverview> {
  return request("/admin/overview");
}

export async function fetchAdminUsers(params: {
  page?: number;
  search?: string;
  role?: string;
}): Promise<AdminUsersPage> {
  const query = new URLSearchParams();
  if (params.page) query.set("page", String(params.page));
  if (params.search) query.set("search", params.search);
  if (params.role) query.set("role", params.role);
  return request(`/admin/users?${query.toString()}`);
}

export async function fetchAdminActivity(): Promise<{ points: Array<{ date: string; value: number }> }> {
  return request("/admin/activity");
}

export async function fetchAdminLetters(): Promise<{
  popular: Array<{ letter: string; attempts: number; accuracy: number | null }>;
  difficult: Array<{ letter: string; attempts: number; accuracy: number | null }>;
}> {
  return request("/admin/letters");
}

export async function fetchAdminWords(): Promise<{
  most_practiced: Array<{ word: string; sessions: number; accuracy: number | null }>;
  hardest: Array<{ word: string; sessions: number; accuracy: number | null }>;
  most_popular_categories: Array<{ category: string; sessions: number }>;
}> {
  return request("/admin/words");
}

export async function fetchAdminRecentActivity(): Promise<{
  items: Array<{ user_id: number; letter: string; correct: boolean; created_at: string }>;
}> {
  return request("/admin/recent-activity");
}
