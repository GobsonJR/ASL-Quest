import type { GameState } from "../game/types";
import { authHeaders, clearToken, setToken } from "./token";

const API = "/api";

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
  if (tokenHeaders.Authorization) {
    headers.set("Authorization", tokenHeaders.Authorization);
  }
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) {
    throw new Error(await parseError(response, "Request failed."));
  }
  return response.json();
}

export async function registerUser(input: {
  username: string;
  email: string;
  password: string;
}): Promise<UserProfile> {
  const auth = await request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify(input),
  });
  setToken(auth.access_token);
  return fetchCurrentUser();
}

export async function loginUser(login: string, password: string): Promise<UserProfile> {
  const auth = await request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ login, password }),
  });
  setToken(auth.access_token);
  return fetchCurrentUser();
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  return request<UserProfile>("/auth/me");
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
  return request("/progress/migration-offer");
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
  } catch {
    clearToken();
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
