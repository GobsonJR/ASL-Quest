import { authHeaders } from "../auth/token";
import { API_BASE as API } from "../config";

export type MentorOverview = {
  total_users: number;
  active_users: number;
  total_practice_sessions: number;
  total_xp_awarded: number;
  average_accuracy: number | null;
  total_letter_attempts: number;
  total_word_practice_sessions: number;
  words_completed: number;
  average_word_accuracy: number | null;
  total_native_sign_attempts: number;
};

export type MentorStudent = {
  id: number;
  username: string;
  role: string;
  level: number;
  xp: number;
  current_streak: number;
  accuracy: number | null;
  native_mastery: number | null;
};

export type MentorLetterStat = { letter: string; attempts: number; accuracy: number | null };

export type MentorWordStat = { word: string; word_id: string; sessions: number; accuracy: number | null };

export type MentorCategoryStat = { category: string; sessions: number };

export type MentorNativeSignStat = {
  sign_id: number;
  gloss: string;
  display_name: string;
  category: string | null;
  attempts: number;
  mastery: number | null;
};

export type MentorNativeCategoryStat = { category: string; attempts: number };

export type MentorDashboard = {
  overview: MentorOverview;
  students: MentorStudent[];
  az: { popular: MentorLetterStat[]; difficult: MentorLetterStat[] };
  words: {
    total_word_practice_sessions: number;
    words_completed: number;
    average_word_accuracy: number | null;
    most_practiced: MentorWordStat[];
    hardest: MentorWordStat[];
    most_popular_categories: MentorCategoryStat[];
  };
  native: {
    total_attempts: number;
    signs_mastered_by_someone: number;
    per_sign: MentorNativeSignStat[];
    category_distribution: MentorNativeCategoryStat[];
  };
  chatbot: {
    total_conversations: number;
    total_messages: number;
    positive_feedback: number;
    negative_feedback: number;
  };
};

async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function fetchMentorDashboard(): Promise<MentorDashboard> {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}/admin/mentor-dashboard`, { headers });
  if (!response.ok) throw new Error(await parseError(response, "Unable to load the mentor dashboard."));
  return response.json();
}

// ------------------------------------------------------------------ management API (backend/routers/admin_ops.py)

async function adminRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { ...init, headers });
  } catch {
    throw new Error("Can't reach the backend.");
  }
  if (!response.ok) {
    const detail = await parseError(response, `Request failed (${response.status}).`);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

const send = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });

export type AdminUserRow = {
  id: number;
  username: string;
  email: string;
  role: string;
  created_at: string;
  practice_sessions: number;
  xp: number;
  last_activity: string | null;
};
export type AdminUsersPage = { page: number; page_size: number; total: number; items: AdminUserRow[] };
export type AdminUserDetail = {
  id: number;
  username: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
  xp: number;
  level: number;
  current_streak: number;
  best_streak: number;
  practice_sessions: number;
  accuracy: number | null;
  achievements: number;
  word_sessions: number;
  native_sessions: number;
  recent_activity: Array<{ letter: string; correct: boolean; created_at: string }>;
};

export const listAdminUsers = (p: { page: number; search?: string; role?: string }) => {
  const q = new URLSearchParams({ page: String(p.page), page_size: "15" });
  if (p.search) q.set("search", p.search);
  if (p.role) q.set("role", p.role);
  return adminRequest<AdminUsersPage>(`/admin/users?${q}`);
};
export type CreatedUser = { id: number; username: string; email: string; role: string; is_active: boolean };
export const createAdminUser = (body: { username: string; email: string; password: string; role: "student" | "admin" }) =>
  adminRequest<CreatedUser>("/admin/users", send("POST", body));
export const getAdminUser = (id: number) => adminRequest<AdminUserDetail>(`/admin/users/${id}`);
export const patchAdminUser = (id: number, body: { role?: string; is_active?: boolean }) =>
  adminRequest<{ id: number; role: string; is_active: boolean }>(`/admin/users/${id}`, send("PATCH", body));
export const resetAdminUserProgress = (id: number) => adminRequest<{ reset: number }>(`/admin/users/${id}/reset-progress`, send("POST"));
export const deleteAdminUser = (id: number, confirm: string) =>
  adminRequest<{ deleted: number }>(`/admin/users/${id}?confirm=${encodeURIComponent(confirm)}`, send("DELETE"));

export type AdminNativeSign = {
  id: number;
  gloss: string;
  display_name: string;
  meaning: string | null;
  description: string | null;
  category: string | null;
  difficulty: string | null;
  active: boolean;
  dataset_available: boolean;
  model_available: boolean;
};
export type AdminAchievement = { id: number; slug: string; name: string; description: string; requirement: string; unlocked_by: number };

export const listAdminNativeSigns = () => adminRequest<{ items: AdminNativeSign[] }>("/admin/native-signs");
export const patchAdminNativeSign = (id: number, body: Partial<Pick<AdminNativeSign, "display_name" | "meaning" | "description" | "category" | "difficulty" | "active">>) =>
  adminRequest<AdminNativeSign>(`/admin/native-signs/${id}`, send("PATCH", body));
export const listAdminAchievements = () => adminRequest<{ items: AdminAchievement[] }>("/admin/achievements");
export const patchAdminAchievement = (id: number, body: { name?: string; description?: string }) =>
  adminRequest<AdminAchievement>(`/admin/achievements/${id}`, send("PATCH", body));

export type AdminModelRow = {
  model_type: string;
  version: string;
  registered: boolean;
  checkpoint_path: string | null;
  num_classes: number | null;
  created_at: string | null;
  predictions: number;
  graded: number;
  accuracy: number | null;
  avg_latency_ms: number | null;
};
export type AdminPrediction = { user_id: number; model_version: string; predicted: string; expected: string | null; confidence: number | null; correct: boolean | null; created_at: string };
export type AdminChatFeedback = {
  summary: { conversations: number; messages: number; positive: number; negative: number };
  items: Array<{ rating: number | null; comment: string | null; answer: string; question: string | null; created_at: string }>;
};
export const listAdminModels = () => adminRequest<{ items: AdminModelRow[] }>("/admin/models");
export const listAdminPredictions = () => adminRequest<{ items: AdminPrediction[] }>("/admin/predictions");
export const getAdminChatFeedback = (negativeOnly: boolean) => adminRequest<AdminChatFeedback>(`/admin/chatbot/feedback?negative_only=${negativeOnly}`);

export type AdminDatasetSession = {
  session_id: string;
  room: string;
  lighting: string;
  background: string;
  glasses: boolean;
  created_at?: string;
  total: number;
  counts: Record<string, number>;
};
export const listAdminDataset = () =>
  adminRequest<{ sessions: AdminDatasetSession[]; totals: Record<string, number>; target_per_letter: number }>("/admin/dataset/sessions");
export const deleteAdminDatasetSession = (id: string) => adminRequest<{ deleted: string }>(`/admin/dataset/sessions/${id}`, send("DELETE"));

export type AuditEntry = { id: number; admin: string; action: string; target_type: string; target_id: string; detail: Record<string, unknown>; created_at: string };
export const listAdminAudit = (p: { page: number; action?: string }) => {
  const q = new URLSearchParams({ page: String(p.page) });
  if (p.action) q.set("action", p.action);
  return adminRequest<{ page: number; page_size: number; total: number; items: AuditEntry[] }>(`/admin/audit-log?${q}`);
};
