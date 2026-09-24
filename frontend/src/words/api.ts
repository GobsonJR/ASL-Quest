import { authHeaders } from "../auth/token";
import { parseError } from "../shared/httpError";

const API = "/api";

export type WordLetterBreakdown = {
  letter: string;
  correct: number;
  attempts: number;
  accuracy: number | null;
  practiced: boolean;
};

export type WordProgress = {
  word_id: string;
  word: string;
  category?: string;
  difficulty?: string;
  attempts: number;
  completions: number;
  correct_letters: number;
  total_letters: number;
  accuracy: number | null;
  mastery: number;
  mastered: boolean;
  best_time_ms: number | null;
  best_time_sec: number | null;
  total_time_ms: number;
  resume_index: number;
  last_practiced_at: string | null;
  letter_breakdown: WordLetterBreakdown[];
};

export type WordItem = {
  id: string;
  word: string;
  category: string;
  difficulty: string;
  description: string;
  letters: string[];
  letter_count: number;
  estimated_xp: number;
  tip: string | null;
  progress: WordProgress | null;
};

export type WordSessionResult = {
  id: number;
  word_id: string;
  word: string;
  completed: boolean;
  accuracy: number | null;
  duration_ms: number | null;
  xp_earned: number;
  xp_events: Array<{ reason: string; amount: number }>;
  new_achievements: Array<{ id: string; name: string }>;
  progress: WordProgress;
};

export type WordAnalytics = {
  words_learned: number;
  words_mastered: number;
  words_completed: number;
  word_sessions: number;
  word_accuracy: number | null;
  word_practice_time_sec: number;
  most_practiced: WordProgress[];
  strongest: WordProgress[];
  needing_practice: WordProgress[];
  category_progress: Array<{ category: string; completed: number; total: number; percent: number }>;
  has_data: boolean;
};

export type WordRecommendations = {
  has_data: boolean;
  recommended: Array<{ word_id: string; word: string; category: string; reason: string; message: string }>;
  continue_word: WordProgress | null;
  message: string | null;
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) throw new Error(await parseError(response, "Word request failed."));
  return response.json();
}

export function fetchWords(category?: string) {
  const query = category && category !== "All" ? `?category=${encodeURIComponent(category)}` : "";
  return request<{ items: WordItem[]; categories: string[] }>(`/words${query}`);
}

export function fetchWord(wordId: string) {
  return request<WordItem>(`/words/${wordId}`);
}

export function fetchWordProgress(wordId: string) {
  return request<WordProgress>(`/words/${wordId}/progress`);
}

export function fetchWordAnalytics() {
  return request<WordAnalytics>("/words/analytics");
}

export function fetchWordRecommendations() {
  return request<WordRecommendations>("/words/recommendations");
}

export function fetchWordChallenge(mode: "word" | "daily" = "word") {
  return request<{ mode: string; items: WordItem[] }>(`/words/challenge?mode=${mode}`);
}

export function recordWordSession(input: {
  word_id: string;
  letters: Array<{ letter: string; prediction: string | null; correct: boolean; response_time?: number | null }>;
  completed: boolean;
  duration_ms?: number | null;
  mistakes?: number;
  challenge_type?: string | null;
  challenge_finished?: boolean;
  resume_index?: number;
}) {
  return request<WordSessionResult>("/words/practice/session", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function fetchAdminWords(): Promise<{
  total_word_practice_sessions: number;
  words_completed: number;
  average_word_accuracy: number | null;
  most_practiced: Array<{ word: string; sessions: number; accuracy: number | null }>;
  hardest: Array<{ word: string; sessions: number; accuracy: number | null }>;
  most_popular_categories: Array<{ category: string; sessions: number }>;
}> {
  return request("/admin/words");
}
