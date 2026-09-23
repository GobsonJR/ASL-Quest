import { authHeaders } from "../auth/token";

const API = "/api";

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
