import type { GameState } from "../game/types";
import { authHeaders } from "../auth/token";

const API = "/api";
export type AnalyticsRange = "7d" | "30d" | "90d" | "12m" | "all";
export type HeatmapRange = "3m" | "6m" | "12m";

export type AnalyticsDashboard = {
  overview: {
    total_practice_sessions: number;
    total_attempts: number;
    total_correct: number;
    overall_accuracy: number | null;
    letters_practiced: number;
    letters_mastered: number;
    current_streak: number;
    best_streak: number;
    total_xp: number;
    average_response_sec: number | null;
    has_data: boolean;
  };
  heatmap: { range: HeatmapRange; cells: HeatmapCell[] };
  activity_trend: TrendSeries;
  accuracy_trend: AccuracyTrend;
  letters: LetterAnalytics[];
  insights: LetterInsights;
  response_time: ResponseTimeAnalytics;
  xp: XpAnalytics;
  streak: StreakAnalytics;
  funnel: LearningFunnel;
  weekly_summary: WeeklySummary;
  progress: GameState;
  words?: WordAnalyticsSummary;
};

export type WordAnalyticsSummary = {
  words_learned: number;
  words_mastered: number;
  words_completed: number;
  word_accuracy: number | null;
  word_practice_time_sec: number;
  most_practiced: Array<{ word: string; word_id: string; attempts: number; accuracy: number | null; mastery: number }>;
  strongest: Array<{ word: string; word_id: string; accuracy: number | null }>;
  needing_practice: Array<{ word: string; word_id: string; accuracy: number | null }>;
  category_progress: Array<{ category: string; completed: number; total: number; percent: number }>;
  has_data: boolean;
};

export type HeatmapCell = {
  date: string;
  attempts: number;
  correct: number;
  accuracy: number | null;
  xp: number;
  intensity: number;
};

export type TrendPoint = { date: string; value: number | null };
export type TrendSeries = { range: AnalyticsRange; points: Array<{ date: string; value: number }>; has_data: boolean };
export type AccuracyTrend = TrendSeries & {
  current_accuracy: number | null;
  previous_accuracy: number | null;
  delta: number | null;
};

export type LetterAnalytics = {
  letter: string;
  attempts: number;
  correct: number;
  accuracy: number | null;
  mastery_percent: number;
  status: string;
  practiced: boolean;
};

export type LetterInsights = {
  strongest: LetterAnalytics[];
  weakest: LetterAnalytics[];
  most_practiced: LetterAnalytics[];
  least_practiced: LetterAnalytics[];
};

export type ResponseTimeAnalytics = {
  average_sec: number | null;
  fastest_sec: number | null;
  trend: Array<{ date: string; seconds: number }>;
  has_data: boolean;
};

export type XpAnalytics = {
  total_xp: number;
  xp_this_week: number;
  xp_this_month: number;
  trend: Array<{ date: string; value: number }>;
  has_data: boolean;
};

export type StreakAnalytics = {
  current_streak: number;
  best_streak: number;
  active_days: number;
  daily_activity: Array<{ date: string; attempts: number }>;
};

export type LearningFunnel = {
  total_letters: number;
  practiced: number;
  learning: number;
  practiced_stage: number;
  proficient: number;
  mastered: number;
  counts: Record<string, number>;
};

export type WeeklySummary = {
  current_week: {
    attempts: number;
    accuracy: number | null;
    xp: number;
    letters_practiced: number;
    average_response_sec: number | null;
  };
  previous_week: {
    attempts: number;
    accuracy: number | null;
    xp: number;
    letters_practiced: number;
    average_response_sec: number | null;
  };
  delta: {
    attempts_pct: number | null;
    accuracy_points: number | null;
    xp_pct: number | null;
  };
  has_comparison: boolean;
};

export type PracticeHistoryPage = {
  page: number;
  page_size: number;
  total: number;
  items: Array<{
    id: number;
    date: string;
    letter: string;
    prediction: string | null;
    correct: boolean;
    response_time_sec: number | null;
    xp_earned: number;
  }>;
};

export type LetterDetail = {
  letter: string;
  mastery_percent: number;
  status: string;
  accuracy: number | null;
  attempts: number;
  correct: number;
  average_response_sec: number | null;
  first_practiced: string | null;
  last_practiced: string | null;
  accuracy_trend: TrendPoint[];
  history: PracticeHistoryPage["items"];
  practiced: boolean;
};

async function request<T>(path: string): Promise<T> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const tokenHeaders = authHeaders();
  if (tokenHeaders.Authorization) headers.set("Authorization", tokenHeaders.Authorization);
  const response = await fetch(`${API}${path}`, { headers });
  if (!response.ok) throw new Error("Failed to load analytics.");
  return response.json();
}

export function fetchAnalyticsDashboard(range: AnalyticsRange, heatmapRange: HeatmapRange) {
  return request<AnalyticsDashboard>(`/analytics/dashboard?range=${range}&heatmap_range=${heatmapRange}`);
}

export function fetchPracticeHistory(params: {
  page?: number;
  letter?: string;
  result?: "correct" | "incorrect";
}) {
  const query = new URLSearchParams();
  if (params.page) query.set("page", String(params.page));
  if (params.letter) query.set("letter", params.letter);
  if (params.result) query.set("result", params.result);
  return request<PracticeHistoryPage>(`/analytics/history?${query.toString()}`);
}

export function fetchLetterDetail(letter: string) {
  return request<LetterDetail>(`/analytics/letters/${letter}`);
}
