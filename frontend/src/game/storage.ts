import { DAILY_TARGET, LETTERS, STORAGE_KEY } from "./constants";
import type { GameState } from "./types";

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function emptyLetterStats(): Record<string, { correct: number; attempts: number }> {
  return Object.fromEntries(LETTERS.map((letter) => [letter, { correct: 0, attempts: 0 }]));
}

export function createInitialState(): GameState {
  return {
    xp: 0,
    streak: 0,
    longestStreak: 0,
    lastPracticeDate: null,
    totalCorrect: 0,
    totalAttempts: 0,
    sessions: 0,
    letterStats: emptyLetterStats(),
    unlockedBadges: [],
    dailyChallenge: {
      date: todayKey(),
      progress: 0,
      target: DAILY_TARGET,
      completed: false,
      mistakes: 0,
      perfectBonusAwarded: false,
    },
    challenges: {
      alphabetCompleted: false,
      speedCompleted: false,
      speedStartedAt: null,
      speedCount: 0,
      wordCompleted: false,
      wordCount: 0,
    },
    recentAchievements: [],
  };
}

function normalizeDailyChallenge(state: GameState): GameState {
  const today = todayKey();
  if (state.dailyChallenge.date === today) return state;
  return {
    ...state,
    dailyChallenge: {
      date: today,
      progress: 0,
      target: DAILY_TARGET,
      completed: false,
      mistakes: 0,
      perfectBonusAwarded: false,
    },
  };
}

export function loadGameState(): GameState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return createInitialState();
    const parsed = JSON.parse(raw) as GameState;
    const merged: GameState = {
      ...createInitialState(),
      ...parsed,
      letterStats: { ...emptyLetterStats(), ...parsed.letterStats },
      dailyChallenge: { ...createInitialState().dailyChallenge, ...parsed.dailyChallenge },
      challenges: { ...createInitialState().challenges, ...parsed.challenges },
      recentAchievements: parsed.recentAchievements ?? [],
      unlockedBadges: parsed.unlockedBadges ?? [],
    };
    return normalizeDailyChallenge(merged);
  } catch {
    return createInitialState();
  }
}

export function saveGameState(state: GameState): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}
