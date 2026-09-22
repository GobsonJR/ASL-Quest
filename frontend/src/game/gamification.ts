import {
  BADGES,
  DAILY_TARGET,
  LEARNING_CORRECT,
  LETTERS,
  MASTERED_CORRECT,
  SPEED_LIMIT_MS,
  SPEED_TARGET,
  XP_ALPHABET_CHALLENGE,
  XP_CORRECT,
  XP_DAILY_COMPLETE,
  XP_FAST_CORRECT,
  XP_PER_LEVEL,
  XP_PERFECT_DAILY,
  XP_SPEED_CHALLENGE,
} from "./constants";
import type { GameState, MasteryTier, PracticeOutcome } from "./types";

export function getLevel(xp: number): number {
  return Math.floor(xp / XP_PER_LEVEL) + 1;
}

export function getLevelProgress(xp: number): { current: number; next: number; percent: number } {
  const level = getLevel(xp);
  const floor = (level - 1) * XP_PER_LEVEL;
  const current = xp - floor;
  const next = XP_PER_LEVEL;
  return { current, next, percent: Math.min(100, (current / next) * 100) };
}

export type LetterStatus = "NEW" | "LEARNING" | "PRACTICED" | "MASTERED";

export function getMasteryTier(correct: number): MasteryTier {
  if (correct >= MASTERED_CORRECT) return "Mastered";
  if (correct >= LEARNING_CORRECT) return "Learning";
  return "Beginner";
}

export function getLetterStatus(correct: number): LetterStatus {
  if (correct >= MASTERED_CORRECT) return "MASTERED";
  if (correct >= LEARNING_CORRECT) return "PRACTICED";
  if (correct > 0) return "LEARNING";
  return "NEW";
}

export function getContinueLetter(state: GameState): string {
  const inProgress = LETTERS.map((letter) => ({
    letter,
    correct: state.letterStats[letter].correct,
  }))
    .filter((item) => item.correct > 0 && item.correct < MASTERED_CORRECT)
    .sort((a, b) => b.correct - a.correct);

  if (inProgress.length > 0) return inProgress[0].letter;

  const fresh = LETTERS.find((letter) => state.letterStats[letter].correct === 0);
  return fresh ?? LETTERS[0];
}

export function getBadgeProgress(
  state: GameState,
  badgeId: string
): { current: number; target: number; label: string } | null {
  switch (badgeId) {
    case "first_sign":
      return { current: Math.min(state.totalCorrect, 1), target: 1, label: "successful sign" };
    case "streak_7":
      return { current: Math.min(state.streak, 7), target: 7, label: "day streak" };
    case "alphabet_beginner":
      return { current: Math.min(lettersLearnedCount(state), 5), target: 5, label: "letters signed" };
    case "alphabet_explorer":
      return { current: Math.min(lettersLearnedCount(state), 13), target: 13, label: "letters signed" };
    case "alphabet_master":
      return { current: lettersLearnedCount(state), target: LETTERS.length, label: "letters signed" };
    case "speed_signer":
      return { current: state.challenges.speedCompleted ? 1 : 0, target: 1, label: "speed challenge" };
    case "daily_perfect":
      return {
        current: state.dailyChallenge.perfectBonusAwarded ? 1 : 0,
        target: 1,
        label: "perfect daily challenge",
      };
    default:
      return null;
  }
}

export function getMasteryPercent(correct: number): number {
  return Math.min(100, (correct / MASTERED_CORRECT) * 100);
}

export function lettersLearnedCount(state: GameState): number {
  return LETTERS.filter((letter) => state.letterStats[letter].correct > 0).length;
}

export function lettersMasteredCount(state: GameState): number {
  return LETTERS.filter((letter) => state.letterStats[letter].correct >= MASTERED_CORRECT).length;
}

export function accuracyPercent(state: GameState): number {
  if (state.totalAttempts === 0) return 0;
  return Math.round((state.totalCorrect / state.totalAttempts) * 100);
}

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function updateStreak(state: GameState): GameState {
  const today = todayKey();
  if (state.lastPracticeDate === today) return state;

  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayKey = yesterday.toISOString().slice(0, 10);

  let streak = 1;
  if (state.lastPracticeDate === yesterdayKey) {
    streak = state.streak + 1;
  }

  return {
    ...state,
    streak,
    longestStreak: Math.max(state.longestStreak, streak),
    lastPracticeDate: today,
  };
}

function unlockBadge(state: GameState, badgeId: string, title: string): GameState {
  if (state.unlockedBadges.includes(badgeId)) return state;
  return {
    ...state,
    unlockedBadges: [...state.unlockedBadges, badgeId],
    recentAchievements: [{ id: badgeId, title, unlockedAt: new Date().toISOString() }, ...state.recentAchievements].slice(0, 8),
  };
}

function evaluateBadges(state: GameState): GameState {
  let next = state;
  if (next.totalCorrect >= 1) {
    next = unlockBadge(next, "first_sign", "First Sign");
  }
  if (next.streak >= 7) {
    next = unlockBadge(next, "streak_7", "7 Day Streak");
  }
  if (lettersLearnedCount(next) >= 5) {
    next = unlockBadge(next, "alphabet_beginner", "Alphabet Beginner");
  }
  if (lettersLearnedCount(next) >= 13) {
    next = unlockBadge(next, "alphabet_explorer", "Alphabet Explorer");
  }
  if (lettersLearnedCount(next) >= LETTERS.length) {
    next = unlockBadge(next, "alphabet_master", "Alphabet Master");
  }
  if (next.challenges.speedCompleted) {
    next = unlockBadge(next, "speed_signer", "Speed Signer");
  }
  if (next.dailyChallenge.perfectBonusAwarded) {
    next = unlockBadge(next, "daily_perfect", "Perfect Practice");
  }
  return next;
}

export function pickPracticeLetter(state: GameState, preferred?: string | null): string {
  if (preferred && LETTERS.includes(preferred)) return preferred;
  const unmastered = LETTERS.filter((letter) => state.letterStats[letter].correct < MASTERED_CORRECT);
  const pool = unmastered.length > 0 ? unmastered : LETTERS;
  return pool[Math.floor(Math.random() * pool.length)];
}

export function recordCorrectSign(
  state: GameState,
  letter: string,
  elapsedMs: number,
  challengeMode: "none" | "daily" | "alphabet" | "speed" | "word" = "none"
): { state: GameState; outcome: PracticeOutcome } {
  let next = updateStreak({ ...state });
  const previousLevel = getLevel(next.xp);

  let xpGained = elapsedMs <= 5000 ? XP_FAST_CORRECT : XP_CORRECT;
  const stats = next.letterStats[letter] ?? { correct: 0, attempts: 0 };
  next = {
    ...next,
    xp: next.xp + xpGained,
    totalCorrect: next.totalCorrect + 1,
    totalAttempts: next.totalAttempts + 1,
    letterStats: {
      ...next.letterStats,
      [letter]: { correct: stats.correct + 1, attempts: stats.attempts + 1 },
    },
  };

  if (!next.dailyChallenge.completed) {
    next = {
      ...next,
      dailyChallenge: {
        ...next.dailyChallenge,
        progress: next.dailyChallenge.progress + 1,
      },
    };
    if (next.dailyChallenge.progress >= DAILY_TARGET) {
      next = {
        ...next,
        dailyChallenge: { ...next.dailyChallenge, completed: true },
        xp: next.xp + XP_DAILY_COMPLETE,
      };
      xpGained += XP_DAILY_COMPLETE;
    }
  }

  if (challengeMode === "alphabet" && lettersLearnedCount(next) >= LETTERS.length && !next.challenges.alphabetCompleted) {
    next = {
      ...next,
      challenges: { ...next.challenges, alphabetCompleted: true },
      xp: next.xp + XP_ALPHABET_CHALLENGE,
    };
    xpGained += XP_ALPHABET_CHALLENGE;
  }

  if (challengeMode === "speed") {
    const startedAt = next.challenges.speedStartedAt ?? Date.now();
    const speedCount = next.challenges.speedCount + 1;
    next = {
      ...next,
      challenges: {
        ...next.challenges,
        speedStartedAt: startedAt,
        speedCount,
      },
    };
    if (speedCount >= SPEED_TARGET && Date.now() - startedAt <= SPEED_LIMIT_MS && !next.challenges.speedCompleted) {
      next = {
        ...next,
        challenges: { ...next.challenges, speedCompleted: true },
        xp: next.xp + XP_SPEED_CHALLENGE,
      };
      xpGained += XP_SPEED_CHALLENGE;
    }
  }

  if (next.dailyChallenge.completed && next.dailyChallenge.mistakes === 0 && !next.dailyChallenge.perfectBonusAwarded) {
    next = {
      ...next,
      dailyChallenge: { ...next.dailyChallenge, perfectBonusAwarded: true },
      xp: next.xp + XP_PERFECT_DAILY,
    };
    xpGained += XP_PERFECT_DAILY;
  }

  next = evaluateBadges(next);
  const newLevel = getLevel(next.xp);
  const badgesUnlocked = next.unlockedBadges.filter((id) => !state.unlockedBadges.includes(id));

  return {
    state: next,
    outcome: {
      xpGained,
      leveledUp: newLevel > previousLevel,
      newLevel,
      badgesUnlocked,
      dailyCompleted: next.dailyChallenge.completed && !state.dailyChallenge.completed,
      message: elapsedMs <= 5000 ? "Quick and clear!" : "Great job!",
    },
  };
}

export function recordIncorrectAttempt(state: GameState, letter: string): GameState {
  let next = updateStreak({ ...state, totalAttempts: state.totalAttempts + 1 });
  const stats = next.letterStats[letter] ?? { correct: 0, attempts: 0 };
  next = {
    ...next,
    letterStats: {
      ...next.letterStats,
      [letter]: { ...stats, attempts: stats.attempts + 1 },
    },
    dailyChallenge: {
      ...next.dailyChallenge,
      mistakes: next.dailyChallenge.mistakes + 1,
    },
  };
  return next;
}

export function startSession(state: GameState): GameState {
  return { ...state, sessions: state.sessions + 1 };
}

export function resetSpeedChallenge(state: GameState): GameState {
  return {
    ...state,
    challenges: {
      ...state.challenges,
      speedStartedAt: Date.now(),
      speedCount: 0,
      speedCompleted: false,
    },
  };
}

export function recordWordLetterCorrect(
  state: GameState,
  letter: string,
  elapsedMs: number
): { state: GameState; outcome: PracticeOutcome } {
  let next = updateStreak({ ...state });
  const previousLevel = getLevel(next.xp);
  const xpGained = elapsedMs <= 5000 ? XP_FAST_CORRECT : XP_CORRECT;
  const stats = next.letterStats[letter] ?? { correct: 0, attempts: 0 };
  next = {
    ...next,
    xp: next.xp + xpGained,
    totalCorrect: next.totalCorrect + 1,
    totalAttempts: next.totalAttempts + 1,
    letterStats: {
      ...next.letterStats,
      [letter]: { correct: stats.correct + 1, attempts: stats.attempts + 1 },
    },
  };
  next = evaluateBadges(next);
  const newLevel = getLevel(next.xp);
  return {
    state: next,
    outcome: {
      xpGained,
      leveledUp: newLevel > previousLevel,
      newLevel,
      badgesUnlocked: next.unlockedBadges.filter((id) => !state.unlockedBadges.includes(id)),
      dailyCompleted: false,
      message: elapsedMs <= 5000 ? "Clear and quick!" : "Nice letter!",
    },
  };
}

export function recordWordLetterIncorrect(state: GameState, letter: string): GameState {
  let next = updateStreak({ ...state, totalAttempts: state.totalAttempts + 1 });
  const stats = next.letterStats[letter] ?? { correct: 0, attempts: 0 };
  return {
    ...next,
    letterStats: {
      ...next.letterStats,
      [letter]: { ...stats, attempts: stats.attempts + 1 },
    },
  };
}

export function applyWordCompletionXp(state: GameState, amount: number, badgeNames: Array<{ id: string; name: string }>): GameState {
  let next = { ...state, xp: state.xp + amount };
  for (const badge of badgeNames) {
    next = unlockBadge(next, badge.id, badge.name);
  }
  return evaluateBadges(next);
}

export function completeWordChallenge(state: GameState): GameState {
  return {
    ...state,
    challenges: {
      ...state.challenges,
      wordCompleted: true,
      wordCount: state.challenges.wordCount + 1,
    },
  };
}

export function badgeById(id: string) {
  return BADGES.find((badge) => badge.id === id);
}
