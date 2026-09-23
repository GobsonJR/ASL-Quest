export type LetterStats = {
  correct: number;
  attempts: number;
};

export type DailyChallengeState = {
  date: string;
  progress: number;
  target: number;
  completed: boolean;
  mistakes: number;
  perfectBonusAwarded: boolean;
};

export type ChallengeProgress = {
  alphabetCompleted: boolean;
  speedCompleted: boolean;
  speedStartedAt: number | null;
  speedCount: number;
  wordCompleted: boolean;
  wordCount: number;
};

export type RecentAchievement = {
  id: string;
  title: string;
  unlockedAt: string;
};

export type GameState = {
  xp: number;
  streak: number;
  longestStreak: number;
  lastPracticeDate: string | null;
  totalCorrect: number;
  totalAttempts: number;
  sessions: number;
  letterStats: Record<string, LetterStats>;
  unlockedBadges: string[];
  dailyChallenge: DailyChallengeState;
  challenges: ChallengeProgress;
  recentAchievements: RecentAchievement[];
};

export type MasteryTier = "Beginner" | "Learning" | "Mastered";

// Duolingo-style sequential view over the existing per-letter mastery data
// (see getLearningPathNodes in gamification.ts) -- a UI derivation only, not
// a new piece of progress state.
export type LearningPathNodeStatus = "completed" | "current" | "locked";

export type LearningPathNode = {
  letter: string;
  status: LearningPathNodeStatus;
  correct: number;
  masteryPercent: number;
};

export type PracticeOutcome = {
  xpGained: number;
  leveledUp: boolean;
  newLevel: number;
  badgesUnlocked: string[];
  dailyCompleted: boolean;
  message: string;
};
