export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export const STORAGE_KEY = "asl-learning-progress-v1";

export const XP_PER_LEVEL = 1000;
export const XP_CORRECT = 20;
export const XP_FAST_CORRECT = 30;
export const XP_DAILY_COMPLETE = 100;
export const XP_PERFECT_DAILY = 150;
export const XP_ALPHABET_CHALLENGE = 500;
export const XP_SPEED_CHALLENGE = 150;
export const XP_WORD_COMPLETED = 50;
export const XP_WORD_PERFECT = 25;
export const XP_WORD_FAST = 20;
export const XP_WORD_CHALLENGE = 150;

export const FAST_ANSWER_MS = 5000;
export const DAILY_TARGET = 10;
export const SPEED_TARGET = 5;
export const SPEED_LIMIT_MS = 45000;
export const MASTERED_CORRECT = 10;
export const LEARNING_CORRECT = 3;

export type NavPage =
  | "home"
  | "learn"
  | "words"
  | "practice"
  | "challenges"
  | "progress"
  | "achievements"
  | "profile"
  | "settings"
  | "admin";

export type BadgeDefinition = {
  id: string;
  icon: string;
  title: string;
  description: string;
};

export const BADGES: BadgeDefinition[] = [
  { id: "first_sign", icon: "🏅", title: "First Sign", description: "Complete your first successful sign" },
  { id: "streak_7", icon: "🔥", title: "7 Day Streak", description: "Practice for 7 consecutive days" },
  { id: "alphabet_beginner", icon: "🔤", title: "Alphabet Beginner", description: "Practice 5 unique letters" },
  { id: "alphabet_explorer", icon: "🧭", title: "Alphabet Explorer", description: "Practice 13 unique letters" },
  { id: "alphabet_master", icon: "🏆", title: "Alphabet Master", description: "Master all 26 letters" },
  { id: "speed_signer", icon: "⚡", title: "Speed Signer", description: "Complete a speed challenge quickly" },
  { id: "fast_signer", icon: "💨", title: "Fast Signer", description: "Complete 5 fast correct signs" },
  { id: "daily_perfect", icon: "✨", title: "Perfect Practice", description: "Complete a daily challenge without mistakes" },
  { id: "word_starter", icon: "📝", title: "Word Starter", description: "Complete your first word" },
  { id: "word_learner", icon: "📚", title: "Word Learner", description: "Complete 5 different words" },
  { id: "word_explorer", icon: "🌍", title: "Word Explorer", description: "Complete words from 3 categories" },
  { id: "word_master", icon: "👑", title: "Word Master", description: "Master 10 words" },
  { id: "vocabulary_builder", icon: "🧠", title: "Vocabulary Builder", description: "Complete 25 words" },
];

export type ChallengeDefinition = {
  id: "daily" | "alphabet" | "speed" | "word";
  title: string;
  description: string;
  reward: number;
  icon: string;
};

export const CHALLENGES: ChallengeDefinition[] = [
  {
    id: "daily",
    title: "Daily Challenge",
    description: "Sign 10 letters correctly today",
    reward: XP_DAILY_COMPLETE,
    icon: "☀️",
  },
  {
    id: "alphabet",
    title: "Alphabet Challenge",
    description: "Sign every letter A-Z at least once",
    reward: XP_ALPHABET_CHALLENGE,
    icon: "🔤",
  },
  {
    id: "speed",
    title: "Speed Challenge",
    description: "Correctly sign 5 letters quickly",
    reward: XP_SPEED_CHALLENGE,
    icon: "⚡",
  },
  {
    id: "word",
    title: "Word Challenge",
    description: "Spell 5 words using sequential alphabet signs",
    reward: XP_WORD_CHALLENGE,
    icon: "📝",
  },
];
