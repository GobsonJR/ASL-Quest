export type SignoMood =
  | "idle"
  | "curious"
  | "listening"
  | "thinking"
  | "happy"
  | "encouraging"
  | "celebrating"
  | "sleepy"
  | "explaining";

export type SignoReaction = "bounce" | "wink" | "squish" | "sway";

export type SignoAccessory = "headband" | "glasses" | "crown";

/** Real app events that other screens report to SIGNO. Streaks, badges, level-ups and
 * completed letters are not listed here: SIGNO detects those itself by watching the
 * actual progress state, so they can never be invented or double-reported. */
export type SignoEvent =
  | { type: "letter_selected"; letter: string }
  | { type: "lesson_opened"; letter: string }
  | { type: "practice_started"; letter: string }
  | { type: "sign_correct"; letter: string; xp: number }
  | { type: "sign_incorrect"; letter: string; predicted: string | null }
  | { type: "hint_requested"; letter: string }
  | { type: "word_complete"; word: string; xp: number };

export type SignoBubble = {
  id: number;
  text: string;
  kind: "tip" | "celebrate" | "info";
};

/** Where a chat message came from, so local reactions are never mistaken for AI answers. */
export type SignoMessageOrigin = "user" | "ai" | "local";

export type SignoChatMessage = {
  id: number;
  origin: SignoMessageOrigin;
  text: string;
  /** For AI replies: the assistant's own source label ("llm" or "knowledge_base"). */
  source?: "llm" | "knowledge_base" | null;
  /** True when the failed request can be retried. */
  failed?: boolean;
};
