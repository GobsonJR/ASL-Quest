import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { NavPage } from "./constants";
import {
  getLevel,
  getLevelProgress,
  pickPracticeLetter,
  recordCorrectSign,
  recordIncorrectAttempt,
  recordWordLetterCorrect,
  recordWordLetterIncorrect,
  applyWordCompletionXp,
  completeWordChallenge,
  resetSpeedChallenge,
  startSession,
} from "./gamification";
import { createInitialState, loadGameState, saveGameState } from "./storage";
import { flushProgressSync, logPracticeAttempt, scheduleProgressSync } from "./sync";
import type { GameState, PracticeOutcome } from "./types";
import { fetchProgress } from "../auth/api";
import { useAuth } from "../auth/AuthContext";

type ChallengeMode = "none" | "daily" | "alphabet" | "speed" | "word";
type PracticeKind = "letter" | "word";

type GameContextValue = {
  state: GameState;
  level: number;
  levelProgress: ReturnType<typeof getLevelProgress>;
  page: NavPage;
  practiceLetter: string | null;
  practiceKind: PracticeKind;
  practiceWordId: string | null;
  wordQueue: string[];
  challengeMode: ChallengeMode;
  ready: boolean;
  navigate: (page: NavPage) => void;
  startPractice: (letter?: string | null, mode?: ChallengeMode) => void;
  startWordPractice: (wordId: string, mode?: ChallengeMode, queue?: string[]) => void;
  beginSession: () => void;
  nextPracticeLetter: () => string;
  markCorrect: (letter: string, elapsedMs: number) => PracticeOutcome;
  markIncorrect: (letter: string, prediction?: string | null) => void;
  markWordLetterCorrect: (letter: string, elapsedMs: number) => PracticeOutcome;
  markWordLetterIncorrect: (letter: string, prediction?: string | null) => void;
  applyWordXp: (amount: number, badges: Array<{ id: string; name: string }>) => void;
  markWordChallengeComplete: () => void;
  restartSpeedChallenge: () => void;
  resetProgress: () => void;
  hydrateState: (state: GameState) => void;
};

const GameContext = createContext<GameContextValue | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const { user, setSyncStatus } = useAuth();
  const [state, setState] = useState<GameState>(() => createInitialState());
  const [page, setPage] = useState<NavPage>("home");
  const [practiceLetter, setPracticeLetter] = useState<string | null>(null);
  const [practiceKind, setPracticeKind] = useState<PracticeKind>("letter");
  const [practiceWordId, setPracticeWordId] = useState<string | null>(null);
  const [wordQueue, setWordQueue] = useState<string[]>([]);
  const [challengeMode, setChallengeMode] = useState<ChallengeMode>("none");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!user) {
      setState(createInitialState());
      setReady(false);
      return;
    }

    setReady(false);
    fetchProgress()
      .then((serverState) => {
        setState(serverState);
        saveGameState(serverState);
        setSyncStatus("synced");
      })
      .catch(() => {
        const cached = loadGameState();
        setState(cached);
        setSyncStatus("offline");
      })
      .finally(() => setReady(true));
  }, [user, setSyncStatus]);

  useEffect(() => {
    if (!user) return;
    saveGameState(state);
    scheduleProgressSync(state, setSyncStatus);
  }, [state, user, setSyncStatus]);

  const value = useMemo<GameContextValue>(() => {
    const level = getLevel(state.xp);
    const levelProgress = getLevelProgress(state.xp);

    return {
      state,
      level,
      levelProgress,
      page,
      practiceLetter,
      practiceKind,
      practiceWordId,
      wordQueue,
      challengeMode,
      ready,
      navigate: setPage,
      startPractice: (letter = null, mode = "none") => {
        setPracticeKind("letter");
        setPracticeWordId(null);
        setWordQueue([]);
        setPracticeLetter(letter);
        setChallengeMode(mode);
        setPage("practice");
      },
      startWordPractice: (wordId, mode = "none", queue = []) => {
        setPracticeKind("word");
        setPracticeWordId(wordId);
        setWordQueue(queue);
        setChallengeMode(mode);
        setPage("practice");
      },
      beginSession: () => setState((current) => startSession(current)),
      nextPracticeLetter: () => pickPracticeLetter(state, practiceLetter),
      markCorrect: (letter, elapsedMs) => {
        let outcome: PracticeOutcome = {
          xpGained: 0,
          leveledUp: false,
          newLevel: level,
          badgesUnlocked: [],
          dailyCompleted: false,
          message: "",
        };
        setState((current) => {
          const result = recordCorrectSign(current, letter, elapsedMs, challengeMode);
          outcome = result.outcome;
          return result.state;
        });
        if (user) {
          void logPracticeAttempt({
            letter,
            prediction: letter,
            correct: true,
            response_time: elapsedMs,
            xp_earned: outcome.xpGained,
            challenge_type: challengeMode === "none" ? null : challengeMode,
            reason: outcome.message || "correct_sign",
          });
        }
        return outcome;
      },
      markIncorrect: (letter, prediction = null) => {
        setState((current) => recordIncorrectAttempt(current, letter));
        if (user) {
          void logPracticeAttempt({
            letter,
            prediction,
            correct: false,
            challenge_type: challengeMode === "none" ? null : challengeMode,
          });
        }
      },
      markWordLetterCorrect: (letter, elapsedMs) => {
        let outcome: PracticeOutcome = {
          xpGained: 0,
          leveledUp: false,
          newLevel: level,
          badgesUnlocked: [],
          dailyCompleted: false,
          message: "",
        };
        setState((current) => {
          const result = recordWordLetterCorrect(current, letter, elapsedMs);
          outcome = result.outcome;
          return result.state;
        });
        if (user) {
          void logPracticeAttempt({
            letter,
            prediction: letter,
            correct: true,
            response_time: elapsedMs,
            xp_earned: outcome.xpGained,
            challenge_type: challengeMode === "word" ? "word" : challengeMode === "daily" ? "daily" : null,
            reason: "word_letter_correct",
          });
        }
        return outcome;
      },
      markWordLetterIncorrect: (letter, prediction = null) => {
        setState((current) => recordWordLetterIncorrect(current, letter));
        if (user) {
          void logPracticeAttempt({
            letter,
            prediction,
            correct: false,
            challenge_type: challengeMode === "word" ? "word" : null,
            reason: "word_letter_incorrect",
          });
        }
      },
      applyWordXp: (amount, badges) => {
        if (amount <= 0 && badges.length === 0) return;
        setState((current) => applyWordCompletionXp(current, amount, badges));
      },
      markWordChallengeComplete: () => setState((current) => completeWordChallenge(current)),
      restartSpeedChallenge: () => setState((current) => resetSpeedChallenge(current)),
      resetProgress: () => {
        const fresh = createInitialState();
        setState(fresh);
        if (user) void flushProgressSync(fresh, setSyncStatus);
      },
      hydrateState: (next) => setState(next),
    };
  }, [state, page, practiceLetter, practiceKind, practiceWordId, wordQueue, challengeMode, user, ready, setSyncStatus]);

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame() {
  const context = useContext(GameContext);
  if (!context) throw new Error("useGame must be used within GameProvider");
  return context;
}
