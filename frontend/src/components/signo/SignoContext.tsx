import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { BADGES, type NavPage } from "../../game/constants";
import { useGame } from "../../game/GameContext";
import { getLearningPathNodes, getLevel, lettersMasteredCount } from "../../game/gamification";
import { SIGN_HINTS } from "../../pages/alphabetFeedback";
import { askSigno } from "../../services/signoService";
import type {
  SignoAccessory,
  SignoBubble,
  SignoChatMessage,
  SignoEvent,
  SignoMood,
  SignoReaction,
} from "./types";

// ---------------------------------------------------------------------------
// Preferences: only harmless UI choices are persisted (never conversations,
// tokens or credentials).
// ---------------------------------------------------------------------------
const PREFS_KEY = "signo-ui-v1";
export type SignoPrefs = { minimized: boolean; effects: boolean };

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function loadPrefs(): SignoPrefs {
  const fallback: SignoPrefs = { minimized: false, effects: !prefersReducedMotion() };
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<SignoPrefs>;
    return {
      minimized: typeof parsed.minimized === "boolean" ? parsed.minimized : fallback.minimized,
      effects: typeof parsed.effects === "boolean" ? parsed.effects : fallback.effects,
    };
  } catch {
    return fallback;
  }
}

function savePrefs(prefs: SignoPrefs): void {
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Storage can be unavailable (private mode); preferences just won't persist.
  }
}

// ---------------------------------------------------------------------------
// Timing constants
// ---------------------------------------------------------------------------
const BUBBLE_MS = 8000;
const SLEEP_AFTER_MS = 90_000;
const AWAY_WELCOME_MS = 10 * 60_000;
const DUPLICATE_WINDOW_MS = 3000;
const PRIORITY_WINDOW_MS = 1500;
const REACTIONS: SignoReaction[] = ["bounce", "wink", "squish", "sway"];

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
export type SignoContextValue = {
  page: NavPage;
  mood: SignoMood;
  pulse: number;
  reaction: SignoReaction | null;
  bubble: SignoBubble | null;
  hasTip: boolean;
  announcement: string;
  open: boolean;
  prefs: SignoPrefs;
  messages: SignoChatMessage[];
  busy: boolean;
  suggestions: string[];
  accessories: SignoAccessory[];
  currentLetter: string | null;
  canHint: boolean;
  emit: (event: SignoEvent) => void;
  setOpen: (open: boolean) => void;
  setPrefs: (patch: Partial<SignoPrefs>) => void;
  setTyping: (typing: boolean) => void;
  dismissBubble: () => void;
  react: () => void;
  ask: (question: string) => Promise<void>;
  retry: () => void;
  hint: () => void;
  celebrateProgress: () => void;
};

const SignoContext = createContext<SignoContextValue | null>(null);

/** Hook for the companion UI. Throws outside <SignoProvider>. */
export function useSigno(): SignoContextValue {
  const ctx = useContext(SignoContext);
  if (!ctx) throw new Error("useSigno must be used within SignoProvider");
  return ctx;
}

const NOOP_EMIT = () => {};

/** Hook for screens that only report real events. Safe (a no-op) without a provider. */
export function useSignoEmit(): (event: SignoEvent) => void {
  const ctx = useContext(SignoContext);
  return ctx?.emit ?? NOOP_EMIT;
}

// ---------------------------------------------------------------------------
// Copy -- deterministic and clearly local; AI answers never come from here.
// ---------------------------------------------------------------------------
function letterTip(letter: string): string {
  const hint = SIGN_HINTS[letter];
  return hint ? `${letter}: ${hint}` : `Take a look at the ${letter} handshape.`;
}

function pageTips(page: NavPage, ctx: { nextLetter: string | null; allDone: boolean; badges: number }): string[] {
  switch (page) {
    case "learn":
      if (ctx.allDone) return ["You've completed the whole alphabet. Revisit any letter to keep it sharp."];
      return ctx.nextLetter
        ? [`Welcome back! Up next: letter ${ctx.nextLetter}.`, `Ready when you are. Letter ${ctx.nextLetter} is next on your path.`]
        : ["Pick a letter from your path to keep going."];
    case "alphabet":
      return ["Pick a letter to see its handshape and a practice tip."];
    case "words":
      return ["Choose a word and spell it one letter at a time."];
    case "achievements":
      return [`${ctx.badges} of ${BADGES.length} badges earned so far.`];
    case "native":
      return ["Native signs are whole words, not single letters."];
    case "profile":
      return ["This is your learning summary."];
    default:
      return [];
  }
}

function suggestionsFor(
  page: NavPage,
  letter: string | null,
  mastered: number,
  nextLetter: string | null
): string[] {
  if (page === "practice" && letter) {
    return [
      `Give me a hint for signing the letter ${letter}.`,
      "Why might my sign not be recognized by the camera?",
      "How do I improve my hand positioning?",
    ];
  }
  if (page === "alphabet" && letter) {
    return [`Explain how to sign the letter ${letter}.`, `Why might my ${letter} sign not be recognized?`, "How do I improve my hand positioning?"];
  }
  if (page === "words") {
    return ["Help me practice spelling a word with fingerspelling.", "What should I learn next?"];
  }
  if (page === "achievements") return ["Celebrate my progress!", "How are badges unlocked?"];
  if (page === "learn") {
    return [
      nextLetter ? `What should I learn next? I've completed ${mastered} of 26 letters and my next letter is ${nextLetter}.` : "What should I learn next?",
      "How do I improve my hand positioning?",
      "How is XP awarded?",
    ];
  }
  return ["What should I learn next?", "How does A-Z recognition work?"];
}

function nowMs(): number {
  return Date.now();
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------
export function SignoProvider({ children }: { children: ReactNode }) {
  const { page, state, ready } = useGame();

  const [prefs, setPrefsState] = useState<SignoPrefs>(loadPrefs);
  const [open, setOpenState] = useState(false);
  const [transient, setTransientState] = useState<{ mood: SignoMood } | null>(null);
  const [sleepy, setSleepy] = useState(false);
  const [typing, setTyping] = useState(false);
  const [practiceActive, setPracticeActive] = useState(false);
  const [currentLetter, setCurrentLetter] = useState<string | null>(null);
  const [bubble, setBubble] = useState<SignoBubble | null>(null);
  const [hasTip, setHasTip] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [pulse, setPulse] = useState(0);
  const [reaction, setReaction] = useState<SignoReaction | null>(null);
  const [messages, setMessages] = useState<SignoChatMessage[]>([]);
  const [busy, setBusy] = useState(false);

  const prefsRef = useRef(prefs);
  prefsRef.current = prefs;
  const stateRef = useRef(state);
  stateRef.current = state;

  const idRef = useRef(1);
  const bubbleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transientTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reactionTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastBubble = useRef<{ text: string; at: number }>({ text: "", at: 0 });
  const lastCelebration = useRef<{ priority: number; at: number }>({ priority: 0, at: 0 });
  const conversationId = useRef<number | null>(null);
  const lastQuestion = useRef<string | null>(null);
  const mounted = useRef(true);
  const reactionIndex = useRef(0);
  const visits = useRef<Partial<Record<NavPage, number>>>({});

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      for (const timer of [bubbleTimer, transientTimer, reactionTimer]) {
        if (timer.current) clearTimeout(timer.current);
      }
    };
  }, []);

  const nodes = useMemo(() => getLearningPathNodes(state), [state]);
  const nextLetter = nodes.find((n) => n.status === "current")?.letter ?? null;
  const mastered = lettersMasteredCount(state);

  // -- low-level helpers ----------------------------------------------------
  const setMoodFor = useCallback((mood: SignoMood, ms: number) => {
    if (transientTimer.current) clearTimeout(transientTimer.current);
    setTransientState({ mood });
    setPulse((n) => n + 1);
    transientTimer.current = setTimeout(() => setTransientState(null), ms);
  }, []);

  const showBubble = useCallback(
    (text: string, kind: SignoBubble["kind"], opts: { announce?: boolean; force?: boolean } = {}) => {
      // Minimized means "stop volunteering tips" -- but a hint the user asked for always shows.
      if (prefsRef.current.minimized && !opts.force) return;
      const t = nowMs();
      if (lastBubble.current.text === text && t - lastBubble.current.at < DUPLICATE_WINDOW_MS) return;
      lastBubble.current = { text, at: t };
      setBubble({ id: idRef.current++, text, kind });
      if (kind === "tip") setHasTip(true);
      if (opts.announce) setAnnouncement(text);
      if (bubbleTimer.current) clearTimeout(bubbleTimer.current);
      bubbleTimer.current = setTimeout(() => setBubble(null), BUBBLE_MS);
    },
    []
  );

  /** A reaction to a genuine achievement. Lower-priority events that arrive
   * right after a higher-priority one are dropped, so one action never produces a pile-up. */
  const celebrate = useCallback(
    (priority: number, text: string, mood: SignoMood = "celebrating") => {
      const t = nowMs();
      if (t - lastCelebration.current.at < PRIORITY_WINDOW_MS && priority < lastCelebration.current.priority) return;
      lastCelebration.current = { priority, at: t };
      showBubble(text, "celebrate", { announce: true });
      setMoodFor(mood, 3200);
    },
    [showBubble, setMoodFor]
  );

  // -- preferences / panel --------------------------------------------------
  const setPrefs = useCallback((patch: Partial<SignoPrefs>) => {
    setPrefsState((prev) => {
      const next = { ...prev, ...patch };
      savePrefs(next);
      return next;
    });
    if (patch.minimized) {
      setBubble(null);
      setOpenState(false);
    }
  }, []);

  const setOpen = useCallback((next: boolean) => {
    setOpenState(next);
    if (next) {
      setHasTip(false);
      setBubble(null);
      // Opening the panel from the minimized launcher restores the full companion.
      setPrefsState((prev) => {
        if (!prev.minimized) return prev;
        const updated = { ...prev, minimized: false };
        savePrefs(updated);
        return updated;
      });
    }
  }, []);

  const dismissBubble = useCallback(() => setBubble(null), []);

  const react = useCallback(() => {
    const next = REACTIONS[reactionIndex.current % REACTIONS.length];
    reactionIndex.current += 1;
    setReaction(next);
    setPulse((n) => n + 1);
    if (reactionTimer.current) clearTimeout(reactionTimer.current);
    reactionTimer.current = setTimeout(() => setReaction(null), 900);
    if (next === "bounce" || next === "wink") setMoodFor("happy", 1400);
  }, [setMoodFor]);

  const pushMessage = useCallback((message: Omit<SignoChatMessage, "id">) => {
    setMessages((prev) => [...prev, { ...message, id: idRef.current++ }]);
  }, []);

  const giveHint = useCallback(
    (letter: string) => {
      setCurrentLetter(letter);
      const text = `Hint: ${SIGN_HINTS[letter] ?? `compare your hand with the ${letter} reference.`}`;
      pushMessage({ origin: "local", text });
      showBubble(text, "info", { announce: true, force: true });
      setMoodFor("explaining", 5000);
    },
    [pushMessage, showBubble, setMoodFor]
  );

  // -- real events from screens ---------------------------------------------
  const emit = useCallback(
    (event: SignoEvent) => {
      switch (event.type) {
        case "letter_selected":
          setCurrentLetter(event.letter);
          showBubble(letterTip(event.letter), "tip");
          setMoodFor("explaining", 3500);
          break;
        case "lesson_opened":
          setCurrentLetter(event.letter);
          setPracticeActive(false);
          showBubble(`New lesson: letter ${event.letter}. Look over the reference sign before you start.`, "tip");
          setMoodFor("curious", 3500);
          break;
        case "practice_started":
          setCurrentLetter(event.letter);
          setPracticeActive(true);
          showBubble(`Match the ${event.letter} handshape and hold it steady until it's confirmed.`, "tip");
          break;
        case "sign_correct":
          setCurrentLetter(event.letter);
          celebrate(1, `Nice! ${event.letter} confirmed${event.xp > 0 ? ` (+${event.xp} XP)` : ""}.`, "happy");
          break;
        case "sign_incorrect":
          showBubble(
            event.predicted
              ? `The camera's last stable reading was ${event.predicted}. Check the handshape tip and try again.`
              : "Not quite yet. Compare your hand with the reference and try again.",
            "info"
          );
          setMoodFor("encouraging", 3500);
          break;
        case "hint_requested":
          giveHint(event.letter);
          break;
        case "word_complete":
          celebrate(3, `Word complete: ${event.word}${event.xp > 0 ? ` (+${event.xp} XP)` : ""}!`);
          break;
      }
    },
    [showBubble, setMoodFor, celebrate, giveHint]
  );

  // -- page navigation: welcome / tips --------------------------------------
  const shownForPage = useRef<NavPage | null>(null);
  useEffect(() => {
    if (!ready || shownForPage.current === page) return;
    shownForPage.current = page;
    setPracticeActive(false);
    setMoodFor("curious", 1800);
    if (page === "practice") return; // lesson events provide the message
    const count = visits.current[page] ?? 0;
    visits.current[page] = count + 1;
    const tips = pageTips(page, {
      nextLetter,
      allDone: mastered >= 26,
      badges: state.unlockedBadges.length,
    });
    if (tips.length > 0) showBubble(tips[count % tips.length], "tip");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, ready]);

  // -- real progress changes: level, completed letters, badges, streak ------
  const baseline = useRef<{
    xp: number;
    streak: number;
    level: number;
    badges: Set<string>;
    completed: Set<string>;
  } | null>(null);
  useEffect(() => {
    if (!ready) {
      baseline.current = null;
      return;
    }
    const snap = {
      xp: state.xp,
      streak: state.streak,
      level: getLevel(state.xp),
      badges: new Set(state.unlockedBadges),
      completed: new Set(nodes.filter((n) => n.status === "completed").map((n) => n.letter)),
    };
    const prev = baseline.current;
    baseline.current = snap;
    if (!prev) return; // first snapshot after load is the baseline, never an event

    const newBadges = [...snap.badges].filter((id) => !prev.badges.has(id));
    const newLetters = [...snap.completed].filter((l) => !prev.completed.has(l));
    const xpDelta = snap.xp - prev.xp;

    if (snap.level > prev.level) {
      celebrate(5, `Level ${snap.level}! You earned it.`);
    } else if (newLetters.length > 0) {
      celebrate(4, `Letter ${newLetters.join(", ")} complete${xpDelta > 0 ? ` (+${xpDelta} XP)` : ""}!`);
    } else if (newBadges.length > 0) {
      const titles = newBadges.map((id) => BADGES.find((b) => b.id === id)?.title ?? id.replace(/_/g, " "));
      celebrate(3, `New badge: ${titles.join(", ")}!`);
    } else if (snap.streak > prev.streak) {
      celebrate(2, `${snap.streak}-day streak!`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, ready]);

  // -- inactivity: sleepy, and a welcome-back on return ---------------------
  useEffect(() => {
    let sleepTimer: ReturnType<typeof setTimeout> | null = null;
    let sleeping = false;
    let hiddenAt: number | null = null;

    const welcomeBack = () => {
      const streak = stateRef.current.streak;
      showBubble(streak > 0 ? `Welcome back! You're on a ${streak}-day streak.` : "Welcome back! Ready for a sign?", "info");
      setMoodFor("happy", 2500);
    };
    const arm = () => {
      if (sleepTimer) clearTimeout(sleepTimer);
      sleepTimer = setTimeout(() => {
        sleeping = true;
        setSleepy(true);
      }, SLEEP_AFTER_MS);
    };
    const onActivity = () => {
      if (sleeping) {
        sleeping = false;
        setSleepy(false);
        welcomeBack();
      }
      arm();
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") {
        hiddenAt = nowMs();
      } else if (hiddenAt !== null && nowMs() - hiddenAt > AWAY_WELCOME_MS) {
        hiddenAt = null;
        welcomeBack();
      }
    };

    arm();
    window.addEventListener("pointerdown", onActivity);
    window.addEventListener("keydown", onActivity);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      if (sleepTimer) clearTimeout(sleepTimer);
      window.removeEventListener("pointerdown", onActivity);
      window.removeEventListener("keydown", onActivity);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [showBubble, setMoodFor]);

  // -- conversation ---------------------------------------------------------
  const run = useCallback(
    async (question: string) => {
      lastQuestion.current = question;
      setBusy(true);
      try {
        const answer = await askSigno(question, conversationId.current);
        if (!mounted.current) return;
        conversationId.current = answer.conversationId;
        pushMessage({ origin: "ai", text: answer.text, source: answer.source });
        setMoodFor("explaining", 2500);
      } catch (error) {
        if (!mounted.current) return;
        pushMessage({
          origin: "local",
          text: error instanceof Error ? error.message : "The assistant couldn't answer just now.",
          failed: true,
        });
        setMoodFor("encouraging", 2500);
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [pushMessage, setMoodFor]
  );

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || busy) return;
      pushMessage({ origin: "user", text });
      await run(text);
    },
    [busy, pushMessage, run]
  );

  const retry = useCallback(() => {
    if (busy || !lastQuestion.current) return;
    setMessages((prev) => prev.filter((m) => !m.failed));
    void run(lastQuestion.current);
  }, [busy, run]);

  const canHint = currentLetter !== null && (page === "practice" || page === "alphabet");

  const hint = useCallback(() => {
    if (currentLetter) giveHint(currentLetter);
  }, [currentLetter, giveHint]);

  const celebrateProgress = useCallback(() => {
    pushMessage({ origin: "user", text: "Celebrate my progress!" });
    const s = stateRef.current;
    const parts = [
      `${mastered} of 26 letters completed`,
      `${s.xp} XP`,
      `${s.streak}-day streak`,
      `${s.unlockedBadges.length} badge${s.unlockedBadges.length === 1 ? "" : "s"} earned`,
    ];
    const anything = mastered > 0 || s.xp > 0 || s.streak > 0 || s.unlockedBadges.length > 0;
    pushMessage({
      origin: "local",
      text: anything ? `Here's where you are: ${parts.join(", ")}. Keep going!` : "Your first sign is waiting for you. I'll cheer when you land it.",
    });
    setMoodFor(anything ? "celebrating" : "encouraging", 3200);
  }, [mastered, pushMessage, setMoodFor]);

  // -- derived --------------------------------------------------------------
  const mood: SignoMood = busy
    ? "thinking"
    : transient
      ? transient.mood
      : typing || practiceActive
        ? "listening"
        : sleepy
          ? "sleepy"
          : "idle";

  const accessories = useMemo<SignoAccessory[]>(() => {
    const list: SignoAccessory[] = [];
    if (state.unlockedBadges.includes("first_sign")) list.push("headband");
    if (state.unlockedBadges.includes("alphabet_explorer")) list.push("glasses");
    if (state.unlockedBadges.includes("alphabet_master")) list.push("crown");
    return list;
  }, [state.unlockedBadges]);

  const suggestions = useMemo(
    () => suggestionsFor(page, currentLetter, mastered, nextLetter),
    [page, currentLetter, mastered, nextLetter]
  );

  const value = useMemo<SignoContextValue>(
    () => ({
      page,
      mood,
      pulse,
      reaction,
      bubble,
      hasTip,
      announcement,
      open,
      prefs,
      messages,
      busy,
      suggestions,
      accessories,
      currentLetter,
      canHint,
      emit,
      setOpen,
      setPrefs,
      setTyping,
      dismissBubble,
      react,
      ask,
      retry,
      hint,
      celebrateProgress,
    }),
    [
      page, mood, pulse, reaction, bubble, hasTip, announcement, open, prefs, messages, busy, suggestions,
      accessories, currentLetter, canHint, emit, setOpen, setPrefs, dismissBubble, react, ask, retry, hint,
      celebrateProgress,
    ]
  );

  return <SignoContext.Provider value={value}>{children}</SignoContext.Provider>;
}
