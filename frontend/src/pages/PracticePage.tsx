import { useEffect, useState } from "react";
import { Card, EmptyState, Panel, PrimaryButton, SecondaryButton, StatusChip } from "../components/AppShell";
import { CameraPractice } from "../components/CameraPractice";
import { CelebrationOverlay, LevelUpOverlay } from "../components/CelebrationOverlay";
import { useToast } from "../components/Toast";
import { BADGES, DAILY_TARGET } from "../game/constants";
import { useGame } from "../game/GameContext";
import type { PracticeOutcome } from "../game/types";

const CHALLENGE_LABELS = {
  none: "Free practice",
  daily: "Daily challenge",
  alphabet: "Alphabet challenge",
  speed: "Speed challenge",
  word: "Word challenge",
} as const;

function badgeTitle(id: string): string {
  return BADGES.find((badge) => badge.id === id)?.title ?? id.replace(/_/g, " ");
}

export function PracticePage({
  ready,
  backendError,
}: {
  ready: boolean;
  backendError: string | null;
}) {
  const {
    state,
    level,
    practiceLetter,
    challengeMode,
    beginSession,
    nextPracticeLetter,
    markCorrect,
    markIncorrect,
    startPractice,
    navigate,
  } = useGame();
  const { push } = useToast();

  const [targetLetter, setTargetLetter] = useState(() => nextPracticeLetter());
  const [phase, setPhase] = useState<"playing" | "success">("playing");
  const [outcome, setOutcome] = useState<PracticeOutcome | null>(null);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [showLevelUp, setShowLevelUp] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Show your hand");

  useEffect(() => {
    if (!sessionStarted) {
      beginSession();
      setSessionStarted(true);
    }
  }, [beginSession, sessionStarted]);

  useEffect(() => {
    setTargetLetter(practiceLetter ?? nextPracticeLetter());
    setPhase("playing");
    setOutcome(null);
    setCameraStatus("Show your hand");
  }, [practiceLetter, challengeMode]);

  function handleCorrect(elapsedMs: number) {
    const result = markCorrect(targetLetter, elapsedMs);
    setOutcome(result);
    setPhase("success");
    push(`+${result.xpGained} XP`, "success");
    if (result.badgesUnlocked.length > 0) {
      push(`Achievement unlocked: ${result.badgesUnlocked.map(badgeTitle).join(", ")}`, "info");
    }
    if (result.leveledUp) {
      push(`Level ${result.newLevel}!`, "info");
      setShowLevelUp(true);
    }
  }

  function handleNext() {
    const next = nextPracticeLetter();
    setTargetLetter(next);
    setPhase("playing");
    setOutcome(null);
    setShowLevelUp(false);
    startPractice(next, challengeMode);
  }

  const challengeLabel = CHALLENGE_LABELS[challengeMode];
  const dailyProgress = `${state.dailyChallenge.progress} / ${DAILY_TARGET}`;

  return (
    <>
      {showLevelUp && outcome?.leveledUp && (
        <LevelUpOverlay level={outcome.newLevel} onDismiss={() => setShowLevelUp(false)} />
      )}

      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--color-muted)]">
          <span>{challengeLabel}</span>
          <div className="flex gap-2">
            <StatusChip tone="accent">Letters</StatusChip>
            <button
              type="button"
              className="text-sm text-[var(--color-accent)] hover:underline"
              onClick={() => navigate("words")}
            >
              Words
            </button>
          </div>
          <span>
            Level {level} · {state.xp} XP · {state.streak}d streak
            {challengeMode === "daily" ? ` · ${dailyProgress}` : ""}
          </span>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
          <div className="order-2 space-y-4 xl:order-1">
            <Panel className="text-center">
              <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">Sign this letter</p>
              <div className="relative mx-auto mt-4 grid h-32 w-32 place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line-soft)] bg-[var(--color-ink)] md:h-40 md:w-40">
                <span className="font-display text-6xl text-[var(--color-accent)] md:text-7xl">{targetLetter}</span>
                {phase === "success" && (
                  <span className="success-ring pointer-events-none absolute inset-0 rounded-[var(--radius-panel)] border-2 border-[var(--color-success)]" />
                )}
              </div>
              <div className="mt-4 flex justify-center">
                <StatusChip tone={phase === "success" ? "success" : "accent"}>{cameraStatus}</StatusChip>
              </div>
            </Panel>

            {phase === "success" && outcome ? (
              <CelebrationOverlay letter={targetLetter} outcome={outcome} onNext={handleNext} />
            ) : (
              <div className="space-y-4 text-sm text-[var(--color-mist)]">
                <p className="font-medium text-[#eef4f0]">How it works</p>
                <ol className="space-y-2 pl-4 [list-style:decimal]">
                  <li>Allow camera access when prompted.</li>
                  <li>Form the target letter clearly in frame.</li>
                  <li>Hold steady until the app confirms your sign.</li>
                </ol>
                <div className="flex flex-wrap gap-3 pt-2">
                  <SecondaryButton onClick={() => navigate("home")}>Back home</SecondaryButton>
                  <PrimaryButton onClick={handleNext}>Skip letter</PrimaryButton>
                </div>
              </div>
            )}
          </div>

          <Card className="order-1 p-3 md:p-4 xl:order-2" padding={false}>
            {!ready ? (
              <EmptyState
                icon="📷"
                title="Practice isn't ready yet"
                description={
                  backendError ??
                  "Start the FastAPI backend on port 8000, then refresh to practice with your camera."
                }
                action={<SecondaryButton onClick={() => navigate("home")}>Back to home</SecondaryButton>}
              />
            ) : (
              <CameraPractice
                targetLetter={targetLetter}
                active={phase === "playing"}
                ready={ready}
                backendError={backendError}
                onCorrect={handleCorrect}
                onIncorrect={() => markIncorrect(targetLetter)}
                onStatusChange={setCameraStatus}
              />
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
