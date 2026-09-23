import { useEffect, useState } from "react";
import { Card, EmptyState, PrimaryButton, SecondaryButton, StatusChip } from "../components/AppShell";
import { AlphabetReferencePanel } from "../components/AlphabetReferencePanel";
import { CameraPractice } from "../components/CameraPractice";
import { CelebrationOverlay, LevelUpOverlay } from "../components/CelebrationOverlay";
import { useToast } from "../components/Toast";
import { BADGES, DAILY_TARGET } from "../game/constants";
import { useGame } from "../game/GameContext";
import type { PracticeOutcome } from "../game/types";
import { buildIncorrectFeedback, pickEncouragement } from "./alphabetFeedback";

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
  // Set once per correct detection, from the same prediction the model used
  // to confirm the sign -- shown as a small secondary line in the
  // celebration, never as the primary success message.
  const [confidence, setConfidence] = useState<number | null>(null);
  const [encouragement, setEncouragement] = useState("");
  // The last stable wrong letter the camera settled on (already
  // temporally-smoothed/debounced upstream in CameraPractice), or null once
  // it's been resolved. Drives the "Almost! You're showing X..." callout.
  const [incorrectFeedback, setIncorrectFeedback] = useState<{ predicted: string } | null>(null);
  // Bumped on "Practice again" to reset the current attempt (votes, handled
  // flags) without tearing down the camera stream -- see CameraPractice.
  const [attemptKey, setAttemptKey] = useState(0);

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
    setConfidence(null);
    setIncorrectFeedback(null);
    setCameraStatus("Show your hand");
    setAttemptKey((key) => key + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [practiceLetter, challengeMode]);

  function handleCorrect(elapsedMs: number, detectedConfidence: number) {
    const result = markCorrect(targetLetter, elapsedMs);
    setOutcome(result);
    setConfidence(detectedConfidence);
    setEncouragement(pickEncouragement(targetLetter));
    setIncorrectFeedback(null);
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

  function handleIncorrect(predictedLetter: string) {
    markIncorrect(targetLetter, predictedLetter);
    setIncorrectFeedback({ predicted: predictedLetter });
  }

  function handleNext() {
    const next = nextPracticeLetter();
    setTargetLetter(next);
    setPhase("playing");
    setOutcome(null);
    setConfidence(null);
    setIncorrectFeedback(null);
    setShowLevelUp(false);
    setCameraStatus("Show your hand");
    setAttemptKey((key) => key + 1);
    startPractice(next, challengeMode);
  }

  // Same target letter, fresh attempt -- no XP/streak change, just resets the
  // camera loop and clears any wrong-guess callout (item 9: "Practice Again
  // should reset only the current attempt without incorrectly awarding
  // another XP event").
  function handlePracticeAgain() {
    setPhase("playing");
    setOutcome(null);
    setConfidence(null);
    setIncorrectFeedback(null);
    setShowLevelUp(false);
    setCameraStatus("Show your hand");
    setAttemptKey((key) => key + 1);
  }

  const challengeLabel = CHALLENGE_LABELS[challengeMode];
  const dailyProgress = `${state.dailyChallenge.progress} / ${DAILY_TARGET}`;
  const feedback = incorrectFeedback ? buildIncorrectFeedback(targetLetter, incorrectFeedback.predicted) : null;

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

        {/* Desktop: camera on the left, ASL reference + target info on the
            right (task requirement). Mobile: camera first, reference stacked
            below it -- same DOM order as the grid columns, so no xl:order
            swap is needed and nothing can overflow horizontally. */}
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
          <Card className="min-w-0 p-3 md:p-4" padding={false}>
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
                attemptKey={attemptKey}
                active={phase === "playing"}
                ready={ready}
                backendError={backendError}
                onCorrect={handleCorrect}
                onIncorrect={handleIncorrect}
                onStatusChange={setCameraStatus}
              />
            )}
          </Card>

          <div className="min-w-0 space-y-4">
            {phase === "success" && outcome ? (
              <CelebrationOverlay
                letter={targetLetter}
                outcome={outcome}
                onNext={handleNext}
                onPracticeAgain={handlePracticeAgain}
                streak={state.streak}
                confidence={confidence}
                encouragement={encouragement}
              />
            ) : (
              <>
                <AlphabetReferencePanel
                  letter={targetLetter}
                  statusLabel={cameraStatus}
                  statusTone={incorrectFeedback ? "warning" : "accent"}
                />

                {feedback && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="rounded-[var(--radius-panel)] border border-[var(--color-warm)]/30 bg-[var(--color-warm)]/[0.06] px-4 py-3 text-sm text-[var(--color-mist)]"
                  >
                    <p className="font-semibold text-[var(--color-warm)]">{feedback.headline}</p>
                    <p className="mt-1 leading-relaxed">{feedback.detail}</p>
                  </div>
                )}

                <details className="rounded-2xl border border-[var(--color-line-soft)] bg-[var(--color-panel-soft)] px-4 py-3 text-sm text-[var(--color-mist)]">
                  <summary className="cursor-pointer font-medium text-[#eef4f0]">How it works</summary>
                  <ol className="mt-2 space-y-2 pl-4 [list-style:decimal]">
                    <li>Allow camera access when prompted.</li>
                    <li>Form the target letter clearly in frame.</li>
                    <li>Hold steady until the app confirms your sign.</li>
                  </ol>
                </details>

                <div className="flex flex-wrap gap-3">
                  <SecondaryButton onClick={() => navigate("home")}>Back home</SecondaryButton>
                  <PrimaryButton onClick={handleNext}>Skip letter</PrimaryButton>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
