import { useEffect, useRef, useState } from "react";
import { Card, EmptyState, PrimaryButton, ProgressBar, SecondaryButton } from "../components/AppShell";
import { AlphabetReferencePanel } from "../components/AlphabetReferencePanel";
import { CameraPractice } from "../components/CameraPractice";
import { LandmarkPractice } from "../components/LandmarkPractice";
import { CelebrationOverlay, LevelUpOverlay } from "../components/CelebrationOverlay";
import { LetterProgressStrip } from "../components/LetterProgressStrip";
import { MissionCard } from "../components/MissionCard";
import type { ExperimentalLandmarkInfo, RecognizerChoice } from "../api";
import { useSignoEmit } from "../components/signo/SignoContext";
import { useToast } from "../components/Toast";
import { BADGES, DAILY_TARGET, MASTERED_CORRECT } from "../game/constants";
import { getLearningPathNodes, lettersMasteredCount } from "../game/gamification";
import { useGame } from "../game/GameContext";
import type { PracticeOutcome } from "../game/types";
import {
  SIGN_HINTS,
  buildIncorrectFeedback,
  pickEncouragement,
  referenceImageAlt,
  referenceImageSrc,
} from "./alphabetFeedback";

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
  experimentalLandmark = null,
}: {
  ready: boolean;
  backendError: string | null;
  /** Set only when the backend has an experimental landmark model loaded; enables the developer comparison switch. */
  experimentalLandmark?: ExperimentalLandmarkInfo | null;
}) {
  // Hand landmarks are the recognizer whenever the backend has the landmark model loaded; there is no user-facing choice.
  // Only if that model is unavailable does Practice fall back to the ResNet image classifier so it keeps working.
  const activeRecognizer: RecognizerChoice = experimentalLandmark?.loaded ? "landmarks" : "resnet";
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
  const emit = useSignoEmit();

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
  // Presentational only -- how many tries the learner has had on the current
  // letter, shown on MissionCard. Never touches game state/XP.
  const [attemptNumber, setAttemptNumber] = useState(1);
  // Lesson introduction: shown when a learner opens a specific letter from
  // Learn/Alphabet (not for challenges, free practice, or "Next letter").
  const [introDone, setIntroDone] = useState(false);
  const skipIntroRef = useRef(false);

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
    setAttemptNumber(1);
    setIntroDone(skipIntroRef.current);
    skipIntroRef.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [practiceLetter, challengeMode]);

  function handleCorrect(elapsedMs: number, detectedConfidence: number) {
    const result = markCorrect(targetLetter, elapsedMs);
    emit({ type: "sign_correct", letter: targetLetter, xp: result.xpGained });
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
    emit({ type: "sign_incorrect", letter: targetLetter, predicted: predictedLetter });
    setIncorrectFeedback({ predicted: predictedLetter });
    setAttemptNumber((value) => value + 1);
  }

  function handleNext() {
    const next = nextPracticeLetter();
    skipIntroRef.current = true;
    setTargetLetter(next);
    setPhase("playing");
    setOutcome(null);
    setConfidence(null);
    setIncorrectFeedback(null);
    setShowLevelUp(false);
    setCameraStatus("Show your hand");
    setAttemptKey((key) => key + 1);
    setAttemptNumber(1);
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
    setAttemptNumber(1);
  }

  const challengeLabel = CHALLENGE_LABELS[challengeMode];
  const dailyProgress = `${state.dailyChallenge.progress} / ${DAILY_TARGET}`;
  const feedback = incorrectFeedback ? buildIncorrectFeedback(targetLetter, incorrectFeedback.predicted) : null;
  const letterCorrect = state.letterStats?.[targetLetter]?.correct ?? 0;
  const masteredCount = lettersMasteredCount(state);
  const pathNodes = getLearningPathNodes(state);
  const showIntro = practiceLetter !== null && challengeMode === "none" && !introDone;

  // Tell SIGNO what is genuinely happening: a lesson intro is on screen, or the
  // camera lesson has started. (Recognition results are reported separately, from
  // the real onCorrect/onIncorrect callbacks.)
  useEffect(() => {
    if (showIntro) emit({ type: "lesson_opened", letter: targetLetter });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showIntro, targetLetter]);
  useEffect(() => {
    if (!showIntro && ready && phase === "playing") emit({ type: "practice_started", letter: targetLetter });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showIntro, ready, phase, targetLetter]);

  return (
    <>
      {showLevelUp && outcome?.leveledUp && (
        <LevelUpOverlay level={outcome.newLevel} onDismiss={() => setShowLevelUp(false)} />
      )}

      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--color-muted)]">
          <button
            type="button"
            className="font-semibold text-[var(--color-primary)] hover:underline"
            onClick={() => navigate("learn")}
          >
            ← Back to Learning
          </button>
          <span>
            Level {level} · {state.xp} XP · {state.streak}d streak
            {challengeMode === "daily" ? ` · ${dailyProgress}` : ""}
          </span>
        </div>

        {showIntro ? (
          <Card className="mx-auto max-w-2xl space-y-5" data-testid="lesson-intro">
            <div>
              <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-[var(--color-primary-dim)]">
                Lesson
              </p>
              <h1 className="mt-1 font-display text-3xl font-bold text-[var(--color-ink)]">Letter {targetLetter}</h1>
              <p className="mt-2 text-base text-[var(--color-ink-soft)]">
                Objective: form the letter {targetLetter} so the camera recognizes it. Each confirmed sign earns XP; a
                letter is complete after {MASTERED_CORRECT} correct signs.
              </p>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-28 shrink-0 overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] p-2">
                <img
                  src={referenceImageSrc(targetLetter)}
                  alt={referenceImageAlt(targetLetter)}
                  className="h-auto w-full"
                  width={100}
                  height={131}
                />
              </div>
              <div className="min-w-0 flex-1 space-y-3">
                <p className="text-sm font-semibold text-[var(--color-ink)]">
                  {SIGN_HINTS[targetLetter] ?? `Match the ${targetLetter} handshape.`}
                </p>
                <ol className="space-y-1.5 pl-4 text-sm text-[var(--color-ink-soft)] [list-style:decimal]">
                  <li>Allow camera access when prompted.</li>
                  <li>Form the letter clearly in frame, with good light.</li>
                  <li>Hold steady until the app confirms your sign.</li>
                </ol>
              </div>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-sm font-bold text-[var(--color-ink)]">
                <span>Your progress</span>
                <span className="tabular-nums text-[var(--color-muted)]">
                  {Math.min(letterCorrect, MASTERED_CORRECT)} / {MASTERED_CORRECT}
                </span>
              </div>
              <ProgressBar percent={(Math.min(letterCorrect, MASTERED_CORRECT) / MASTERED_CORRECT) * 100} showPercent={false} />
            </div>
            <div className="flex flex-wrap gap-3">
              <PrimaryButton onClick={() => setIntroDone(true)}>Start lesson</PrimaryButton>
              <SecondaryButton onClick={() => navigate("learn")}>Back to Learn</SecondaryButton>
            </div>
          </Card>
        ) : (
        <>
        <MissionCard challengeLabel={challengeLabel} attemptNumber={attemptNumber} streak={state.streak} />

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-sm font-bold text-[var(--color-ink)]">
            <span>Letter {targetLetter}</span>
            <span className="tabular-nums text-[var(--color-muted)]">
              {Math.min(letterCorrect, MASTERED_CORRECT)} / {MASTERED_CORRECT} correct to complete
            </span>
          </div>
          <ProgressBar
            percent={(Math.min(letterCorrect, MASTERED_CORRECT) / MASTERED_CORRECT) * 100}
            showPercent={false}
          />
        </div>

        <div className="space-y-2">
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">
            {masteredCount} / 26 letters mastered
          </p>
          <LetterProgressStrip nodes={pathNodes} />
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
                description={backendError ?? "The recognition service isn't available yet. Refresh in a moment to practice with your camera."}
                action={<SecondaryButton onClick={() => navigate("learn")}>Back to Learn</SecondaryButton>}
              />
            ) : (
              <div className="space-y-3">
              {activeRecognizer === "landmarks" ? (
                // Experimental: MediaPipe runs in the browser and only landmarks are sent. Same callbacks, so XP and
                // progress still go through handleCorrect/handleIncorrect and the existing game logic.
                <LandmarkPractice
                  targetLetter={targetLetter}
                  attemptKey={attemptKey}
                  active={phase === "playing"}
                  ready={ready}
                  backendError={backendError}
                  onCorrect={handleCorrect}
                  onIncorrect={handleIncorrect}
                  onStatusChange={setCameraStatus}
                />
              ) : (
                <CameraPractice
                  recognizer={activeRecognizer}
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
              </div>
            )}
          </Card>

          <div className="min-w-0 space-y-4">
            {phase === "success" && outcome ? (
              <CelebrationOverlay
                letter={targetLetter}
                outcome={outcome}
                onNext={handleNext}
                onPracticeAgain={handlePracticeAgain}
                onBack={() => navigate("learn")}
                streak={state.streak}
                confidence={confidence}
                encouragement={encouragement}
                correct={letterCorrect}
                lettersCompleted={masteredCount}
              />
            ) : (
              <>
                <AlphabetReferencePanel
                  letter={targetLetter}
                  statusLabel={cameraStatus}
                  statusTone={incorrectFeedback ? "warning" : "accent"}
                  correct={letterCorrect}
                />

                {feedback && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="rounded-[var(--radius-panel)] border border-[var(--color-streak)]/25 bg-[var(--color-streak-soft)] px-4 py-3 text-sm text-[var(--color-ink-soft)]"
                  >
                    <p className="font-bold text-[var(--color-streak)]">{feedback.headline}</p>
                    <p className="mt-1 leading-relaxed">{feedback.detail}</p>
                  </div>
                )}

                <details className="rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3 text-sm text-[var(--color-ink-soft)]">
                  <summary className="cursor-pointer font-semibold text-[var(--color-ink)]">How it works</summary>
                  <ol className="mt-2 space-y-2 pl-4 [list-style:decimal]">
                    <li>Allow camera access when prompted.</li>
                    <li>Form the target letter clearly in frame.</li>
                    <li>Hold steady until the app confirms your sign.</li>
                  </ol>
                </details>

                <div className="flex flex-wrap gap-3">
                  <SecondaryButton onClick={() => navigate("learn")}>Back to Learn</SecondaryButton>
                  <SecondaryButton onClick={() => emit({ type: "hint_requested", letter: targetLetter })}>
                    Give me a hint
                  </SecondaryButton>
                  <PrimaryButton onClick={handleNext}>Skip letter</PrimaryButton>
                </div>
              </>
            )}
          </div>
        </div>
        </>
        )}
      </div>
    </>
  );
}
