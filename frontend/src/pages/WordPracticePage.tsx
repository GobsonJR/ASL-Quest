import { useEffect, useRef, useState } from "react";
import { Card, PageHeader, Panel, SecondaryButton, StatusChip } from "../components/AppShell";
import { CameraPractice } from "../components/CameraPractice";
import { WordCompleteOverlay } from "../components/CelebrationOverlay";
import { useToast } from "../components/Toast";
import { useGame } from "../game/GameContext";
import { getWordById, nextWordId, WORD_CATALOG } from "../words/catalog";
import { recordWordSession } from "../words/api";

type LetterLog = {
  letter: string;
  prediction: string | null;
  correct: boolean;
  response_time?: number | null;
};

export function WordPracticePage({
  ready,
  backendError,
}: {
  ready: boolean;
  backendError: string | null;
}) {
  const {
    practiceWordId,
    wordQueue,
    challengeMode,
    startWordPractice,
    navigate,
    beginSession,
    applyWordXp,
    markWordChallengeComplete,
  } = useGame();
  const { push } = useToast();

  const word = getWordById(practiceWordId ?? "") ?? WORD_CATALOG[0];
  const letters = word.letters;
  const [index, setIndex] = useState(0);
  const [logs, setLogs] = useState<LetterLog[]>([]);
  const [mistakes, setMistakes] = useState(0);
  const [phase, setPhase] = useState<"playing" | "complete">("playing");
  const [completionXp, setCompletionXp] = useState(0);
  const [challengeFinished, setChallengeFinished] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Show your hand");
  const [sessionStarted, setSessionStarted] = useState(false);
  const startedAt = useRef(Date.now());
  const finishingRef = useRef(false);
  const queueIndex = Math.max(0, wordQueue.indexOf(word.id));

  useEffect(() => {
    if (!sessionStarted) {
      beginSession();
      setSessionStarted(true);
    }
  }, [beginSession, sessionStarted]);

  useEffect(() => {
    setIndex(0);
    setLogs([]);
    setMistakes(0);
    setPhase("playing");
    setCompletionXp(0);
    setChallengeFinished(false);
    setCameraStatus("Show your hand");
    startedAt.current = Date.now();
    finishingRef.current = false;
  }, [word.id, challengeMode]);

  const currentLetter = letters[index] ?? letters[letters.length - 1];
  const challengeLabel =
    challengeMode === "word"
      ? `Word challenge · ${queueIndex + 1} / ${Math.max(wordQueue.length, 1)}`
      : challengeMode === "daily"
        ? "Daily spelling words"
        : "Word practice";

  async function finishWord(finalLogs: LetterLog[], finalMistakes: number) {
    if (finishingRef.current) return;
    finishingRef.current = true;
    setPhase("complete");
    const duration = Date.now() - startedAt.current;
    const lastInQueue = challengeMode === "word" && wordQueue.length > 0 && queueIndex === wordQueue.length - 1;
    try {
      const result = await recordWordSession({
        word_id: word.id,
        letters: finalLogs,
        completed: true,
        duration_ms: duration,
        mistakes: finalMistakes,
        challenge_type: challengeMode === "none" ? null : challengeMode,
        challenge_finished: lastInQueue,
        resume_index: 0,
      });
      applyWordXp(result.xp_earned, result.new_achievements);
      setCompletionXp(result.xp_earned);
      if (lastInQueue) {
        markWordChallengeComplete();
        setChallengeFinished(true);
      }
      if (result.new_achievements.length > 0) {
        push(`Achievement unlocked: ${result.new_achievements.map((item) => item.name).join(", ")}`, "info");
      }
      if (result.xp_earned > 0) push(`+${result.xp_earned} XP`, "success");
    } catch {
      push("Saved locally — we'll sync word progress when you're back online.", "warning");
    }
  }

  function handleCorrect(elapsedMs: number) {
    const nextLogs = [
      ...logs,
      { letter: currentLetter, prediction: currentLetter, correct: true, response_time: elapsedMs },
    ];
    setLogs(nextLogs);
    if (index + 1 >= letters.length) {
      void finishWord(nextLogs, mistakes);
    } else {
      setIndex((value) => value + 1);
    }
  }

  function handleIncorrect() {
    setMistakes((value) => value + 1);
    setCameraStatus(`Try again — show ${currentLetter}`);
  }

  function restart() {
    startWordPractice(word.id, challengeMode, wordQueue);
  }

  function goNextWord() {
    if (wordQueue.length > 0) {
      const next = wordQueue[queueIndex + 1] ?? nextWordId(word.id);
      startWordPractice(next, challengeMode, wordQueue);
      return;
    }
    startWordPractice(nextWordId(word.id), challengeMode);
  }

  return (
    <>
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--color-muted)]">
          <span>{challengeLabel}</span>
          <span>
            {Math.min(index, letters.length)} / {letters.length} letters
          </span>
        </div>

        <PageHeader
          eyebrow="Target word"
          title={word.word}
          description="Spell this word one static alphabet sign at a time. The model recognizes letters, not the whole word at once."
          compact
        />

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
          <div className="order-2 space-y-4 xl:order-1">
            <Panel>
              <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">Letter sequence</p>
              <ol className="mt-4 flex flex-wrap gap-3">
                {letters.map((letter, letterIndex) => {
                  const done = letterIndex < index || phase === "complete";
                  const current = letterIndex === index && phase === "playing";
                  return (
                    <li
                      key={`${letter}-${letterIndex}`}
                      className={`grid h-12 w-12 place-items-center rounded-[var(--radius-control)] border text-lg font-semibold ${
                        done
                          ? "border-[var(--color-success)]/40 bg-[var(--color-success)]/10 text-[var(--color-success)]"
                          : current
                            ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                            : "border-[var(--color-line-soft)] text-[var(--color-muted)]"
                      }`}
                      aria-current={current ? "step" : undefined}
                      aria-label={`${letter}${done ? " complete" : current ? " current" : " upcoming"}`}
                    >
                      {letter}
                    </li>
                  );
                })}
              </ol>
              <p className="mt-4 text-sm text-[var(--color-mist)]">
                {letters.map((letter, letterIndex) => (
                  <span key={`${letter}-status-${letterIndex}`} className="mr-3">
                    {letter} {letterIndex < index || phase === "complete" ? "✓" : "○"}
                  </span>
                ))}
              </p>
            </Panel>

            {phase === "complete" ? (
              <WordCompleteOverlay
                word={word.word}
                letters={letters}
                xp={completionXp}
                accuracyLabel={`${letters.length} / ${letters.length} signs correct${mistakes ? ` · ${mistakes} retries` : ""}`}
                challengeSummary={
                  challengeFinished ? `${wordQueue.length} / ${wordQueue.length} words · challenge bonus awarded` : undefined
                }
                onAgain={restart}
                onNext={goNextWord}
                onBack={() => navigate("words")}
              />
            ) : (
              <Panel className="text-center">
                <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">Current sign</p>
                <div className="mx-auto mt-4 grid h-28 w-28 place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line-soft)] bg-[var(--color-ink)] font-display text-6xl text-[var(--color-accent)]">
                  {currentLetter}
                </div>
                <div className="mt-4 flex justify-center">
                  <StatusChip tone="accent">{cameraStatus}</StatusChip>
                </div>
                <p className="mt-4 text-sm text-[var(--color-mist)]">
                  Make the {currentLetter} sign. Hold steady until it is confirmed, then continue to the next letter.
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-3">
                  <SecondaryButton onClick={() => navigate("words")}>Back to words</SecondaryButton>
                </div>
              </Panel>
            )}
          </div>

          <Card className="order-1 p-3 md:p-4 xl:order-2" padding={false}>
            <CameraPractice
              targetLetter={currentLetter}
              active={phase === "playing" && ready}
              ready={ready}
              backendError={backendError}
              onCorrect={handleCorrect}
              onIncorrect={handleIncorrect}
              onStatusChange={setCameraStatus}
              incorrectHint={`Try again — show ${currentLetter}`}
            />
          </Card>
        </div>
      </div>
    </>
  );
}
