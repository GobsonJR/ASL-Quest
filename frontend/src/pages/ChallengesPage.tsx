import { Card, PageHeader, PrimaryButton, ProgressBar, SectionLabel, StatusChip } from "../components/AppShell";
import { CHALLENGES, DAILY_TARGET, LETTERS, SPEED_TARGET } from "../game/constants";
import { useGame } from "../game/GameContext";
import { lettersLearnedCount } from "../game/gamification";
import { fetchWordChallenge } from "../words/api";
import { useEffect, useState } from "react";

const DIFFICULTY: Record<string, string> = {
  daily: "Easy",
  alphabet: "Hard",
  speed: "Medium",
  word: "Medium",
};

export function ChallengesPage() {
  const { state, startPractice, startWordPractice, restartSpeedChallenge } = useGame();
  const [dailyWords, setDailyWords] = useState<string[]>([]);

  useEffect(() => {
    fetchWordChallenge("daily")
      .then((payload) => setDailyWords(payload.items.map((item) => item.word)))
      .catch(() => setDailyWords([]));
  }, []);

  const progressByChallenge = {
    daily: (state.dailyChallenge.progress / DAILY_TARGET) * 100,
    alphabet: (lettersLearnedCount(state) / LETTERS.length) * 100,
    speed: state.challenges.speedCompleted ? 100 : (state.challenges.speedCount / SPEED_TARGET) * 100,
    word: state.challenges.wordCompleted ? 100 : 0,
  };

  const progressLabel = {
    daily: `${state.dailyChallenge.progress} / ${DAILY_TARGET} signs`,
    alphabet: `${lettersLearnedCount(state)} / ${LETTERS.length} letters`,
    speed: state.challenges.speedCompleted
      ? "Completed"
      : `${state.challenges.speedCount} / ${SPEED_TARGET} signs`,
    word: state.challenges.wordCompleted ? "Completed" : `${state.challenges.wordCount} completed`,
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="🎯 Today's missions"
        title="Game modes & bonus XP"
        description="Complete challenges to level up faster, maintain your streak, and unlock badges."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        {CHALLENGES.map((challenge) => {
          const completed =
            challenge.id === "daily"
              ? state.dailyChallenge.completed
              : challenge.id === "alphabet"
                ? state.challenges.alphabetCompleted
                : challenge.id === "word"
                  ? state.challenges.wordCompleted
                  : state.challenges.speedCompleted;

          return (
            <Card key={challenge.id} hover className="flex h-full flex-col">
              <div className="flex items-start justify-between gap-3">
                <div className="text-4xl" aria-hidden="true">
                  {challenge.icon}
                </div>
                <div className="flex flex-wrap gap-2">
                  <StatusChip tone="neutral">{DIFFICULTY[challenge.id]}</StatusChip>
                  {completed && <StatusChip tone="success">Complete</StatusChip>}
                </div>
              </div>

              <h3 className="mt-4 font-display text-2xl">{challenge.title}</h3>
              <p className="mt-2 flex-1 text-sm text-[var(--color-ink-soft)]">{challenge.description}</p>

              <div className="mt-5 space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <SectionLabel>Progress</SectionLabel>
                  <span className="text-[var(--color-ink-soft)]">{progressLabel[challenge.id]}</span>
                </div>
                <ProgressBar percent={progressByChallenge[challenge.id]} showPercent={false} />
                <p className="text-sm font-semibold text-[var(--color-primary)]">Reward: +{challenge.reward} XP</p>
              </div>

              <PrimaryButton
                className="mt-5 w-full"
                onClick={async () => {
                  if (challenge.id === "speed") restartSpeedChallenge();
                  if (challenge.id === "word") {
                    const payload = await fetchWordChallenge("word");
                    const ids = payload.items.map((item) => item.id);
                    if (ids[0]) startWordPractice(ids[0], "word", ids);
                    return;
                  }
                  if (challenge.id === "daily") {
                    startPractice(null, "daily");
                    return;
                  }
                  startPractice(null, challenge.id);
                }}
              >
                {completed ? "Play Again" : "Start Challenge"}
              </PrimaryButton>
            </Card>
          );
        })}
      </div>

      {dailyWords.length > 0 && (
        <Card>
          <SectionLabel>Today's spelling words</SectionLabel>
          <p className="mt-2 text-sm text-[var(--color-ink-soft)]">
            Keep the daily letter goal, and also spell these three words if you want extra practice. They stay the same all day.
          </p>
          <p className="mt-3 font-display text-2xl tracking-wide">{dailyWords.join("   ·   ")}</p>
          <PrimaryButton
            className="mt-4"
            onClick={async () => {
              const payload = await fetchWordChallenge("daily");
              const ids = payload.items.map((item) => item.id);
              if (ids[0]) startWordPractice(ids[0], "daily", ids);
            }}
          >
            Practice today's words
          </PrimaryButton>
        </Card>
      )}
    </div>
  );
}
