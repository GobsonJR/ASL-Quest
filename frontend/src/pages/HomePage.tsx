import { useEffect, useState } from "react";
import { fetchRecommendations, type RecommendationsResponse } from "../auth/api";
import { fetchAnalyticsDashboard } from "../analytics/api";
import { fetchWordRecommendations, type WordRecommendations } from "../words/api";
import { getWordById } from "../words/catalog";
import {
  Divider,
  PageHeader,
  PageLayout,
  Panel,
  PrimaryButton,
  ProgressBar,
  SecondaryButton,
  SectionHeader,
  StatGroup,
} from "../components/AppShell";
import { LetterMasteryBar } from "../components/LetterMasteryBar";
import { DAILY_TARGET, XP_DAILY_COMPLETE } from "../game/constants";
import { useGame } from "../game/GameContext";
import {
  getContinueLetter,
  getMasteryPercent,
  lettersLearnedCount,
  lettersMasteredCount,
} from "../game/gamification";

export function HomePage() {
  const { state, level, levelProgress, startPractice, startWordPractice, navigate } = useGame();
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null);
  const [wordRecs, setWordRecs] = useState<WordRecommendations | null>(null);
  const [weeklyAttempts, setWeeklyAttempts] = useState<number | null>(null);
  const daily = state.dailyChallenge;
  const continueLetter = getContinueLetter(state);
  const continueProgress = getMasteryPercent(state.letterStats[continueLetter].correct);
  const dailyPercent = (daily.progress / daily.target) * 100;
  const weakest = recommendations?.weakest;
  const recommendedLetter = recommendations?.recommended[0]?.letter ?? weakest?.letter ?? continueLetter;

  useEffect(() => {
    fetchRecommendations()
      .then(setRecommendations)
      .catch(() => setRecommendations(null));
    fetchWordRecommendations()
      .then(setWordRecs)
      .catch(() => setWordRecs(null));
    fetchAnalyticsDashboard("7d", "3m")
      .then((data) => setWeeklyAttempts(data.weekly_summary.current_week.attempts))
      .catch(() => setWeeklyAttempts(null));
  }, []);

  return (
    <PageLayout>
      <PageHeader
        eyebrow="Welcome back"
        title="What should you practice next?"
        description="One letter at a time — build mastery, keep your streak, and earn XP."
        compact
      />

      <Panel className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-5">
          <div className="grid h-20 w-20 shrink-0 place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line-soft)] bg-[var(--color-ink)] font-display text-4xl text-[var(--color-accent)]">
            {continueLetter}
          </div>
          <div className="min-w-0">
            <p className="text-sm text-[var(--color-muted)]">Continue learning</p>
            <h3 className="mt-1 font-display text-2xl">Letter {continueLetter}</h3>
            <p className="mt-1 text-sm text-[var(--color-mist)]">{Math.round(continueProgress)}% toward mastery</p>
            <div className="mt-3 max-w-xs">
              <LetterMasteryBar correct={state.letterStats[continueLetter].correct} showBadge={false} />
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <PrimaryButton onClick={() => startPractice(continueLetter)} ariaLabel={`Continue practicing letter ${continueLetter}`}>
            Continue practice
          </PrimaryButton>
          {weakest && weakest.letter !== continueLetter && (
            <SecondaryButton onClick={() => startPractice(weakest.letter)}>Practice {weakest.letter}</SecondaryButton>
          )}
        </div>
      </Panel>

      {wordRecs?.continue_word && (
        <section className="space-y-3">
          <SectionHeader title="Continue spelling" description="Pick up a word you've already started." />
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="font-display text-2xl">{wordRecs.continue_word.word}</h3>
              <p className="mt-2 text-sm text-[var(--color-mist)]">
                {(getWordById(wordRecs.continue_word.word_id)?.letters ?? wordRecs.continue_word.word.split("")).map((letter, idx) => (
                  <span key={`${letter}-${idx}`} className="mr-3">
                    {letter} {idx < (wordRecs.continue_word?.resume_index ?? 0) ? "✓" : "○"}
                  </span>
                ))}
              </p>
            </div>
            <PrimaryButton onClick={() => startWordPractice(wordRecs.continue_word!.word_id)}>Continue</PrimaryButton>
          </div>
        </section>
      )}

      {wordRecs?.recommended[0] && (
        <p className="text-sm text-[var(--color-mist)]">
          Recommended word: <button type="button" className="font-medium text-[var(--color-accent)] hover:underline" onClick={() => startWordPractice(wordRecs.recommended[0].word_id)}>{wordRecs.recommended[0].message}</button>
        </p>
      )}

      <div>
        <StatGroup
          items={[
            { label: "Level", value: level, sub: `${levelProgress.current} / ${levelProgress.next} XP` },
            { label: "Streak", value: `${state.streak} days`, sub: `Best ${state.longestStreak} days` },
            { label: "Mastered", value: `${lettersMasteredCount(state)} / 26` },
            { label: "Correct signs", value: state.totalCorrect },
            ...(weeklyAttempts != null ? [{ label: "This week", value: weeklyAttempts, sub: "attempts" }] : []),
          ]}
        />
        <div className="mt-4">
          <ProgressBar percent={levelProgress.percent} label={`Level ${level}`} size="sm" />
        </div>
      </div>

      <Divider />

      <div className="grid gap-8 lg:grid-cols-2">
        <section className="space-y-4">
          <SectionHeader
            title="Today's goal"
            description={`Sign ${DAILY_TARGET} letters correctly${!daily.completed ? ` · +${XP_DAILY_COMPLETE} XP` : ""}`}
          />
          <ProgressBar percent={dailyPercent} shimmer={!daily.completed} size="sm" />
          <p className="text-sm text-[var(--color-mist)]">
            {daily.progress} of {daily.target} completed
          </p>
          <div className="flex flex-wrap gap-3">
            <PrimaryButton onClick={() => startPractice(null, "daily")}>
              {daily.completed ? "Keep practicing" : "Start daily challenge"}
            </PrimaryButton>
            <SecondaryButton onClick={() => navigate("challenges")}>All challenges</SecondaryButton>
          </div>
        </section>

        <section className="space-y-4">
          <SectionHeader
            title="Recommended for you"
            description={
              recommendations?.has_data
                ? "Based on your recent practice."
                : recommendations?.message ?? "Loading recommendations..."
            }
          />
          {recommendations?.has_data ? (
            <>
              <div className="flex flex-wrap gap-2">
                {recommendations.recommended.slice(0, 3).map((item) => (
                  <button
                    key={item.letter}
                    type="button"
                    className="rounded-[var(--radius-control)] border border-[var(--color-line)] px-4 py-2 text-sm font-medium text-[var(--color-mist)] transition hover:border-[var(--color-accent)]/40 hover:text-[var(--color-accent)]"
                    onClick={() => startPractice(item.letter)}
                  >
                    {item.letter}
                  </button>
                ))}
              </div>
              {weakest && (
                <p className="text-sm text-[var(--color-muted)]">
                  Weakest practiced: {weakest.letter}
                  {weakest.accuracy != null ? ` · ${Math.round(weakest.accuracy)}% accuracy` : ""}
                </p>
              )}
            </>
          ) : (
            <PrimaryButton onClick={() => startPractice(recommendedLetter)}>Start your first session</PrimaryButton>
          )}
        </section>
      </div>

      <Divider />

      <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <section className="space-y-4">
          <SectionHeader title="Alphabet progress" description={`${lettersLearnedCount(state)} learned · ${lettersMasteredCount(state)} mastered`} />
          <div className="space-y-3">
            <ProgressBar percent={(lettersLearnedCount(state) / 26) * 100} label="Learned" size="sm" />
            <ProgressBar percent={(lettersMasteredCount(state) / 26) * 100} label="Mastered" size="sm" />
          </div>
          <SecondaryButton onClick={() => navigate("learn")}>Explore the alphabet</SecondaryButton>
        </section>

        <section className="space-y-4">
          <SectionHeader
            title="Word spelling"
            description="Spell everyday words one A–Z sign at a time. This is not native ASL word recognition."
          />
          <PrimaryButton onClick={() => navigate("words")}>Word practice</PrimaryButton>
        </section>

        <section className="space-y-4">
          <SectionHeader
            title="Recent badges"
            action={
              state.recentAchievements.length > 0 ? (
                <button
                  type="button"
                  className="text-sm font-medium text-[var(--color-accent)] hover:underline"
                  onClick={() => navigate("achievements")}
                >
                  View all
                </button>
              ) : undefined
            }
          />
          {state.recentAchievements.length === 0 ? (
            <p className="text-sm text-[var(--color-mist)]">Complete your first sign to unlock a badge.</p>
          ) : (
            <ul className="space-y-3">
              {state.recentAchievements.slice(0, 3).map((item) => (
                <li key={`${item.id}-${item.unlockedAt}`} className="flex items-center justify-between gap-3 text-sm">
                  <span className="font-medium text-[#eef4f0]">{item.title}</span>
                  <span className="text-[var(--color-muted)]">{new Date(item.unlockedAt).toLocaleDateString()}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </PageLayout>
  );
}
