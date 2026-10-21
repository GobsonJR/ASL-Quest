import { useEffect, useState } from "react";
import { fetchRecommendations, type RecommendationsResponse } from "../auth/api";
import { fetchAnalyticsDashboard } from "../analytics/api";
import { fetchWordRecommendations, type WordRecommendations } from "../words/api";
import { getWordById } from "../words/catalog";
import {
  Card,
  PageLayout,
  PrimaryButton,
  ProgressBar,
  SecondaryButton,
  SectionHeader,
  StatCard,
} from "../components/AppShell";
import { useAuth } from "../auth/AuthContext";
import { DAILY_TARGET, XP_CORRECT, XP_DAILY_COMPLETE } from "../game/constants";
import { useGame } from "../game/GameContext";
import {
  getContinueLetter,
  getMasteryPercent,
  lettersLearnedCount,
  lettersMasteredCount,
} from "../game/gamification";

function timeOfDayGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export function HomePage() {
  const { state, level, startPractice, startWordPractice, navigate } = useGame();
  const { user } = useAuth();
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null);
  const [wordRecs, setWordRecs] = useState<WordRecommendations | null>(null);
  const [weeklyAttempts, setWeeklyAttempts] = useState<number | null>(null);
  const daily = state.dailyChallenge;
  const continueLetter = getContinueLetter(state);
  const continueProgress = getMasteryPercent(state.letterStats[continueLetter].correct);
  const dailyPercent = (daily.progress / daily.target) * 100;
  const weakest = recommendations?.weakest;
  const recommendedLetter = recommendations?.recommended[0]?.letter ?? weakest?.letter ?? continueLetter;
  const learned = lettersLearnedCount(state);
  const mastered = lettersMasteredCount(state);

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
      {/* Hero: greeting + strong "continue learning" CTA */}
      <div>
        <p className="font-display text-xl text-[var(--color-ink-soft)] md:text-2xl">
          {timeOfDayGreeting()}, {user?.username ?? "there"} 👋
        </p>
        <h1 className="mt-1 font-display text-3xl font-semibold text-[var(--color-ink)] md:text-4xl">
          Ready for another sign?
        </h1>
      </div>

      <Card className="relative overflow-hidden !bg-[var(--color-primary)] !border-transparent">
        <div className="pointer-events-none absolute -right-10 -top-16 h-56 w-56 rounded-full bg-white/10" aria-hidden="true" />
        <div className="pointer-events-none absolute -bottom-20 -left-10 h-48 w-48 rounded-full bg-white/5" aria-hidden="true" />
        <div className="relative flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 items-center gap-5">
            <div className="grid h-20 w-20 shrink-0 place-items-center rounded-[var(--radius-panel)] bg-white font-display text-4xl font-bold text-[var(--color-primary)] shadow-lg">
              {continueLetter}
            </div>
            <div className="min-w-0 text-white">
              <p className="text-xs font-semibold uppercase tracking-wide text-white/70">Current lesson</p>
              <h3 className="mt-1 font-display text-2xl font-semibold">Master the {continueLetter} handshape</h3>
              <p className="mt-1 text-sm text-white/80">{Math.round(continueProgress)}% toward mastery · +{XP_CORRECT} XP available</p>
              <div className="mt-3 max-w-xs">
                <div className="h-2.5 overflow-hidden rounded-full bg-white/25">
                  <div
                    className="h-full rounded-full bg-white transition-all duration-500"
                    style={{ width: `${continueProgress}%` }}
                    role="progressbar"
                    aria-valuenow={Math.round(continueProgress)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-[var(--radius-control)] bg-white px-6 py-2.5 text-sm font-bold text-[var(--color-primary)] shadow-lg transition hover:brightness-95 active:scale-[0.98]"
              onClick={() => startPractice(continueLetter)}
              aria-label={`Continue practicing letter ${continueLetter}`}
            >
              Continue learning →
            </button>
            {weakest && weakest.letter !== continueLetter && (
              <button
                type="button"
                className="inline-flex min-h-11 items-center justify-center rounded-[var(--radius-control)] border-2 border-white/40 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-white/10"
                onClick={() => startPractice(weakest.letter)}
              >
                Practice {weakest.letter} instead
              </button>
            )}
          </div>
        </div>
      </Card>

      {wordRecs?.continue_word && (
        <Card className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-native)]">Continue spelling</p>
            <h3 className="mt-1 font-display text-xl font-semibold text-[var(--color-ink)]">{wordRecs.continue_word.word}</h3>
            <p className="mt-2 text-sm text-[var(--color-ink-soft)]">
              {(getWordById(wordRecs.continue_word.word_id)?.letters ?? wordRecs.continue_word.word.split("")).map((letter, idx) => (
                <span key={`${letter}-${idx}`} className="mr-3">
                  {letter} {idx < (wordRecs.continue_word?.resume_index ?? 0) ? "✅" : "○"}
                </span>
              ))}
            </p>
          </div>
          <PrimaryButton onClick={() => startWordPractice(wordRecs.continue_word!.word_id)}>Continue</PrimaryButton>
        </Card>
      )}

      {/* Today's goal */}
      <Card>
        <SectionHeader
          title="🎯 Today's goal"
          description={`Sign ${DAILY_TARGET} letters correctly${!daily.completed ? ` for +${XP_DAILY_COMPLETE} XP` : " — done for today!"}`}
        />
        <div className="mt-4">
          <ProgressBar percent={dailyPercent} shimmer={!daily.completed} size="md" showPercent={false} />
          <p className="mt-2 text-sm text-[var(--color-ink-soft)]">
            {daily.progress} / {daily.target} completed
          </p>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <PrimaryButton onClick={() => startPractice(null, "daily")}>
            {daily.completed ? "Keep practicing" : "Start daily practice"}
          </PrimaryButton>
          <SecondaryButton onClick={() => navigate("challenges")}>All missions</SecondaryButton>
        </div>
      </Card>

      {/* Your journey preview */}
      <Card>
        <SectionHeader
          title="🗺️ Your journey"
          description={`${learned} letters learned · ${mastered} mastered`}
          action={<SecondaryButton onClick={() => navigate("learn")}>View path →</SecondaryButton>}
        />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {"ABCDEFGHIJKLM".split("").map((letter) => {
            const correct = state.letterStats[letter]?.correct ?? 0;
            const status = correct >= 10 ? "mastered" : correct > 0 ? "started" : "locked";
            return (
              <span
                key={letter}
                className={`grid h-9 w-9 place-items-center rounded-full font-display text-sm font-semibold ${
                  status === "mastered"
                    ? "bg-[var(--color-success)] text-white"
                    : status === "started"
                      ? "bg-[var(--color-primary)] text-white"
                      : "bg-[var(--color-locked-soft)] text-[var(--color-muted)]"
                }`}
              >
                {letter}
              </span>
            );
          })}
          <span className="text-sm text-[var(--color-muted)]">…and more</span>
        </div>
      </Card>

      {/* Quick practice */}
      <div>
        <SectionHeader title="⚡ Quick practice" description="Jump straight into a learning mode." />
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <QuickPracticeCard
            icon="🔤"
            label="Alphabet"
            description="Sign a letter A–Z"
            tone="primary"
            onClick={() => startPractice(recommendedLetter)}
          />
          <QuickPracticeCard icon="🧩" label="Words" description="Spell a word" tone="native" onClick={() => navigate("words")} />
          <QuickPracticeCard
            icon="🎥"
            label="Native Signs"
            description="Full ASL signs"
            tone="streak"
            onClick={() => navigate("native")}
          />
        </div>
      </div>

      {recommendations?.has_data && recommendations.recommended.length > 0 && (
        <Card>
          <SectionHeader title="Recommended for you" description="Based on your recent practice." />
          <div className="mt-4 flex flex-wrap gap-2">
            {recommendations.recommended.slice(0, 3).map((item) => (
              <button
                key={item.letter}
                type="button"
                className="rounded-[var(--radius-control)] border-2 border-[var(--color-border)] px-4 py-2 text-sm font-semibold text-[var(--color-ink-soft)] transition hover:border-[var(--color-primary)] hover:text-[var(--color-primary)]"
                onClick={() => startPractice(item.letter)}
              >
                {item.letter}
              </button>
            ))}
          </div>
          {weakest && (
            <p className="mt-3 text-sm text-[var(--color-muted)]">
              Weakest practiced: {weakest.letter}
              {weakest.accuracy != null ? ` · ${Math.round(weakest.accuracy)}% accuracy` : ""}
            </p>
          )}
        </Card>
      )}

      {/* Streak / XP / achievements strip */}
      <div>
        <SectionHeader title="🏆 Your stats" />
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard icon="⭐" label="Level" value={level} tone="primary" />
          <StatCard icon="🔥" label="Streak" value={`${state.streak}d`} tone="streak" />
          <StatCard icon="🔤" label="Mastered" value={`${mastered}/26`} tone="success" />
          <StatCard icon="✋" label="Correct signs" value={state.totalCorrect} tone="native" />
        </div>
        {weeklyAttempts != null && (
          <p className="mt-3 text-sm text-[var(--color-muted)]">{weeklyAttempts} practice attempts this week</p>
        )}
      </div>

      {state.recentAchievements.length > 0 && (
        <Card>
          <SectionHeader
            title="🏅 Recent badges"
            action={
              <button
                type="button"
                className="text-sm font-semibold text-[var(--color-primary)] hover:underline"
                onClick={() => navigate("achievements")}
              >
                View all →
              </button>
            }
          />
          <ul className="mt-4 space-y-3">
            {state.recentAchievements.slice(0, 3).map((item) => (
              <li key={`${item.id}-${item.unlockedAt}`} className="flex items-center justify-between gap-3 text-sm">
                <span className="font-semibold text-[var(--color-ink)]">🏅 {item.title}</span>
                <span className="text-[var(--color-muted)]">{new Date(item.unlockedAt).toLocaleDateString()}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </PageLayout>
  );
}

function QuickPracticeCard({
  icon,
  label,
  description,
  tone,
  onClick,
}: {
  icon: string;
  label: string;
  description: string;
  tone: "primary" | "native" | "streak";
  onClick: () => void;
}) {
  const tones: Record<string, string> = {
    primary: "bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]",
    native: "bg-[var(--color-native-soft)] text-[var(--color-native)]",
    streak: "bg-[var(--color-streak-soft)] text-[var(--color-streak)]",
  };
  return (
    <button
      type="button"
      className="lift-hover flex flex-col items-start gap-2 rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface)] p-5 text-left shadow-[0_1px_2px_rgba(35,40,66,0.04),0_10px_28px_-16px_rgba(35,40,66,0.14)]"
      onClick={onClick}
    >
      <span className={`grid h-12 w-12 place-items-center rounded-2xl text-2xl ${tones[tone]}`} aria-hidden="true">
        {icon}
      </span>
      <span className="font-display text-base font-semibold text-[var(--color-ink)]">{label}</span>
      <span className="text-sm text-[var(--color-ink-soft)]">{description}</span>
    </button>
  );
}
