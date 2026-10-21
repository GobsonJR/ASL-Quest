import { useEffect, useState } from "react";
import {
  Card,
  Divider,
  EmptyState,
  PageHeader,
  PageLayout,
  PrimaryButton,
  SectionHeader,
  SectionLabel,
  StatGroup,
  StatPill,
} from "../components/AppShell";
import {
  fetchAnalyticsDashboard,
  fetchLetterDetail,
  fetchPracticeHistory,
  type AnalyticsDashboard,
  type AnalyticsRange,
  type HeatmapRange,
  type LetterDetail,
} from "../analytics/api";
import { fetchWord, type WordItem } from "../words/api";
import { ActivityHeatmap, DateRangeFilter, HeatmapRangeFilter, LineChart } from "../analytics/components/charts";
import { LetterDetailModal } from "../analytics/components/LetterDetailModal";
import { LETTERS } from "../game/constants";
import { useGame } from "../game/GameContext";
import { getNativeProgress, type NativeProgress } from "../native/api";

type SortMode = "alpha" | "accuracy_desc" | "accuracy_asc" | "attempts_desc";

export function ProgressPage() {
  const { startPractice, startWordPractice } = useGame();
  const [range, setRange] = useState<AnalyticsRange>("30d");
  const [heatmapRange, setHeatmapRange] = useState<HeatmapRange>("12m");
  const [sort, setSort] = useState<SortMode>("alpha");
  const [data, setData] = useState<AnalyticsDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLetter, setSelectedLetter] = useState<string | null>(null);
  const [letterDetail, setLetterDetail] = useState<LetterDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyLetter, setHistoryLetter] = useState("");
  const [historyResult, setHistoryResult] = useState<"" | "correct" | "incorrect">("");
  const [history, setHistory] = useState<Awaited<ReturnType<typeof fetchPracticeHistory>> | null>(null);
  const [selectedWordId, setSelectedWordId] = useState<string | null>(null);
  const [wordDetail, setWordDetail] = useState<WordItem | null>(null);
  const [nativeProgress, setNativeProgress] = useState<NativeProgress | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchAnalyticsDashboard(range, heatmapRange)
      .then(setData)
      .catch(() => setError("Unable to load analytics right now."))
      .finally(() => setLoading(false));
  }, [range, heatmapRange]);

  useEffect(() => {
    fetchPracticeHistory({
      page: historyPage,
      letter: historyLetter || undefined,
      result: historyResult || undefined,
    })
      .then(setHistory)
      .catch(() => setHistory(null));
  }, [historyPage, historyLetter, historyResult, data]);

  useEffect(() => {
    if (!selectedWordId) return;
    fetchWord(selectedWordId).then(setWordDetail).catch(() => setWordDetail(null));
  }, [selectedWordId]);

  useEffect(() => {
    // Native progress is a separate, independent fetch (not part of the analytics
    // dashboard payload) and never blocks the rest of this page — if it fails, the
    // section below simply doesn't render, same as the pattern in NativeSignsPage.
    getNativeProgress()
      .then(setNativeProgress)
      .catch(() => setNativeProgress(null));
  }, []);

  useEffect(() => {
    if (!selectedLetter) return;
    setDetailLoading(true);
    fetchLetterDetail(selectedLetter)
      .then(setLetterDetail)
      .finally(() => setDetailLoading(false));
  }, [selectedLetter]);

  if (loading) {
    return <div className="grid min-h-[40vh] place-items-center text-[var(--color-ink-soft)]">Loading learning analytics...</div>;
  }

  if (error || !data) {
    return (
      <EmptyState
        icon="📊"
        title="Analytics unavailable"
        description={error ?? "Try again in a moment."}
        action={
          <PrimaryButton onClick={() => window.location.reload()}>Retry</PrimaryButton>
        }
      />
    );
  }

  if (!data.overview.has_data) {
    return (
      <EmptyState
        icon="📈"
        title="Your analytics are waiting for you"
        description="Complete your first practice session to start building your learning history."
        action={<PrimaryButton onClick={() => startPractice()}>Start Practicing →</PrimaryButton>}
      />
    );
  }

  const sortedLetters = [...data.letters].sort((a, b) => {
    if (sort === "accuracy_desc") return (b.accuracy ?? -1) - (a.accuracy ?? -1);
    if (sort === "accuracy_asc") return (a.accuracy ?? 101) - (b.accuracy ?? 101);
    if (sort === "attempts_desc") return b.attempts - a.attempts;
    return a.letter.localeCompare(b.letter);
  });

  return (
    <PageLayout>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <PageHeader
          eyebrow="📈 Your journey"
          title="Your learning progress"
          description="Every metric below comes from your saved practice history."
        />
        <DateRangeFilter value={range} onChange={setRange} />
      </div>

      <StatGroup
        items={[
          { label: "Total practice", value: data.overview.total_attempts, sub: `${data.overview.total_correct} correct` },
          { label: "Accuracy", value: data.overview.overall_accuracy != null ? `${data.overview.overall_accuracy}%` : "—" },
          { label: "Mastered", value: `${data.overview.letters_mastered} / 26`, sub: `${data.overview.letters_practiced} practiced` },
          { label: "Streak", value: `${data.overview.current_streak} days`, sub: `Best ${data.overview.best_streak} days` },
          { label: "Total XP", value: data.overview.total_xp, sub: `+${data.xp.xp_this_week} this week` },
          { label: "Avg response", value: data.overview.average_response_sec != null ? `${data.overview.average_response_sec}s` : "—" },
        ]}
      />

      {data.words?.has_data && (
        <>
          <Divider />
          <SectionHeader title="Word spelling" description="Sequential alphabet spelling — not whole-word sign recognition." />
          <StatGroup
            items={[
              { label: "Words learned", value: data.words.words_learned },
              { label: "Words completed", value: data.words.words_completed },
              { label: "Word accuracy", value: data.words.word_accuracy != null ? `${data.words.word_accuracy}%` : "—" },
              { label: "Word practice time", value: `${data.words.word_practice_time_sec}s` },
            ]}
          />
          <div className="grid gap-6 lg:grid-cols-3">
            <Card>
              <SectionLabel>Most practiced words</SectionLabel>
              <ul className="mt-3 space-y-2 text-sm">
                {data.words.most_practiced.map((item) => (
                  <li key={item.word_id}>
                    <button type="button" className="flex w-full justify-between text-left" onClick={() => setSelectedWordId(item.word_id)}>
                      <span>{item.word}</span>
                      <span className="text-[var(--color-muted)]">{item.attempts} attempts</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
            <Card>
              <SectionLabel>Strongest words</SectionLabel>
              <ul className="mt-3 space-y-2 text-sm">
                {data.words.strongest.map((item) => (
                  <li key={item.word_id} className="flex justify-between">
                    <span>{item.word}</span>
                    <span>{item.accuracy != null ? `${item.accuracy}%` : "—"}</span>
                  </li>
                ))}
              </ul>
            </Card>
            <Card>
              <SectionLabel>Words needing practice</SectionLabel>
              <ul className="mt-3 space-y-2 text-sm">
                {data.words.needing_practice.map((item) => (
                  <li key={item.word_id}>
                    <button type="button" className="flex w-full justify-between text-left" onClick={() => startWordPractice(item.word_id)}>
                      <span>{item.word}</span>
                      <span className="text-[var(--color-streak)]">{item.accuracy != null ? `${item.accuracy}%` : "—"}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
          <Card>
            <SectionLabel>Category progress</SectionLabel>
            <div className="mt-4 space-y-3">
              {data.words.category_progress.map((item) => (
                <div key={item.category} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3 text-sm">
                  <span>{item.category}</span>
                  <div className="h-2 overflow-hidden rounded-full bg-[var(--color-surface-sunken)]">
                    <div className="h-full rounded-full bg-[var(--color-primary)]" style={{ width: `${item.percent}%` }} />
                  </div>
                  <span className="text-right text-[var(--color-muted)]">{item.percent}%</span>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      {nativeProgress && nativeProgress.total_signs > 0 && (
        <>
          <Divider />
          <SectionHeader
            title="Native signs"
            description="Isolated native ASL sign recognition — a separate skill from A-Z and Word Spelling."
          />
          <StatGroup
            items={[
              { label: "Signs started", value: `${nativeProgress.started_signs} / ${nativeProgress.total_signs}` },
              { label: "Signs mastered", value: `${nativeProgress.mastered_signs} / ${nativeProgress.total_signs}` },
              { label: "Overall mastery", value: `${nativeProgress.overall_mastery}%` },
            ]}
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {nativeProgress.signs
              .filter((entry) => entry.attempts > 0)
              .map((entry) => (
                <div
                  key={entry.sign_id}
                  className="rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{entry.display_name}</span>
                    <span className="text-[var(--color-muted)]">{entry.mastery}%</span>
                  </div>
                  <p className="mt-0.5 text-xs text-[var(--color-muted)]">
                    {entry.correct} / {entry.attempts} correct
                  </p>
                </div>
              ))}
          </div>
          {nativeProgress.started_signs === 0 && (
            <p className="text-sm text-[var(--color-ink-soft)]">
              Practice a native sign to start building your native-sign progress.
            </p>
          )}
        </>
      )}

      <Divider />

      <Card>
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <SectionLabel>Activity Heatmap</SectionLabel>
          <HeatmapRangeFilter value={heatmapRange} onChange={setHeatmapRange} />
        </div>
        <ActivityHeatmap cells={data.heatmap.cells} />
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <LineChart
            title="Practice Activity"
            points={data.activity_trend.points}
            emptyMessage="No practice activity in this range yet."
          />
        </Card>
        <Card>
          <LineChart
            title="Accuracy Over Time"
            points={data.accuracy_trend.points}
            valueSuffix="%"
            emptyMessage="Accuracy trends appear after a few practice sessions."
          />
          {data.accuracy_trend.delta != null && (
            <p className="mt-3 text-sm text-[var(--color-ink-soft)]">
              {data.accuracy_trend.delta >= 0 ? "↑" : "↓"} {Math.abs(data.accuracy_trend.delta)} percentage points vs previous period
            </p>
          )}
        </Card>
      </div>

      <Card>
        <SectionHeader title="This week" />
        <div className="mt-4">
          <StatGroup
            items={[
              { label: "Practice", value: data.weekly_summary.current_week.attempts },
              { label: "Accuracy", value: data.weekly_summary.current_week.accuracy ?? "—" },
              { label: "XP", value: data.weekly_summary.current_week.xp },
              { label: "Letters", value: data.weekly_summary.current_week.letters_practiced },
              {
                label: "Avg response",
                value:
                  data.weekly_summary.current_week.average_response_sec != null
                    ? `${data.weekly_summary.current_week.average_response_sec}s`
                    : "—",
              },
            ]}
          />
        </div>
      </Card>

      <Card>
        <SectionHeader title="Learning funnel" />
        <div className="mt-4">
          <StatGroup
            items={[
              { label: "26 letters", value: data.funnel.total_letters },
              { label: "Practiced", value: data.funnel.practiced },
              { label: "Learning", value: data.funnel.learning },
              { label: "Proficient", value: data.funnel.proficient },
              { label: "Mastered", value: data.funnel.mastered },
            ]}
          />
        </div>
      </Card>

      <Card>
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <SectionLabel>A-Z Mastery Grid</SectionLabel>
          <select
            className="rounded-full border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-3 py-1.5 text-xs"
            value={sort}
            onChange={(event) => setSort(event.target.value as SortMode)}
            aria-label="Sort letters"
          >
            <option value="alpha">Alphabetical</option>
            <option value="accuracy_desc">Highest accuracy</option>
            <option value="accuracy_asc">Lowest accuracy</option>
            <option value="attempts_desc">Most practiced</option>
          </select>
        </div>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-7 lg:grid-cols-[repeat(13,minmax(0,1fr))]">
          {sortedLetters.map((letter) => (
            <button
              key={letter.letter}
              type="button"
              className="rounded-[var(--radius-control)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] p-3 text-left transition hover:border-[var(--color-primary)]/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-primary)]"
              onClick={() => setSelectedLetter(letter.letter)}
              aria-label={`Letter ${letter.letter}, ${letter.status}, ${letter.mastery_percent}% mastery`}
            >
              <span className="font-display text-xl text-[var(--color-primary)]">{letter.letter}</span>
              <p className="mt-1 text-[0.65rem] uppercase text-[var(--color-muted)]">{letter.status}</p>
              <p className="text-xs font-semibold">{letter.practiced ? `${letter.mastery_percent}%` : "Not practiced"}</p>
            </button>
          ))}
        </div>
      </Card>

      <Card>
        <SectionLabel>Accuracy by Letter</SectionLabel>
        <div className="mt-4 space-y-2">
          {sortedLetters.map((letter) => (
            <div key={letter.letter} className="grid grid-cols-[2rem_1fr_4rem] items-center gap-3">
              <span className="font-display text-lg text-[var(--color-primary)]">{letter.letter}</span>
              <div className="h-2 overflow-hidden rounded-full bg-[var(--color-surface-sunken)]">
                <div
                  className="h-full rounded-full bg-[var(--color-primary)]"
                  style={{ width: `${letter.accuracy ?? 0}%` }}
                />
              </div>
              <span className="text-right text-sm text-[var(--color-ink-soft)]">
                {letter.practiced && letter.accuracy != null ? `${letter.accuracy}%` : "Not practiced"}
              </span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionLabel>Your Strongest</SectionLabel>
          <div className="mt-4 space-y-2">
            {data.insights.strongest.map((item) => (
              <div key={item.letter} className="flex justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2">
                <span>{item.letter}</span>
                <span className="font-semibold text-[var(--color-success)]">{item.accuracy}%</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <SectionLabel>Needs Practice</SectionLabel>
          <div className="mt-4 space-y-2">
            {data.insights.weakest.map((item) => (
              <div key={item.letter} className="flex justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2">
                <span>{item.letter}</span>
                <span className="font-semibold text-[var(--color-streak)]">{item.accuracy}%</span>
              </div>
            ))}
          </div>
          <PrimaryButton
            className="mt-4"
            onClick={() => {
              const weak = data.insights.weakest[0]?.letter;
              if (weak) startPractice(weak);
            }}
          >
            Practice weak letters
          </PrimaryButton>
        </Card>
      </div>

      <Card>
        <SectionLabel>Response Time</SectionLabel>
        {data.response_time.has_data ? (
          <>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <StatPill icon="⏱️" label="Average" value={data.response_time.average_sec != null ? `${data.response_time.average_sec}s` : "—"} />
              <StatPill icon="⚡" label="Fastest" value={data.response_time.fastest_sec != null ? `${data.response_time.fastest_sec}s` : "—"} />
            </div>
            <div className="mt-6">
              <LineChart
                title="Average Response Time"
                points={data.response_time.trend.map((point) => ({ date: point.date, value: point.seconds }))}
                valueSuffix="s"
                emptyMessage="Keep practicing to unlock response-time trends."
              />
            </div>
          </>
        ) : (
          <p className="mt-4 text-sm text-[var(--color-ink-soft)]">Keep practicing to unlock response-time trends.</p>
        )}
      </Card>

      <Card>
        <SectionLabel>XP Progress</SectionLabel>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <StatPill icon="⭐" label="Total XP" value={data.xp.total_xp} />
          <StatPill icon="📅" label="This Week" value={data.xp.xp_this_week} />
          <StatPill icon="🗓️" label="This Month" value={data.xp.xp_this_month} />
        </div>
        <div className="mt-6">
          <LineChart
            title="XP earned over time"
            points={data.xp.trend}
            emptyMessage="XP history appears after you earn XP from practice."
          />
        </div>
      </Card>

      <Card>
        <SectionLabel>Practice History</SectionLabel>
        <div className="mt-4 flex flex-wrap gap-2">
          <select
            className="rounded-full border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-3 py-1.5 text-xs"
            value={historyLetter}
            onChange={(event) => {
              setHistoryPage(1);
              setHistoryLetter(event.target.value);
            }}
          >
            <option value="">All letters</option>
            {LETTERS.map((letter) => (
              <option key={letter} value={letter}>{letter}</option>
            ))}
          </select>
          <select
            className="rounded-full border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-3 py-1.5 text-xs"
            value={historyResult}
            onChange={(event) => {
              setHistoryPage(1);
              setHistoryResult(event.target.value as "" | "correct" | "incorrect");
            }}
          >
            <option value="">All results</option>
            <option value="correct">Correct</option>
            <option value="incorrect">Incorrect</option>
          </select>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-[var(--color-muted)]">
              <tr>
                <th className="px-2 py-2">Date</th>
                <th className="px-2 py-2">Letter</th>
                <th className="px-2 py-2">Prediction</th>
                <th className="px-2 py-2">Result</th>
                <th className="px-2 py-2">Response</th>
                <th className="px-2 py-2">XP</th>
              </tr>
            </thead>
            <tbody>
              {history?.items.map((item) => (
                <tr key={item.id} className="border-t border-[var(--color-border-soft)]">
                  <td className="px-2 py-2">{new Date(item.date).toLocaleString()}</td>
                  <td className="px-2 py-2">{item.letter}</td>
                  <td className="px-2 py-2">{item.prediction ?? "—"}</td>
                  <td className="px-2 py-2">{item.correct ? "✓ Correct" : "✕ Incorrect"}</td>
                  <td className="px-2 py-2">{item.response_time_sec != null ? `${item.response_time_sec}s` : "—"}</td>
                  <td className="px-2 py-2">{item.xp_earned > 0 ? `+${item.xp_earned}` : "0"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {history && history.total > history.page_size && (
          <div className="mt-4 flex gap-2">
            <PrimaryButton disabled={historyPage <= 1} onClick={() => setHistoryPage((page) => page - 1)}>
              Previous
            </PrimaryButton>
            <PrimaryButton
              disabled={historyPage * history.page_size >= history.total}
              onClick={() => setHistoryPage((page) => page + 1)}
            >
              Next
            </PrimaryButton>
          </div>
        )}
      </Card>

      {selectedLetter && (
        <LetterDetailModal
          detail={letterDetail}
          loading={detailLoading}
          onClose={() => {
            setSelectedLetter(null);
            setLetterDetail(null);
          }}
          onPractice={(letter) => {
            setSelectedLetter(null);
            startPractice(letter);
          }}
        />
      )}

      {selectedWordId && wordDetail?.progress && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 px-4" role="dialog" aria-modal="true">
          <Card className="w-full max-w-lg">
            <div className="flex items-start justify-between">
              <div>
                <SectionLabel>Word detail</SectionLabel>
                <h2 className="mt-2 font-display text-4xl">{wordDetail.word}</h2>
              </div>
              <button type="button" className="text-sm text-[var(--color-ink-soft)]" onClick={() => { setSelectedWordId(null); setWordDetail(null); }}>
                Close
              </button>
            </div>
            <StatGroup
              className="mt-5"
              items={[
                { label: "Attempts", value: wordDetail.progress.attempts },
                { label: "Completions", value: wordDetail.progress.completions },
                { label: "Accuracy", value: wordDetail.progress.accuracy != null ? `${wordDetail.progress.accuracy}%` : "—" },
                { label: "Best time", value: wordDetail.progress.best_time_sec != null ? `${wordDetail.progress.best_time_sec}s` : "—" },
                { label: "Mastery", value: `${wordDetail.progress.mastery}%` },
              ]}
            />
            <SectionLabel>Letter breakdown</SectionLabel>
            <ul className="mt-3 space-y-2 text-sm">
              {wordDetail.progress.letter_breakdown.map((item, index) => (
                <li key={`${item.letter}-${index}`} className="flex justify-between">
                  <span>
                    {item.letter} {item.practiced ? "✓" : "○"}
                  </span>
                  <span>{item.accuracy != null ? `${item.accuracy}%` : "—"}</span>
                </li>
              ))}
            </ul>
            <PrimaryButton className="mt-5" onClick={() => { setSelectedWordId(null); startWordPractice(wordDetail.id); }}>
              Practice {wordDetail.word}
            </PrimaryButton>
          </Card>
        </div>
      )}
    </PageLayout>
  );
}
