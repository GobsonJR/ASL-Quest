import { useEffect, useState } from "react";
import {
  fetchAdminActivity,
  fetchAdminLetters,
  fetchAdminOverview,
  fetchAdminRecentActivity,
  fetchAdminWords,
} from "../auth/api";
import { useAuth } from "../auth/AuthContext";
import { Card, EmptyState, PageHeader, SectionLabel, StatPill } from "../components/AppShell";
import { useGame } from "../game/GameContext";
import { AuditTab } from "../admin/AuditTab";
import { ContentTab } from "../admin/ContentTab";
import { CollectPage } from "../collect/CollectPage";
import { DatasetTab } from "../admin/DatasetTab";
import { ModelsTab } from "../admin/ModelsTab";
import { UsersTab } from "../admin/UsersTab";

function AdminOverview() {
  const { user } = useAuth();
  const { navigate } = useGame();
  const [overview, setOverview] = useState<Awaited<ReturnType<typeof fetchAdminOverview>> | null>(null);
  const [letters, setLetters] = useState<Awaited<ReturnType<typeof fetchAdminLetters>> | null>(null);
  const [activity, setActivity] = useState<Awaited<ReturnType<typeof fetchAdminActivity>> | null>(null);
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof fetchAdminRecentActivity>> | null>(null);
  const [words, setWords] = useState<Awaited<ReturnType<typeof fetchAdminWords>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user?.role !== "admin") {
      navigate("home");
      return;
    }
    Promise.all([
      fetchAdminOverview(),
      fetchAdminLetters(),
      fetchAdminActivity(),
      fetchAdminRecentActivity(),
      fetchAdminWords(),
    ])
      .then(([overviewData, lettersData, activityData, recentData, wordsData]) => {
        setOverview(overviewData);
        setLetters(lettersData);
        setActivity(activityData);
        setRecent(recentData);
        setWords(wordsData);
      })
      .catch(() => setError("Unable to load admin dashboard."));
  }, [user, navigate]);

  if (user?.role !== "admin") {
    return (
      <EmptyState
        icon="🔒"
        title="Admin access required"
        description="This area is restricted to administrators."
      />
    );
  }

  if (error) {
    return <EmptyState icon="⚠️" title="Admin dashboard unavailable" description={error} />;
  }

  if (!overview || !letters || !activity || !recent) {
    return <div className="grid min-h-[40vh] place-items-center text-[var(--color-ink-soft)]">Loading admin dashboard...</div>;
  }

  return (
    <div className="space-y-6">
      <Card>
        <PageHeader
          eyebrow="Admin"
          title="System overview"
          description="Aggregate learner activity from the database. Individual student analytics stay private."
        />
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <StatPill icon="👥" label="Total users" value={overview.total_users} />
        <StatPill icon="✅" label="Active users" value={overview.active_users} />
        <StatPill icon="🎮" label="Practice sessions" value={overview.total_practice_sessions} />
        <StatPill icon="⭐" label="Total XP awarded" value={overview.total_xp_awarded} />
        <StatPill icon="🎯" label="Average accuracy" value={overview.average_accuracy != null ? `${overview.average_accuracy}%` : "—"} />
        <StatPill icon="🔤" label="Letter attempts" value={overview.total_letter_attempts} />
        <StatPill icon="📝" label="Word sessions" value={overview.total_word_practice_sessions ?? 0} />
        <StatPill icon="✅" label="Words completed" value={overview.words_completed ?? 0} />
        <StatPill icon="🎯" label="Word accuracy" value={overview.average_word_accuracy != null ? `${overview.average_word_accuracy}%` : "—"} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionLabel>Popular letters</SectionLabel>
          <div className="mt-4 space-y-2">
            {letters.popular.slice(0, 8).map((item) => (
              <div key={item.letter} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                <span className="font-semibold">{item.letter}</span>
                <span className="text-[var(--color-muted)]">{item.attempts} attempts</span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <SectionLabel>Difficult letters</SectionLabel>
          <div className="mt-4 space-y-2">
            {letters.difficult.slice(0, 8).map((item) => (
              <div key={item.letter} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                <span className="font-semibold">{item.letter}</span>
                <span className="text-[var(--color-muted)]">{item.accuracy != null ? `${item.accuracy}%` : "—"}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {words && (
        <div className="grid gap-6 lg:grid-cols-3">
          <Card>
            <SectionLabel>Most practiced words</SectionLabel>
            <div className="mt-4 space-y-2">
              {words.most_practiced.slice(0, 8).map((item) => (
                <div key={item.word} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                  <span className="font-semibold">{item.word}</span>
                  <span className="text-[var(--color-muted)]">{item.sessions} sessions</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <SectionLabel>Hardest words</SectionLabel>
            <div className="mt-4 space-y-2">
              {words.hardest.slice(0, 8).map((item) => (
                <div key={item.word} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                  <span className="font-semibold">{item.word}</span>
                  <span className="text-[var(--color-muted)]">{item.accuracy != null ? `${item.accuracy}%` : "—"}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <SectionLabel>Popular categories</SectionLabel>
            <div className="mt-4 space-y-2">
              {words.most_popular_categories.map((item) => (
                <div key={item.category} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                  <span>{item.category}</span>
                  <span className="text-[var(--color-muted)]">{item.sessions}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionLabel>System activity</SectionLabel>
          <div className="mt-4 space-y-2">
            {activity.points.slice(-7).map((point) => (
              <div key={point.date} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                <span>{point.date}</span>
                <span className="font-semibold">{point.value} attempts</span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <SectionLabel>Recent learner activity</SectionLabel>
          <div className="mt-4 space-y-2">
            {recent.items.slice(0, 10).map((item, index) => (
              <div key={`${item.user_id}-${item.created_at}-${index}`} className="flex items-center justify-between rounded-xl border border-[var(--color-border-soft)] px-3 py-2 text-sm">
                <span>
                  User #{item.user_id} · {item.letter}
                </span>
                <span className={item.correct ? "text-[var(--color-success)]" : "text-[var(--color-streak)]"}>
                  {item.correct ? "Correct" : "Incorrect"}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "users", label: "Users" },
  { id: "content", label: "Content" },
  { id: "dataset", label: "Dataset" },
  { id: "collection", label: "Collect data" },
  { id: "models", label: "Models & chatbot" },
  { id: "audit", label: "Audit log" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export function AdminPage() {
  const { user } = useAuth();
  const [tab, setTab] = useState<TabId>("overview");

  if (user?.role !== "admin") {
    return <EmptyState icon="🔒" title="Admin access required" description="This area is restricted to administrators." />;
  }

  return (
    <div className="space-y-6">
      <div role="tablist" aria-label="Admin sections" className="flex flex-wrap gap-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`min-h-10 rounded-full border-2 px-4 text-sm font-bold ${tab === t.id ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]" : "border-[var(--color-border)]"}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tab === "overview" && <AdminOverview />}
      {tab === "users" && <UsersTab />}
      {tab === "content" && <ContentTab />}
      {tab === "dataset" && <DatasetTab onOpenCollection={() => setTab("collection")} />}
      {tab === "collection" && <CollectPage />}
      {tab === "models" && <ModelsTab />}
      {tab === "audit" && <AuditTab />}
    </div>
  );
}
