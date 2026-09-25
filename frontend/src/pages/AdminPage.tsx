import { useEffect, useState } from "react";
import {
  fetchAdminActivity,
  fetchAdminLetters,
  fetchAdminOverview,
  fetchAdminRecentActivity,
  fetchAdminUsers,
  fetchAdminWords,
} from "../auth/api";
import { useAuth } from "../auth/AuthContext";
import { Card, EmptyState, PageHeader, SectionLabel, StatPill } from "../components/AppShell";
import { useGame } from "../game/GameContext";

export function AdminPage() {
  const { user } = useAuth();
  const { navigate } = useGame();
  const [overview, setOverview] = useState<Awaited<ReturnType<typeof fetchAdminOverview>> | null>(null);
  const [users, setUsers] = useState<Awaited<ReturnType<typeof fetchAdminUsers>> | null>(null);
  const [letters, setLetters] = useState<Awaited<ReturnType<typeof fetchAdminLetters>> | null>(null);
  const [activity, setActivity] = useState<Awaited<ReturnType<typeof fetchAdminActivity>> | null>(null);
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof fetchAdminRecentActivity>> | null>(null);
  const [words, setWords] = useState<Awaited<ReturnType<typeof fetchAdminWords>> | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user?.role !== "admin") {
      navigate("home");
      return;
    }
    Promise.all([
      fetchAdminOverview(),
      fetchAdminUsers({ page: 1 }),
      fetchAdminLetters(),
      fetchAdminActivity(),
      fetchAdminRecentActivity(),
      fetchAdminWords(),
    ])
      .then(([overviewData, usersData, lettersData, activityData, recentData, wordsData]) => {
        setOverview(overviewData);
        setUsers(usersData);
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

  if (!overview || !users || !letters || !activity || !recent) {
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

      <Card>
        <SectionLabel>User management</SectionLabel>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row">
          <input
            className="w-full rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3"
            placeholder="Search username or email"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <button
            type="button"
            className="rounded-full bg-[var(--color-primary)] px-6 py-3 text-sm font-bold text-white"
            onClick={() => fetchAdminUsers({ page: 1, search }).then(setUsers)}
          >
            Search
          </button>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-[var(--color-muted)]">
              <tr>
                <th className="px-2 py-2">Username</th>
                <th className="px-2 py-2">Email</th>
                <th className="px-2 py-2">Role</th>
                <th className="px-2 py-2">Sessions</th>
                <th className="px-2 py-2">XP</th>
                <th className="px-2 py-2">Last activity</th>
              </tr>
            </thead>
            <tbody>
              {users.items.map((item) => (
                <tr key={item.id} className="border-t border-[var(--color-border-soft)]">
                  <td className="px-2 py-2 font-semibold">{item.username}</td>
                  <td className="px-2 py-2">{item.email}</td>
                  <td className="px-2 py-2">{item.role}</td>
                  <td className="px-2 py-2">{item.practice_sessions}</td>
                  <td className="px-2 py-2">{item.xp}</td>
                  <td className="px-2 py-2">{item.last_activity ? new Date(item.last_activity).toLocaleDateString() : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--color-muted)]">
          Showing page {users.page} · {users.total} users total
        </p>
      </Card>

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
