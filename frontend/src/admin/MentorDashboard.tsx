import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, EmptyState, PageHeader, PageLayout, ProgressBar, SectionHeader, StatusChip } from "../components/AppShell";
import { useToast } from "../components/Toast";
import { useAuth } from "../auth/AuthContext";
import { useGame } from "../game/GameContext";
import { fetchMentorDashboard, type MentorDashboard, type MentorStudent } from "./api";

// Validated (scripts/validate_palette.js, all-adjacent-pairs PASS on this app's
// dark panel surface) fixed categorical order — never cycled/reassigned per
// filter, extra categories fold into "Other". Kept separate from the bright
// UI accent tokens (--color-accent etc.), which are too light to clear the
// chart-mark lightness band, so these are chart-only steps in the same hue
// family as the brand where possible.
const CATEGORY_COLORS = ["#7a9a30", "#9085e9", "#d95926", "#199e70", "#c98500"];
const OTHER_COLOR = "#5b6b63";
const CHART_TEXT = "#8fa399";
const CHART_GRID = "#223028";
const ACCENT = "#d7f36a";
const SUCCESS = "#6ee7a8";
const WARM = "#f3c16b";

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function useCountUp(target: number, durationMs = 900): number {
  const [value, setValue] = useState(prefersReducedMotion() ? target : 0);
  useEffect(() => {
    if (prefersReducedMotion()) {
      setValue(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(target * eased));
      if (progress < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);
  return value;
}

function foldIntoOther<T extends { attempts?: number; sessions?: number }>(
  items: T[],
  labelKey: keyof T,
  valueKey: "attempts" | "sessions",
  max = 5
): Array<{ label: string; value: number }> {
  const sorted = [...items].sort((a, b) => (b[valueKey] as number) - (a[valueKey] as number));
  const head = sorted.slice(0, max).map((item) => ({ label: String(item[labelKey]), value: item[valueKey] as number }));
  const rest = sorted.slice(max);
  const otherTotal = rest.reduce((sum, item) => sum + (item[valueKey] as number), 0);
  return otherTotal > 0 ? [...head, { label: "Other", value: otherTotal }] : head;
}

export function MentorDashboard() {
  const { user } = useAuth();
  const { navigate } = useGame();
  const { push } = useToast();
  const [data, setData] = useState<MentorDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user?.role !== "admin") {
      navigate("home");
      return;
    }
    setLoading(true);
    setError(null);
    fetchMentorDashboard()
      .then(setData)
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Unable to load the mentor dashboard.";
        setError(message);
        push(message, "error");
      })
      .finally(() => setLoading(false));
  }, [user, navigate, push]);

  if (user?.role !== "admin") {
    return (
      <EmptyState icon="🔒" title="Mentor access required" description="This dashboard is restricted to administrators." />
    );
  }

  if (loading) {
    return (
      <PageLayout>
        <div className="grid min-h-[40vh] place-items-center text-[var(--color-mist)]">Loading mentor analytics...</div>
      </PageLayout>
    );
  }

  if (error || !data) {
    return (
      <PageLayout>
        <EmptyState icon="⚠️" title="Unable to load the mentor dashboard" description={error ?? "Unknown error."} />
      </PageLayout>
    );
  }

  const animate = !prefersReducedMotion();

  return (
    <PageLayout>
      <HeroSection />

      <PlatformOverview overview={data.overview} />

      <section className="space-y-4">
        <SectionHeader title="Student progress" description="Every registered learner, most active first." />
        <StudentTable students={data.students} />
      </section>

      <section className="space-y-4">
        <SectionHeader title="A-Z analytics" description="Static alphabet handshape recognition." />
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Most practiced letters</p>
            <MagnitudeBarChart
              rows={data.az.popular.slice(0, 10).map((r) => ({ label: r.letter, value: r.attempts }))}
              valueLabel="attempts"
              animate={animate}
            />
          </Card>
          <Card>
            <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Most difficult letters (lowest accuracy)</p>
            {data.az.difficult.length === 0 ? (
              <EmptyChartMessage text="No accuracy data yet." />
            ) : (
              <ul className="space-y-2">
                {data.az.difficult.slice(0, 8).map((item) => (
                  <li key={item.letter} className="flex items-center gap-3">
                    <span className="w-6 shrink-0 font-display text-lg text-[#eef4f0]">{item.letter}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--color-ink)]">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${item.accuracy ?? 0}%`, background: WARM }}
                      />
                    </div>
                    <span className="w-12 shrink-0 text-right text-xs text-[var(--color-muted)]">{item.accuracy}%</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </section>

      <section className="space-y-4">
        <SectionHeader title="Word spelling" description="Sequential alphabet-sign spelling of whole words." />
        <div className="grid gap-4 sm:grid-cols-3">
          <StatTile label="Practice sessions" value={data.words.total_word_practice_sessions} animate={animate} />
          <StatTile
            label="Completion rate"
            value={
              data.words.total_word_practice_sessions
                ? Math.round((data.words.words_completed / data.words.total_word_practice_sessions) * 100)
                : 0
            }
            suffix="%"
            animate={animate}
          />
          <StatTile
            label="Average accuracy"
            value={data.words.average_word_accuracy ?? 0}
            suffix={data.words.average_word_accuracy != null ? "%" : ""}
            animate={animate}
          />
        </div>
        <Card>
          <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Most practiced words</p>
          <MagnitudeBarChart
            rows={data.words.most_practiced.slice(0, 8).map((r) => ({ label: r.word, value: r.sessions }))}
            valueLabel="sessions"
            animate={animate}
          />
        </Card>
      </section>

      <section className="space-y-4">
        <SectionHeader title="Native signs" description="Isolated native ASL sign recognition (I3D)." />
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Attempts per sign</p>
            <MagnitudeBarChart
              rows={data.native.per_sign.map((r) => ({ label: r.display_name, value: r.attempts }))}
              valueLabel="attempts"
              animate={animate}
            />
          </Card>
          <Card>
            <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Category distribution (attempts)</p>
            {data.native.category_distribution.length === 0 ? (
              <EmptyChartMessage text="No native sign attempts yet." />
            ) : (
              <CategoryPieChart
                items={foldIntoOther(data.native.category_distribution, "category", "attempts")}
                animate={animate}
              />
            )}
          </Card>
        </div>
        <Card>
          <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Mastery % by sign</p>
          <ul className="grid gap-3 sm:grid-cols-2">
            {data.native.per_sign.map((sign) => (
              <li key={sign.sign_id} className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-[#eef4f0]">{sign.display_name}</span>
                  <span className="text-[var(--color-muted)]">{sign.mastery != null ? `${sign.mastery}%` : "—"}</span>
                </div>
                <ProgressBar percent={sign.mastery ?? 0} showPercent={false} size="sm" />
              </li>
            ))}
          </ul>
        </Card>
      </section>

      <section className="space-y-4">
        <SectionHeader title="ASL-Quest Assistant" description="Project-only chatbot usage." />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile label="Conversations" value={data.chatbot.total_conversations} animate={animate} />
          <StatTile label="Messages sent" value={data.chatbot.total_messages} animate={animate} />
          <StatTile label="Helpful votes" value={data.chatbot.positive_feedback} tone="success" animate={animate} />
          <StatTile label="Not helpful votes" value={data.chatbot.negative_feedback} tone="warning" animate={animate} />
        </div>
        {data.chatbot.positive_feedback + data.chatbot.negative_feedback > 0 && (
          <Card>
            <p className="mb-3 text-sm font-semibold text-[#eef4f0]">Feedback breakdown</p>
            <div className="flex h-3 overflow-hidden rounded-full bg-[var(--color-ink)]">
              <div
                className="h-full"
                style={{
                  width: `${(data.chatbot.positive_feedback / (data.chatbot.positive_feedback + data.chatbot.negative_feedback)) * 100}%`,
                  background: SUCCESS,
                }}
              />
              <div
                className="h-full"
                style={{
                  width: `${(data.chatbot.negative_feedback / (data.chatbot.positive_feedback + data.chatbot.negative_feedback)) * 100}%`,
                  background: WARM,
                }}
              />
            </div>
            <div className="mt-2 flex gap-4 text-xs text-[var(--color-muted)]">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: SUCCESS }} /> Helpful
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: WARM }} /> Not helpful
              </span>
            </div>
          </Card>
        )}
      </section>
    </PageLayout>
  );
}

function HeroSection() {
  return (
    <div className="glass-panel relative overflow-hidden rounded-[var(--radius-panel)] p-6 md:p-8">
      <div
        className="pointer-events-none absolute inset-0 opacity-70"
        style={{
          background:
            "radial-gradient(600px 260px at 15% -10%, rgba(215,243,106,0.16) 0%, transparent 55%), radial-gradient(500px 220px at 100% 10%, rgba(144,133,233,0.12) 0%, transparent 50%)",
        }}
        aria-hidden="true"
      />
      <div className="relative">
        <StatusChip tone="accent">MENTOR ANALYTICS</StatusChip>
        <PageHeader
          title="Mentor Dashboard"
          description="A read-only view of real platform activity across A-Z, Word Spelling, Native Signs, and the ASL-Quest Assistant."
          compact
        />
      </div>
    </div>
  );
}

function PlatformOverview({ overview }: { overview: MentorDashboard["overview"] }) {
  const animate = !prefersReducedMotion();
  return (
    <section className="space-y-4">
      <SectionHeader title="Platform overview" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Total users" value={overview.total_users} animate={animate} icon="👥" />
        <StatTile label="A-Z attempts" value={overview.total_letter_attempts} animate={animate} icon="🔤" />
        <StatTile label="Word attempts" value={overview.total_word_practice_sessions} animate={animate} icon="📝" />
        <StatTile label="Native sign attempts" value={overview.total_native_sign_attempts} animate={animate} icon="🤟" />
        <StatTile label="Total XP earned" value={overview.total_xp_awarded} animate={animate} icon="⚡" />
      </div>
    </section>
  );
}

function StatTile({
  label,
  value,
  suffix = "",
  icon,
  tone = "accent",
  animate,
}: {
  label: string;
  value: number;
  suffix?: string;
  icon?: string;
  tone?: "accent" | "success" | "warning";
  animate: boolean;
}) {
  const displayed = useCountUp(value, animate ? 900 : 0);
  const toneColor =
    tone === "success" ? "var(--color-success)" : tone === "warning" ? "var(--color-warm)" : "var(--color-accent)";
  return (
    <Card className="lift-hover relative overflow-hidden">
      <div
        className="pointer-events-none absolute -right-6 -top-6 h-24 w-24 rounded-full opacity-20"
        style={{ background: `radial-gradient(circle, ${toneColor} 0%, transparent 70%)` }}
        aria-hidden="true"
      />
      <div className="relative">
        {icon && (
          <span className="mb-1 block text-xl" aria-hidden="true">
            {icon}
          </span>
        )}
        <p className="font-display text-3xl tabular-nums text-[#eef4f0]">
          {displayed.toLocaleString()}
          {suffix}
        </p>
        <p className="mt-0.5 text-xs uppercase tracking-[0.1em] text-[var(--color-muted)]">{label}</p>
      </div>
    </Card>
  );
}

function StudentTable({ students }: { students: MentorStudent[] }) {
  if (students.length === 0) {
    return <EmptyState icon="👥" title="No students yet" description="Registered users will appear here." />;
  }
  return (
    <Card padding={false} className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-[var(--color-line-soft)] bg-[var(--color-panel-soft)]/60 text-left text-xs uppercase tracking-[0.08em] text-[var(--color-muted)]">
              <th className="px-4 py-3 font-medium">Student</th>
              <th className="px-4 py-3 font-medium">Level</th>
              <th className="px-4 py-3 font-medium">XP</th>
              <th className="px-4 py-3 font-medium">Streak</th>
              <th className="px-4 py-3 font-medium">Accuracy</th>
              <th className="px-4 py-3 font-medium">Native mastery</th>
            </tr>
          </thead>
          <tbody>
            {students.map((student, index) => (
              <tr
                key={student.id}
                className={`border-b border-[var(--color-line-soft)]/60 transition hover:bg-[var(--color-panel-soft)]/50 ${
                  index % 2 === 1 ? "bg-[var(--color-panel-soft)]/20" : ""
                }`}
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2.5">
                    <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-gradient-to-br from-[var(--color-accent)]/30 to-[var(--color-accent-dim)]/10 text-xs font-semibold text-[var(--color-accent)]">
                      {student.username.slice(0, 1).toUpperCase()}
                    </span>
                    <span className="font-medium text-[#eef4f0]">{student.username}</span>
                    {student.role === "admin" && <StatusChip tone="neutral">Admin</StatusChip>}
                  </div>
                </td>
                <td className="px-4 py-3 tabular-nums text-[var(--color-mist)]">{student.level}</td>
                <td className="px-4 py-3 tabular-nums text-[var(--color-mist)]">{student.xp.toLocaleString()}</td>
                <td className="px-4 py-3 tabular-nums text-[var(--color-mist)]">{student.current_streak}d</td>
                <td className="px-4 py-3 tabular-nums text-[var(--color-mist)]">
                  {student.accuracy != null ? `${student.accuracy}%` : "—"}
                </td>
                <td className="px-4 py-3">
                  {student.native_mastery != null ? (
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-[var(--color-ink)]">
                        <div
                          className="h-full rounded-full bg-[var(--color-accent)]"
                          style={{ width: `${student.native_mastery}%` }}
                        />
                      </div>
                      <span className="text-xs tabular-nums text-[var(--color-muted)]">{student.native_mastery}%</span>
                    </div>
                  ) : (
                    <span className="text-[var(--color-muted)]">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function MagnitudeBarChart({
  rows,
  valueLabel,
  animate,
}: {
  rows: Array<{ label: string; value: number }>;
  valueLabel: string;
  animate: boolean;
}) {
  if (rows.every((r) => r.value === 0)) {
    return <EmptyChartMessage text="No attempts recorded yet." />;
  }
  return (
    <div className="h-64 w-full" role="img" aria-label={`Bar chart of ${valueLabel} by item`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} vertical={false} />
          <XAxis dataKey="label" tick={{ fill: CHART_TEXT, fontSize: 11 }} axisLine={{ stroke: CHART_GRID }} tickLine={false} />
          <YAxis tick={{ fill: CHART_TEXT, fontSize: 11 }} axisLine={{ stroke: CHART_GRID }} tickLine={false} allowDecimals={false} />
          <Tooltip
            cursor={{ fill: "rgba(215,243,106,0.06)" }}
            contentStyle={{ background: "#17221d", border: "1px solid #2a3b34", borderRadius: 10, fontSize: 12 }}
            labelStyle={{ color: "#eef4f0" }}
            formatter={(value) => [value, valueLabel]}
          />
          <Bar dataKey="value" fill={ACCENT} radius={[4, 4, 0, 0]} isAnimationActive={animate} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CategoryPieChart({ items, animate }: { items: Array<{ label: string; value: number }>; animate: boolean }) {
  return (
    <div className="h-64 w-full" role="img" aria-label="Category distribution pie chart">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={items}
            dataKey="value"
            nameKey="label"
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={2}
            isAnimationActive={animate}
          >
            {items.map((item, index) => (
              <Cell
                key={item.label}
                fill={item.label === "Other" ? OTHER_COLOR : CATEGORY_COLORS[index % CATEGORY_COLORS.length]}
              />
            ))}
          </Pie>
          <Legend
            verticalAlign="bottom"
            iconType="circle"
            formatter={(value) => <span style={{ color: CHART_TEXT, fontSize: 12 }}>{value}</span>}
          />
          <Tooltip
            contentStyle={{ background: "#17221d", border: "1px solid #2a3b34", borderRadius: 10, fontSize: 12 }}
            labelStyle={{ color: "#eef4f0" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

function EmptyChartMessage({ text }: { text: string }) {
  return (
    <div className="grid h-40 place-items-center rounded-[var(--radius-control)] border border-dashed border-[var(--color-line)] text-sm text-[var(--color-muted)]">
      {text}
    </div>
  );
}
