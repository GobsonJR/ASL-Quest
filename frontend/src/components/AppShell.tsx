import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { useAuth } from "../auth/AuthContext";
import { DEMO_MODE } from "../config";
import type { NavPage } from "../game/constants";
import { useGame } from "../game/GameContext";
import { lettersMasteredCount } from "../game/gamification";
import { ExplorePopup } from "./ExplorePopup";
import { ProfileMenu } from "./ProfileMenu";
import { Glyph, Icon } from "./Icon";

// Real app screens only (src/game/constants.ts::NavPage) -- no invented pages.
// Five primary learning destinations; everything else stays reachable from "More" or the account menu.
const PRIMARY_NAV: Array<{ id: NavPage; label: string; icon: string }> = [
  { id: "learn", label: "Learn", icon: "📚" },
  { id: "practice", label: "Practice", icon: "✋" },
  { id: "alphabet", label: "Alphabet", icon: "🔤" },
  { id: "words", label: "Words", icon: "🧩" },
  { id: "achievements", label: "Badges", icon: "🏅" },
];

const SECONDARY_NAV: Array<{ id: NavPage; label: string; icon: string; adminOnly?: boolean }> = [
  { id: "home", label: "Dashboard", icon: "🏠" },
  { id: "native", label: "Native Signs", icon: "🎥" },
  { id: "challenges", label: "Challenges", icon: "🎯" },
  { id: "progress", label: "Progress", icon: "📈" },
  { id: "assistant", label: "AURA", icon: "✨" },
  { id: "admin", label: "Admin", icon: "🛠️", adminOnly: true },
  { id: "mentor-dashboard", label: "Mentor", icon: "📊", adminOnly: true },
  // Profile and Settings live in the account menu (top-right avatar); dataset collection is a tab of the Admin workspace.
];

function isNavActive(id: NavPage, page: NavPage, practiceKind: string): boolean {
  if (id === "words") return page === "words" || (page === "practice" && practiceKind === "word");
  if (id === "practice") return page === "practice" && practiceKind !== "word";
  if (id === "native") return page === "native" || page === "native-practice";
  return page === id;
}

function goNav(id: NavPage, navigate: (page: NavPage) => void, startPractice: () => void): void {
  if (id === "practice") {
    startPractice();
    return;
  }
  navigate(id);
}

export function AppShell({
  ready,
  backendNote,
  children,
}: {
  ready: boolean;
  backendNote: string;
  children: ReactNode;
}) {
  const { page, navigate, startPractice, practiceKind, state, level, levelProgress } = useGame();
  const { user, logout, syncStatus } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [exploreOpen, setExploreOpen] = useState(false);
  const isAdmin = user?.role === "admin";
  const secondaryItems = SECONDARY_NAV.filter((item) => !item.adminOnly || isAdmin);
  const secondaryActive = secondaryItems.some((item) => isNavActive(item.id, page, practiceKind));

  // Close the mobile "More" menu on navigation and on Escape.
  useEffect(() => {
    setMenuOpen(false);
    setExploreOpen(false);
  }, [page]);
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  const syncLabel =
    syncStatus === "syncing"
      ? "Syncing…"
      : syncStatus === "offline"
        ? "Offline"
        : syncStatus === "error"
          ? "Sync issue"
          : null;

  const go = (id: NavPage) => goNav(id, navigate, startPractice);

  return (
    <div className="min-h-screen md:flex">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 flex-col border-r-2 border-[var(--color-border)] bg-[var(--color-surface)] md:sticky md:top-0 md:flex md:h-screen">
        <button
          type="button"
          className="flex items-center gap-2.5 px-6 py-6 text-left"
          onClick={() => navigate("home")}
          aria-label="ASL Quest dashboard"
        >
          <span
            className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-[var(--color-primary)] text-lg text-white"
            aria-hidden="true"
          >
            <Glyph value="🤟" />
          </span>
          <span className="block font-display text-xl font-black leading-tight text-[var(--color-primary)]">
            ASL Quest
          </span>
        </button>

        <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-4" aria-label="Main navigation">
          {PRIMARY_NAV.map((item) => (
            <SidebarNavButton
              key={item.id}
              icon={item.icon}
              active={isNavActive(item.id, page, practiceKind)}
              onClick={() => go(item.id)}
            >
              {item.label === "Badges" ? "Achievements" : item.label}
            </SidebarNavButton>
          ))}
          <button
            type="button"
            onClick={() => setExploreOpen(true)}
            aria-haspopup="dialog"
            className="mt-2 flex w-full items-center gap-3 rounded-[var(--radius-control)] bg-gradient-to-r from-[var(--color-primary-soft)] to-[var(--color-native-soft)] px-3 py-2.5 text-sm font-bold text-[var(--color-primary-dim)] shadow-[0_3px_0_0_rgba(0,0,0,0.06)] transition hover:-translate-y-0.5 hover:shadow-[0_6px_14px_-8px_rgba(0,0,0,0.3)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)]"
          >
            <span className="text-base" aria-hidden="true">
              <Glyph value="🧭" />
            </span>
            Explore paths
          </button>
          <p className="px-3 pb-1 pt-4 text-xs font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">More</p>
          {secondaryItems.map((item) => (
            <SidebarNavButton
              key={item.id}
              icon={item.icon}
              active={isNavActive(item.id, page, practiceKind)}
              onClick={() => go(item.id)}
            >
              {item.label}
            </SidebarNavButton>
          ))}
        </nav>

        <div className="border-t border-[var(--color-border-soft)] p-4">
          <div className="flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${ready ? "bg-[var(--color-success)]" : "bg-[var(--color-xp)]"}`}
              aria-hidden="true"
            />
            <span className="truncate">{backendNote}</span>
          </div>
        </div>
      </aside>

      <div className="flex min-h-screen min-w-0 flex-1 flex-col pb-36 md:pb-24">
        {DEMO_MODE && (
          <p className="bg-[var(--color-xp-soft)] px-4 py-1.5 text-center text-xs font-bold text-[var(--color-ink)]">
            Demo: no account or server connected. Progress stays in this browser, and camera recognition and AI chat are unavailable.
          </p>
        )}
        <header className="sticky top-0 z-30 border-b-2 border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center justify-between gap-3 px-4 py-3 md:px-8">
            <button
              type="button"
              className="flex items-center gap-2 md:hidden"
              onClick={() => navigate("home")}
              aria-label="ASL Quest dashboard"
            >
              <span className="grid h-8 w-8 place-items-center rounded-xl bg-[var(--color-primary)] text-sm text-white" aria-hidden="true">
                <Glyph value="🤟" />
              </span>
              <span className="whitespace-nowrap font-display text-base font-black text-[var(--color-primary)]">ASL Quest</span>
            </button>
            <span className="hidden font-display text-base font-bold text-[var(--color-ink)] md:block">
              {greetingForPage(page)}
            </span>

            <div className="flex items-center gap-2">
              <span className="flex items-center gap-2 min-[1500px]:hidden">
                <XPChip value={state.xp} />
                <StreakChip value={state.streak} />
              </span>
              {syncLabel && (
                <span className="hidden text-xs text-[var(--color-xp)] sm:inline" role="status">
                  {syncLabel}
                </span>
              )}
              <ProfileMenu
                name={user?.username ?? "Account"}
                email={DEMO_MODE ? undefined : user?.email}
                isAdmin={isAdmin}
                currentPage={page}
                onNavigate={(target) => go(target)}
                onLogout={DEMO_MODE ? undefined : logout}
              />
              <button
                type="button"
                className={`grid h-11 min-w-11 place-items-center rounded-xl border-2 text-lg md:hidden ${
                  menuOpen || secondaryActive
                    ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]"
                    : "border-[var(--color-border)]"
                }`}
                aria-label="More pages"
                aria-expanded={menuOpen}
                aria-controls="mobile-more-menu"
                onClick={() => setMenuOpen((open) => !open)}
              >
                <Icon name={menuOpen ? "x" : "menu"} className="h-5 w-5" />
              </button>
            </div>
          </div>
          {menuOpen && (
            <nav
              id="mobile-more-menu"
              aria-label="More pages"
              className="grid grid-cols-2 gap-2 border-t border-[var(--color-border-soft)] bg-[var(--color-surface)] p-3 md:hidden"
            >
              {secondaryItems.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`flex min-h-11 items-center gap-2 rounded-xl border-2 px-3 text-sm font-bold ${
                    isNavActive(item.id, page, practiceKind)
                      ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]"
                      : "border-[var(--color-border)] text-[var(--color-ink-soft)]"
                  }`}
                  onClick={() => go(item.id)}
                  aria-current={isNavActive(item.id, page, practiceKind) ? "page" : undefined}
                >
                  <span aria-hidden="true"><Glyph value={item.icon} /></span>
                  {item.label}
                </button>
              ))}
            </nav>
          )}
        </header>

        <div className="mx-auto flex w-full max-w-[88rem] flex-1 gap-8 px-4 py-6 md:px-8 md:py-10">
          <main className="mx-auto w-full min-w-0 max-w-5xl flex-1">
            <div key={page} className="page-enter">
              {children}
            </div>
          </main>
          <ProgressPanel level={level} levelProgress={levelProgress} onNavigate={go} />
        </div>
      </div>

      {/* Mobile bottom nav: all six destinations in one row, no scrolling. */}
      <nav
        className="fixed inset-x-0 bottom-0 z-40 border-t-2 border-[var(--color-border)] bg-[var(--color-surface)] pb-[env(safe-area-inset-bottom)] md:hidden"
        aria-label="Mobile navigation"
      >
        <ul className="grid grid-cols-6 px-1 py-1">
          {PRIMARY_NAV.map((item) => (
            <li key={item.id} className="min-w-0">
              <MobileNavButton
                icon={item.icon}
                active={isNavActive(item.id, page, practiceKind)}
                onClick={() => go(item.id)}
                label={item.label}
              />
            </li>
          ))}
        </ul>
      </nav>

      {exploreOpen && (
        <ExplorePopup
          lettersMastered={lettersMasteredCount(state)}
          onClose={() => setExploreOpen(false)}
          onSelect={(target) => {
            setExploreOpen(false);
            go(target);
          }}
        />
      )}
    </div>
  );
}

/** Right-hand progress panel. Shown only on very wide screens, and only real
 * state: XP/level, streak, today's daily-challenge progress, badges earned. */
function ProgressPanel({
  level,
  levelProgress,
  onNavigate,
}: {
  level: number;
  levelProgress: { current: number; next: number; percent: number };
  onNavigate: (page: NavPage) => void;
}) {
  const { state } = useGame();
  const daily = state.dailyChallenge;
  const dailyPercent = daily.target > 0 ? Math.min(100, (daily.progress / daily.target) * 100) : 0;
  return (
    <aside
      className="hidden w-72 shrink-0 space-y-4 min-[1500px]:sticky min-[1500px]:top-24 min-[1500px]:block min-[1500px]:self-start"
      aria-label="Your progress"
    >
      <Card className="space-y-4">
        <div className="flex gap-2">
          <XPChip value={state.xp} />
          <StreakChip value={state.streak} />
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs font-semibold text-[var(--color-muted)]">
            <span>Level {level}</span>
            <span>
              {levelProgress.current}/{levelProgress.next} XP
            </span>
          </div>
          <ProgressBar percent={levelProgress.percent} showPercent={false} />
        </div>
      </Card>
      <Card className="space-y-3">
        <SectionLabel>Daily goal</SectionLabel>
        <p className="text-sm font-semibold text-[var(--color-ink)]">
          {daily.completed ? "Completed today" : `${daily.progress} / ${daily.target} signs`}
        </p>
        <ProgressBar percent={daily.completed ? 100 : dailyPercent} showPercent={false} />
        <button
          type="button"
          className="text-sm font-bold text-[var(--color-primary-dim)] hover:underline"
          onClick={() => onNavigate("challenges")}
        >
          View challenges →
        </button>
      </Card>
      <Card className="space-y-2">
        <SectionLabel>Achievements</SectionLabel>
        <p className="text-sm text-[var(--color-ink-soft)]">
          {state.unlockedBadges.length === 0
            ? "No badges yet — your first correct sign unlocks one."
            : `${state.unlockedBadges.length} badge${state.unlockedBadges.length === 1 ? "" : "s"} earned`}
        </p>
        <button
          type="button"
          className="text-sm font-bold text-[var(--color-primary-dim)] hover:underline"
          onClick={() => onNavigate("achievements")}
        >
          See all →
        </button>
      </Card>
    </aside>
  );
}

function greetingForPage(page: NavPage): string {
  const titles: Partial<Record<NavPage, string>> = {
    home: "Your learning space",
    alphabet: "The alphabet",
    learn: "Your ASL journey",
    words: "Word Spelling",
    practice: "Practice",
    native: "Native ASL",
    "native-practice": "Native ASL",
    challenges: "Today's missions",
    progress: "Your progress",
    achievements: "Your badges",
    assistant: "AURA",
    profile: "Your profile",
    settings: "Settings",
    admin: "Admin workspace",
    "mentor-dashboard": "Mentor dashboard",
  };
  return titles[page] ?? "ASL Quest";
}

function SidebarNavButton({
  icon,
  active,
  onClick,
  children,
}: {
  icon: string;
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={`flex w-full items-center gap-3 rounded-[var(--radius-control)] px-3 py-2.5 text-sm font-medium transition ${
        active
          ? "bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]"
          : "text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-soft)] hover:text-[var(--color-ink)]"
      }`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      <span className="text-base" aria-hidden="true">
        <Glyph value={icon} />
      </span>
      {children}
    </button>
  );
}

function MobileNavButton({
  icon,
  active,
  onClick,
  label,
}: {
  icon: string;
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`flex min-h-12 w-full min-w-0 flex-col items-center justify-center gap-0.5 rounded-[var(--radius-control)] px-0.5 py-1 text-[0.62rem] font-bold transition ${
        active ? "text-[var(--color-primary)]" : "text-[var(--color-muted)]"
      }`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      <span className={`text-lg leading-none ${active ? "" : "opacity-80"}`} aria-hidden="true">
        <Glyph value={icon} />
      </span>
      <span className="w-full truncate text-center">{label}</span>
    </button>
  );
}

export function PageLayout({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`space-y-8 md:space-y-10 ${className}`}>{children}</div>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  compact = false,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  compact?: boolean;
}) {
  return (
    <div>
      {eyebrow && (
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--color-primary)]">{eyebrow}</p>
      )}
      <h2
        className={`font-display font-semibold text-[var(--color-ink)] ${
          compact ? "mt-1 text-xl md:text-2xl" : "mt-1 text-2xl md:text-4xl"
        }`}
      >
        {title}
      </h2>
      {description && <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-[var(--color-ink-soft)]">{description}</p>}
    </div>
  );
}

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface)] p-5 shadow-[0_1px_2px_rgba(35,40,66,0.04),0_10px_28px_-16px_rgba(35,40,66,0.14)] md:p-6 ${className}`}
    >
      {children}
    </section>
  );
}

export function Card({
  children,
  className = "",
  hover = false,
  padding = true,
  style,
  "data-testid": testId,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  padding?: boolean;
  style?: CSSProperties;
  "data-testid"?: string;
}) {
  return (
    <section
      data-testid={testId}
      className={`rounded-[var(--radius-panel)] border-2 border-[var(--color-border)] bg-[var(--color-surface)] shadow-[0_2px_0_0_var(--color-border)] ${
        padding ? "p-5 md:p-6" : ""
      } ${hover ? "card-hover" : ""} ${className}`}
      style={style}
    >
      {children}
    </section>
  );
}

export function Divider() {
  return <hr className="border-0 border-t border-[var(--color-border-soft)]" />;
}

export function StatGroup({
  items,
  className = "",
}: {
  items: Array<{ label: string; value: string | number; sub?: string }>;
  className?: string;
}) {
  return (
    <div className={`flex flex-wrap gap-x-8 gap-y-4 ${className}`}>
      {items.map((item) => (
        <div key={item.label}>
          <p className="text-xs text-[var(--color-muted)]">{item.label}</p>
          <p className="mt-0.5 text-lg font-semibold tabular-nums text-[var(--color-ink)]">{item.value}</p>
          {item.sub && <p className="mt-0.5 text-xs text-[var(--color-muted)]">{item.sub}</p>}
        </div>
      ))}
    </div>
  );
}

export function PrimaryButton({
  children,
  onClick,
  className = "",
  disabled = false,
  ariaLabel,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  className?: string;
  disabled?: boolean;
  ariaLabel?: string;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      className={`inline-flex min-h-11 items-center justify-center gap-1.5 rounded-[var(--radius-control)] bg-[var(--color-primary)] px-5 py-2.5 text-sm font-extrabold uppercase tracking-wide text-white shadow-[0_4px_0_0_var(--color-primary-dim)] transition hover:brightness-105 active:translate-y-[3px] active:shadow-[0_1px_0_0_var(--color-primary-dim)] disabled:cursor-not-allowed disabled:bg-[var(--color-locked-soft)] disabled:text-[var(--color-locked)] disabled:shadow-[0_4px_0_0_var(--color-border)] ${className}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  onClick,
  className = "",
  ariaLabel,
  type = "button",
  disabled = false,
}: {
  children: ReactNode;
  onClick?: () => void;
  className?: string;
  ariaLabel?: string;
  type?: "button" | "submit";
  disabled?: boolean;
}) {
  return (
    <button
      type={type}
      className={`inline-flex min-h-11 items-center justify-center rounded-[var(--radius-control)] border-2 border-[var(--color-border)] bg-transparent px-5 py-2.5 text-sm font-semibold text-[var(--color-ink-soft)] transition hover:border-[var(--color-primary-soft)] hover:bg-[var(--color-primary-soft)] hover:text-[var(--color-primary-dim)] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      disabled={disabled}
      onClick={onClick}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}

export function ProgressBar({
  percent,
  label,
  showPercent = true,
  shimmer = false,
  size = "md",
}: {
  percent: number;
  label?: string;
  showPercent?: boolean;
  shimmer?: boolean;
  size?: "sm" | "md";
}) {
  const clamped = Math.min(100, Math.max(0, percent));
  const barHeight = size === "sm" ? "h-2" : "h-3";
  return (
    <div>
      {label && (
        <div className="mb-2 flex justify-between text-sm text-[var(--color-ink-soft)]">
          <span>{label}</span>
          {showPercent && <span className="font-semibold text-[var(--color-ink)]">{Math.round(clamped)}%</span>}
        </div>
      )}
      <div className={`overflow-hidden rounded-full bg-[var(--color-surface-sunken)] ${barHeight}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ${shimmer ? "shimmer-bar" : "bg-[var(--color-primary)]"}`}
          style={{ width: `${clamped}%` }}
          role="progressbar"
          aria-valuenow={Math.round(clamped)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

export function StatPill({
  icon,
  label,
  value,
  sub,
}: {
  icon?: string;
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="rounded-[var(--radius-control)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
      <p className="text-xs text-[var(--color-muted)]">
        {icon && <span aria-hidden="true"><Glyph value={icon} /> </span>}
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-[var(--color-ink)]">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-[var(--color-muted)]">{sub}</p>}
    </div>
  );
}

export function StatusChip({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "success" | "warning" | "accent";
  children: ReactNode;
}) {
  const tones = {
    neutral: "border-[var(--color-border)] bg-[var(--color-surface-soft)] text-[var(--color-ink-soft)]",
    success: "border-transparent bg-[var(--color-success-soft)] text-[var(--color-success)]",
    warning: "border-transparent bg-[var(--color-streak-soft)] text-[var(--color-streak)]",
    accent: "border-transparent bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[var(--radius-panel)] border-2 border-dashed border-[var(--color-border)] bg-[var(--color-surface-soft)]/50 px-6 py-10 text-center">
      <div className="text-4xl" aria-hidden="true">
        <Glyph value={icon} />
      </div>
      <h3 className="mt-4 text-base font-semibold text-[var(--color-ink)]">{title}</h3>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-[var(--color-ink-soft)]">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function MasteryBadge({ status }: { status: "NEW" | "LEARNING" | "PRACTICED" | "MASTERED" }) {
  const styles = {
    NEW: "text-[var(--color-muted)] border-transparent bg-[var(--color-locked-soft)]",
    LEARNING: "text-[var(--color-xp)] border-transparent bg-[var(--color-xp-soft)]",
    PRACTICED: "text-[var(--color-primary-dim)] border-transparent bg-[var(--color-primary-soft)]",
    MASTERED: "text-[var(--color-success)] border-transparent bg-[var(--color-success-soft)]",
  };
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[0.65rem] font-bold uppercase tracking-wide ${styles[status]}`}>
      {status}
    </span>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--color-muted)]">{children}</p>;
}

export function SectionHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h3 className="text-lg font-semibold text-[var(--color-ink)]">{title}</h3>
        {description && <p className="mt-1 text-sm text-[var(--color-ink-soft)]">{description}</p>}
      </div>
      {action}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small reusable "game feel" primitives -- shared by the shell header, Home,
// Profile, and anywhere else a learner's identity/rewards show up, so these
// stay visually identical everywhere rather than each page reinventing them.
// ---------------------------------------------------------------------------

const AVATAR_PALETTE = [
  "var(--color-primary)",
  "var(--color-native)",
  "var(--color-streak)",
  "var(--color-success)",
];

function paletteIndexFor(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return hash % AVATAR_PALETTE.length;
}

export function Avatar({ name, size = "md" }: { name: string; size?: "sm" | "md" | "lg" }) {
  const dimensions = size === "sm" ? "h-7 w-7 text-xs" : size === "lg" ? "h-16 w-16 text-2xl" : "h-9 w-9 text-sm";
  const initial = name.trim().charAt(0).toUpperCase() || "?";
  const background = AVATAR_PALETTE[paletteIndexFor(name)];
  return (
    <span
      className={`grid shrink-0 place-items-center rounded-full font-display font-semibold text-white ${dimensions}`}
      style={{ background }}
      aria-hidden="true"
    >
      {initial}
    </span>
  );
}

export function XPChip({ value }: { value: number | string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--color-xp-soft)] px-2.5 py-1 text-xs font-bold text-[var(--color-xp)]">
      <span aria-hidden="true"><Glyph value="⭐" /></span>
      {value}
      <span className="hidden sm:inline"> XP</span>
    </span>
  );
}

export function StreakChip({ value }: { value: number | string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--color-streak-soft)] px-2.5 py-1 text-xs font-bold text-[var(--color-streak)]">
      <span aria-hidden="true"><Glyph value="🔥" /></span>
      {value}
    </span>
  );
}

export function StatCard({
  icon,
  label,
  value,
  tone = "primary",
}: {
  icon: string;
  label: string;
  value: string | number;
  tone?: "primary" | "success" | "xp" | "streak" | "native";
}) {
  const tones: Record<string, string> = {
    primary: "bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]",
    success: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
    xp: "bg-[var(--color-xp-soft)] text-[var(--color-xp)]",
    streak: "bg-[var(--color-streak-soft)] text-[var(--color-streak)]",
    native: "bg-[var(--color-native-soft)] text-[var(--color-native)]",
  };
  return (
    <div className="flex items-center gap-3 rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface)] p-4">
      <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-2xl text-lg ${tones[tone]}`} aria-hidden="true">
        <Glyph value={icon} />
      </span>
      <div className="min-w-0">
        <p className="truncate text-xs text-[var(--color-muted)]">{label}</p>
        <p className="text-lg font-bold tabular-nums text-[var(--color-ink)]">{value}</p>
      </div>
    </div>
  );
}
