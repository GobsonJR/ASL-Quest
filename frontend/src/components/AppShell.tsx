import type { CSSProperties, ReactNode } from "react";
import { useAuth } from "../auth/AuthContext";
import type { NavPage } from "../game/constants";
import { useGame } from "../game/GameContext";

const NAV: Array<{ id: NavPage; label: string; adminOnly?: boolean }> = [
  { id: "home", label: "Home" },
  { id: "learn", label: "Learn" },
  { id: "words", label: "Words" },
  { id: "practice", label: "Practice" },
  { id: "native", label: "Native Signs" },
  { id: "challenges", label: "Challenges" },
  { id: "progress", label: "Progress" },
  { id: "achievements", label: "Badges" },
  { id: "profile", label: "Profile" },
  { id: "admin", label: "Admin", adminOnly: true },
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
  const navItems = NAV.filter((item) => !item.adminOnly || user?.role === "admin");

  const syncLabel =
    syncStatus === "syncing"
      ? "Syncing"
      : syncStatus === "offline"
        ? "Offline"
        : syncStatus === "error"
          ? "Sync issue"
          : null;

  return (
    <div className="min-h-screen pb-20 md:pb-10">
      <header className="sticky top-0 z-30 border-b border-[var(--color-line)]/70 bg-[var(--color-panel)]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3 md:px-6">
          <button
            type="button"
            className="group min-w-0 text-left"
            onClick={() => navigate("home")}
            aria-label="ASL Quest home"
          >
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.22em] text-[var(--color-accent)]">
              ASL Quest
            </p>
            <h1 className="font-display text-lg font-medium transition group-hover:text-[var(--color-accent)] md:text-xl">
              Sign & Learn
            </h1>
          </button>

          <div className="flex items-center gap-3 text-sm">
            <div className="hidden items-center gap-3 text-[var(--color-muted)] md:flex">
              <span>
                Lv <strong className="text-[#eef4f0]">{level}</strong>
              </span>
              <span aria-hidden="true">·</span>
              <span>
                <strong className="text-[#eef4f0]">{state.xp}</strong> XP
              </span>
              <span aria-hidden="true">·</span>
              <span>
                <strong className="text-[#eef4f0]">{state.streak}</strong>d streak
              </span>
            </div>
            <button
              type="button"
              className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] px-3 py-1.5 text-xs font-medium text-[var(--color-mist)] transition hover:border-[var(--color-line)] hover:text-[#eef4f0]"
              onClick={() => navigate("profile")}
            >
              {user?.username}
            </button>
            <button
              type="button"
              className="rounded-[var(--radius-control)] px-2 py-1.5 text-xs text-[var(--color-muted)] transition hover:text-[var(--color-mist)]"
              onClick={logout}
            >
              Logout
            </button>
          </div>
        </div>

        <div className="mx-auto hidden max-w-5xl items-center justify-between gap-4 border-t border-[var(--color-line-soft)] px-4 py-2 md:flex md:px-6">
          <nav className="flex flex-wrap gap-0.5" aria-label="Main navigation">
            {navItems.map((item) => (
              <NavButton
                key={item.id}
                active={isNavActive(item.id, page, practiceKind)}
                onClick={() => goNav(item.id, navigate, startPractice)}
              >
                {item.label}
              </NavButton>
            ))}
          </nav>
          <div className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${ready ? "bg-[var(--color-accent)]" : "bg-[var(--color-warm)]"}`}
              aria-hidden="true"
            />
            <span>{backendNote}</span>
            {syncLabel && <span className="text-[var(--color-warm)]">· {syncLabel}</span>}
          </div>
        </div>

        <div className="mx-auto hidden max-w-5xl px-4 pb-2 md:block md:px-6">
          <div className="h-0.5 overflow-hidden rounded-full bg-[var(--color-ink)]">
            <div
              className="h-full rounded-full bg-[var(--color-accent)] transition-all duration-500"
              style={{ width: `${levelProgress.percent}%` }}
              role="progressbar"
              aria-valuenow={levelProgress.current}
              aria-valuemin={0}
              aria-valuemax={levelProgress.next}
              aria-label="Level progress"
            />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-10">
        <div key={page} className="page-enter">
          {children}
        </div>
      </main>

      <nav
        className="fixed inset-x-0 bottom-0 z-40 border-t border-[var(--color-line)]/80 bg-[var(--color-panel)]/95 backdrop-blur-md md:hidden"
        aria-label="Mobile navigation"
      >
        <div className="flex justify-around gap-0.5 px-1 py-1.5">
          {navItems.slice(0, 5).map((item) => (
            <MobileNavButton
              key={item.id}
              active={isNavActive(item.id, page, practiceKind)}
              onClick={() => goNav(item.id, navigate, startPractice)}
              label={item.label}
            />
          ))}
        </div>
        <div className="flex justify-around gap-0.5 border-t border-[var(--color-line-soft)] px-1 py-1.5">
          {navItems.slice(5).map((item) => (
            <MobileNavButton
              key={item.id}
              active={isNavActive(item.id, page, practiceKind)}
              onClick={() => goNav(item.id, navigate, startPractice)}
              label={item.label}
            />
          ))}
        </div>
      </nav>
    </div>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={`rounded-[var(--radius-control)] px-3 py-1.5 text-sm font-medium transition ${
        active
          ? "bg-[var(--color-accent)] text-[var(--color-ink)]"
          : "text-[var(--color-mist)] hover:bg-[var(--color-panel-soft)] hover:text-[#eef4f0]"
      }`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      {children}
    </button>
  );
}

function MobileNavButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`min-w-0 flex-1 rounded-[var(--radius-control)] px-1 py-2 text-[0.65rem] font-medium transition ${
        active ? "bg-[var(--color-accent)] text-[var(--color-ink)]" : "text-[var(--color-muted)]"
      }`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      {label}
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
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-[var(--color-muted)]">{eyebrow}</p>
      )}
      <h2
        className={`font-display font-medium text-[#eef4f0] ${
          compact ? "mt-1 text-xl md:text-2xl" : "mt-1 text-2xl md:text-3xl"
        }`}
      >
        {title}
      </h2>
      {description && <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--color-mist)]">{description}</p>}
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
      className={`rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel)]/80 p-5 md:p-6 ${className}`}
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
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  padding?: boolean;
  style?: CSSProperties;
}) {
  return (
    <section
      className={`rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-[var(--color-panel)]/90 ${
        padding ? "p-5 md:p-6" : ""
      } ${hover ? "card-hover" : ""} ${className}`}
      style={style}
    >
      {children}
    </section>
  );
}

export function Divider() {
  return <hr className="border-0 border-t border-[var(--color-line-soft)]" />;
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
          <p className="mt-0.5 text-lg font-semibold tabular-nums text-[#eef4f0]">{item.value}</p>
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
      className={`inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-[var(--color-ink)] transition hover:brightness-[1.04] active:brightness-95 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
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
}: {
  children: ReactNode;
  onClick?: () => void;
  className?: string;
  ariaLabel?: string;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      className={`inline-flex min-h-10 items-center justify-center rounded-[var(--radius-control)] border border-[var(--color-line)] bg-transparent px-4 py-2 text-sm font-medium text-[var(--color-mist)] transition hover:border-[var(--color-line-soft)] hover:bg-[var(--color-panel-soft)] hover:text-[#eef4f0] ${className}`}
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
  const barHeight = size === "sm" ? "h-1.5" : "h-2";
  return (
    <div>
      {label && (
        <div className="mb-2 flex justify-between text-sm text-[var(--color-mist)]">
          <span>{label}</span>
          {showPercent && <span>{Math.round(clamped)}%</span>}
        </div>
      )}
      <div className={`overflow-hidden rounded-full bg-[var(--color-ink)] ${barHeight}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ${shimmer ? "shimmer-bar" : "bg-[var(--color-accent)]"}`}
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
    <div className="rounded-[var(--radius-control)] border border-[var(--color-line-soft)] bg-[var(--color-panel-soft)]/70 px-4 py-3">
      <p className="text-xs text-[var(--color-muted)]">
        {icon && <span aria-hidden="true">{icon} </span>}
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold tabular-nums">{value}</p>
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
    neutral: "border-[var(--color-line)] bg-[var(--color-panel-soft)] text-[var(--color-mist)]",
    success: "border-[var(--color-success)]/30 bg-[var(--color-success)]/10 text-[var(--color-success)]",
    warning: "border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10 text-[var(--color-warm)]",
    accent: "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 text-[var(--color-accent)]",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}>
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
    <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[var(--radius-panel)] border border-dashed border-[var(--color-line)] px-6 py-10 text-center">
      <div className="text-3xl" aria-hidden="true">
        {icon}
      </div>
      <h3 className="mt-4 text-base font-semibold">{title}</h3>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-[var(--color-mist)]">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function MasteryBadge({ status }: { status: "NEW" | "LEARNING" | "PRACTICED" | "MASTERED" }) {
  const styles = {
    NEW: "text-[var(--color-muted)] border-[var(--color-line)] bg-[var(--color-panel-soft)]",
    LEARNING: "text-[var(--color-warm)] border-[var(--color-warm)]/30 bg-[var(--color-warm)]/10",
    PRACTICED: "text-[var(--color-accent)] border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10",
    MASTERED: "text-[var(--color-success)] border-[var(--color-success)]/30 bg-[var(--color-success)]/10",
  };
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide ${styles[status]}`}>
      {status}
    </span>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-medium uppercase tracking-[0.12em] text-[var(--color-muted)]">{children}</p>;
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
        <h3 className="text-base font-semibold text-[#eef4f0]">{title}</h3>
        {description && <p className="mt-1 text-sm text-[var(--color-mist)]">{description}</p>}
      </div>
      {action}
    </div>
  );
}
