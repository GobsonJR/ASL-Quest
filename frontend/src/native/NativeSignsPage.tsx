import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Card, EmptyState, PageLayout, PrimaryButton, ProgressBar, StatusChip } from "../components/AppShell";
import { useToast } from "../components/Toast";
import { useGame } from "../game/GameContext";
import { getNativeProgress, getNativeSigns, type NativeProgress, type NativeSign } from "./api";

export function NativeSignsPage() {
  const { push } = useToast();
  const { startNativeSignPractice } = useGame();
  const [signs, setSigns] = useState<NativeSign[] | null>(null);
  const [progress, setProgress] = useState<NativeProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getNativeSigns()
      .then((response) => {
        if (cancelled) return;
        setSigns(response.items);
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Unable to load native signs. Please try again.";
        setError(message);
        push(message, "error");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    // Progress is a secondary enhancement to the catalog, not the source of truth for
    // it: if this call fails, the page still shows the real catalog with the honest
    // "not started yet" placeholders rather than blocking on it.
    getNativeProgress()
      .then((response) => {
        if (!cancelled) setProgress(response);
      })
      .catch(() => {
        if (!cancelled) setProgress(null);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadToken]);

  const progressBySignId = useMemo(() => {
    const map = new Map<number, NativeProgress["signs"][number]>();
    for (const entry of progress?.signs ?? []) map.set(entry.sign_id, entry);
    return map;
  }, [progress]);

  const categories = useMemo(
    () => Array.from(new Set((signs ?? []).map((s) => s.category).filter((v): v is string => Boolean(v)))).sort(),
    [signs]
  );
  const difficulties = useMemo(
    () => Array.from(new Set((signs ?? []).map((s) => s.difficulty).filter((v): v is string => Boolean(v)))).sort(),
    [signs]
  );

  const filtered = useMemo(() => {
    if (!signs) return [];
    const q = search.trim().toLowerCase();
    return signs.filter((sign) => {
      if (category && sign.category !== category) return false;
      if (difficulty && sign.difficulty !== difficulty) return false;
      if (!q) return true;
      return (
        sign.gloss.toLowerCase().includes(q) ||
        sign.display_name.toLowerCase().includes(q) ||
        (sign.meaning ?? "").toLowerCase().includes(q)
      );
    });
  }, [signs, search, category, difficulty]);

  if (loading) {
    return (
      <PageLayout>
        <HeroSection />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-label="Loading native signs" aria-busy="true">
          {Array.from({ length: 6 }).map((_, index) => (
            <div
              key={index}
              className="h-56 animate-pulse rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)]"
            />
          ))}
        </div>
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout>
        <HeroSection />
        <EmptyState
          icon="⚠️"
          title="Unable to load native signs"
          description={error}
          action={<PrimaryButton onClick={() => setReloadToken((value) => value + 1)}>Retry</PrimaryButton>}
        />
      </PageLayout>
    );
  }

  if (!signs || signs.length === 0) {
    return (
      <PageLayout>
        <HeroSection />
        <EmptyState
          icon="🤟"
          title="No native signs available yet"
          description="The native sign catalog is empty right now. Check back once signs have been added."
        />
      </PageLayout>
    );
  }

  const mastered = progress?.mastered_signs ?? 0;
  const overallMastery = progress?.overall_mastery ?? 0;
  const started = progress?.started_signs ?? 0;

  return (
    <PageLayout>
      <HeroSection />

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard value={signs.length} label="Native Signs" icon="🤟" />
        <StatCard
          value={mastered}
          label="Mastered"
          icon="⭐"
          sub={progress ? undefined : "Start learning"}
        />
        <StatCard
          value={`${overallMastery}%`}
          label="Overall Mastery"
          icon="📈"
          sub={progress ? `${started} of ${signs.length} started` : "Progress tracking coming soon"}
        />
      </div>

      <section className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-base font-semibold text-[var(--color-ink)]">Sign catalog</h3>
            <p className="mt-1 text-sm text-[var(--color-ink-soft)]">
              {filtered.length} of {signs.length} signs shown
            </p>
          </div>
          <input
            className="input-field sm:max-w-xs"
            placeholder="Search signs..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search native signs"
          />
        </div>

        {(categories.length > 1 || difficulties.length > 1) && (
          <div className="flex flex-wrap items-center gap-2">
            {categories.length > 1 && (
              <FilterGroup
                label="Category"
                options={categories}
                selected={category}
                onSelect={setCategory}
              />
            )}
            {difficulties.length > 1 && (
              <FilterGroup
                label="Difficulty"
                options={difficulties}
                selected={difficulty}
                onSelect={setDifficulty}
              />
            )}
          </div>
        )}

        {filtered.length === 0 ? (
          <p className="rounded-[var(--radius-panel)] border border-dashed border-[var(--color-border)] px-6 py-10 text-center text-sm text-[var(--color-ink-soft)]">
            No signs match your filters.
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.map((sign, index) => (
              <SignCard
                key={sign.id}
                sign={sign}
                index={index}
                progress={progressBySignId.get(sign.id) ?? null}
                onPractice={() => startNativeSignPractice(sign.id)}
              />
            ))}
          </div>
        )}
      </section>
    </PageLayout>
  );
}

function HeroSection() {
  return (
    <div className="relative overflow-hidden rounded-[var(--radius-panel)] bg-[var(--color-native)] p-6 text-white md:p-8">
      <div className="pointer-events-none absolute -right-12 -top-16 h-56 w-56 rounded-full bg-white/10" aria-hidden="true" />
      <div className="relative">
        <div className="flex flex-wrap items-center gap-3">
          <span className="grid h-14 w-14 shrink-0 place-items-center rounded-[var(--radius-panel)] bg-white/15 text-3xl">
            🤟
          </span>
          <StatusChip tone="neutral">
            <span className="text-[var(--color-native)]">ISOLATED SIGN RECOGNITION</span>
          </StatusChip>
        </div>

        <p className="mt-5 font-display text-2xl font-bold md:text-4xl">Native ASL</p>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-white/85">
          Learn complete signs, not just letters — real ASL signs, not letter-by-letter spelling.
        </p>

        <div className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
          <p className="rounded-[var(--radius-control)] bg-white/10 px-4 py-3">
            <span className="font-bold">A–Z</span> teaches individual handshapes.
          </p>
          <p className="rounded-[var(--radius-control)] bg-white/10 px-4 py-3">
            <span className="font-bold">Word Spelling</span> combines letters.
          </p>
          <p className="rounded-[var(--radius-control)] bg-white/20 px-4 py-3">
            <span className="font-bold">Native Signs</span> recognizes a complete isolated ASL sign.
          </p>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  value,
  label,
  icon,
  sub,
}: {
  value: string | number;
  label: string;
  icon: string;
  sub?: string;
}) {
  return (
    <Card className="lift-hover flex items-center gap-4">
      <span className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-[var(--color-native-soft)] text-2xl">
        {icon}
      </span>
      <div>
        <p className="font-display text-3xl font-bold text-[var(--color-ink)]">{value}</p>
        <p className="text-xs font-bold uppercase tracking-[0.1em] text-[var(--color-muted)]">{label}</p>
        {sub && <p className="mt-0.5 text-xs text-[var(--color-muted)]">{sub}</p>}
      </div>
    </Card>
  );
}

function FilterGroup({
  label,
  options,
  selected,
  onSelect,
}: {
  label: string;
  options: string[];
  selected: string | null;
  onSelect: (value: string | null) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={`Filter by ${label}`}>
      <span className="mr-1 text-xs font-bold uppercase tracking-[0.1em] text-[var(--color-muted)]">{label}</span>
      <button
        type="button"
        className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
          selected === null
            ? "bg-[var(--color-native)] text-white"
            : "border-2 border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-native-soft)]"
        }`}
        onClick={() => onSelect(null)}
        aria-pressed={selected === null}
      >
        All
      </button>
      {options.map((option) => (
        <button
          key={option}
          type="button"
          className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
            selected === option
              ? "bg-[var(--color-native)] text-white"
              : "border-2 border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-native-soft)]"
          }`}
          onClick={() => onSelect(option)}
          aria-pressed={selected === option}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function SignCard({
  sign,
  index,
  progress,
  onPractice,
}: {
  sign: NativeSign;
  index: number;
  progress: NativeProgress["signs"][number] | null;
  onPractice: () => void;
}) {
  return (
    <Card
      className="lift-hover animate-pop flex h-full flex-col"
      style={{ animationDelay: `${Math.min(index, 10) * 40}ms` } as CSSProperties}
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-display text-2xl font-bold text-[var(--color-ink)]">{sign.display_name}</h3>
        <StatusChip tone={sign.active ? "success" : "neutral"}>{sign.active ? "Active" : "Inactive"}</StatusChip>
      </div>
      <p className="mt-1 text-xs font-bold uppercase tracking-[0.14em] text-[var(--color-native)]">
        {sign.gloss}
      </p>

      {progress && (
        <div className="mt-3">
          <ProgressBar percent={progress.mastery} label="Mastery" size="sm" />
          {progress.attempts > 0 && (
            <p className="mt-1 text-xs text-[var(--color-muted)]">
              {progress.correct} / {progress.attempts} correct
            </p>
          )}
        </div>
      )}

      {sign.meaning && <p className="mt-3 text-sm text-[var(--color-ink-soft)]">{sign.meaning}</p>}
      {sign.description && <p className="mt-2 text-sm text-[var(--color-ink-soft)]">{sign.description}</p>}
      {sign.example_text && (
        <p className="mt-2 text-sm italic text-[var(--color-muted)]">&ldquo;{sign.example_text}&rdquo;</p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        {sign.category && <StatusChip tone="neutral">{sign.category}</StatusChip>}
        {sign.difficulty && <StatusChip tone="accent">{sign.difficulty}</StatusChip>}
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        <StatusChip tone={sign.dataset_available ? "success" : "neutral"}>
          Dataset {sign.dataset_available ? "available" : "unavailable"}
        </StatusChip>
        <StatusChip tone={sign.model_available ? "success" : "neutral"}>
          Model {sign.model_available ? "available" : "in progress"}
        </StatusChip>
      </div>

      <div className="mt-auto pt-4">
        <PrimaryButton className="w-full" onClick={onPractice}>
          Start Practice
        </PrimaryButton>
      </div>
    </Card>
  );
}
