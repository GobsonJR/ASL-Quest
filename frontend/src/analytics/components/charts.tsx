import type { AnalyticsRange } from "../api";

const RANGE_OPTIONS: Array<{ id: AnalyticsRange; label: string }> = [
  { id: "7d", label: "7 Days" },
  { id: "30d", label: "30 Days" },
  { id: "90d", label: "90 Days" },
  { id: "12m", label: "12 Months" },
  { id: "all", label: "All Time" },
];

export function DateRangeFilter({
  value,
  onChange,
}: {
  value: AnalyticsRange;
  onChange: (value: AnalyticsRange) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Date range filter">
      {RANGE_OPTIONS.map((option) => (
        <button
          key={option.id}
          type="button"
          className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
            value === option.id
              ? "bg-[var(--color-primary)] text-white"
              : "border border-[var(--color-border-soft)] text-[var(--color-ink-soft)] hover:border-[var(--color-primary)]/40"
          }`}
          onClick={() => onChange(option.id)}
          aria-pressed={value === option.id}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function LineChart({
  title,
  points,
  valueSuffix = "",
  emptyMessage,
}: {
  title: string;
  points: Array<{ date: string; value: number | null }>;
  valueSuffix?: string;
  emptyMessage: string;
}) {
  if (points.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-[var(--color-border-soft)] bg-[var(--color-surface-soft)]/40 px-4 py-8 text-center text-sm text-[var(--color-ink-soft)]">
        {emptyMessage}
      </div>
    );
  }

  const values = points.map((point) => point.value ?? 0);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const width = 640;
  const height = 180;
  const coords = points.map((point, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * width;
    const y = height - ((Number(point.value ?? 0) - min) / Math.max(max - min, 1)) * (height - 20) - 10;
    return `${x},${y}`;
  });

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[var(--color-ink-soft)]">{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-44 w-full" role="img" aria-label={title}>
        <polyline
          fill="none"
          stroke="var(--color-primary)"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={coords.join(" ")}
        />
      </svg>
      <div className="mt-2 flex justify-between text-[0.65rem] text-[var(--color-muted)]">
        <span>{points[0]?.date}</span>
        <span>
          Latest: {points[points.length - 1]?.value ?? "—"}
          {valueSuffix}
        </span>
        <span>{points[points.length - 1]?.date}</span>
      </div>
    </div>
  );
}
