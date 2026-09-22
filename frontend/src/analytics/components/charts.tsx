import type { AnalyticsRange, HeatmapRange } from "../api";

const RANGE_OPTIONS: Array<{ id: AnalyticsRange; label: string }> = [
  { id: "7d", label: "7 Days" },
  { id: "30d", label: "30 Days" },
  { id: "90d", label: "90 Days" },
  { id: "12m", label: "12 Months" },
  { id: "all", label: "All Time" },
];

const HEATMAP_OPTIONS: Array<{ id: HeatmapRange; label: string }> = [
  { id: "3m", label: "3 Months" },
  { id: "6m", label: "6 Months" },
  { id: "12m", label: "12 Months" },
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
              ? "bg-[var(--color-accent)] text-[var(--color-ink)]"
              : "border border-[var(--color-line-soft)] text-[var(--color-mist)] hover:border-[var(--color-accent)]/40"
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

export function HeatmapRangeFilter({
  value,
  onChange,
}: {
  value: HeatmapRange;
  onChange: (value: HeatmapRange) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Heatmap range filter">
      {HEATMAP_OPTIONS.map((option) => (
        <button
          key={option.id}
          type="button"
          className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
            value === option.id
              ? "bg-[var(--color-accent)] text-[var(--color-ink)]"
              : "border border-[var(--color-line-soft)] text-[var(--color-mist)]"
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
      <div className="rounded-2xl border border-dashed border-[var(--color-line-soft)] bg-[var(--color-panel-soft)]/40 px-4 py-8 text-center text-sm text-[var(--color-mist)]">
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
      <h3 className="mb-3 text-sm font-semibold text-[var(--color-mist)]">{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-44 w-full" role="img" aria-label={title}>
        <polyline
          fill="none"
          stroke="var(--color-accent)"
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

const HEAT_COLORS = [
  "bg-[var(--color-panel-soft)]",
  "bg-[var(--color-accent)]/20",
  "bg-[var(--color-accent)]/40",
  "bg-[var(--color-accent)]/65",
  "bg-[var(--color-accent)]",
];

export function ActivityHeatmap({
  cells,
}: {
  cells: Array<{
    date: string;
    attempts: number;
    correct: number;
    accuracy: number | null;
    xp: number;
    intensity: number;
  }>;
}) {
  if (cells.length === 0) {
    return <p className="text-sm text-[var(--color-mist)]">No activity recorded yet.</p>;
  }

  const weeks: typeof cells[] = [];
  for (let index = 0; index < cells.length; index += 7) {
    weeks.push(cells.slice(index, index + 7));
  }

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[720px]">
        <div className="flex gap-1">
          {weeks.map((week, weekIndex) => (
            <div key={weekIndex} className="grid grid-rows-7 gap-1">
              {week.map((cell) => {
                const label = `${cell.date}: ${cell.attempts} practice attempts${
                  cell.accuracy != null ? `, ${cell.accuracy}% accuracy` : ""
                }${cell.xp ? `, +${cell.xp} XP` : ""}`;
                return (
                  <div
                    key={cell.date}
                    title={label}
                    aria-label={label}
                    className={`h-3.5 w-3.5 rounded-sm ${HEAT_COLORS[cell.intensity] ?? HEAT_COLORS[0]}`}
                  />
                );
              })}
            </div>
          ))}
        </div>
        <div className="mt-3 flex items-center gap-2 text-[0.65rem] text-[var(--color-muted)]">
          <span>Less</span>
          {HEAT_COLORS.map((color, index) => (
            <span key={index} className={`h-3 w-3 rounded-sm ${color}`} />
          ))}
          <span>More</span>
        </div>
      </div>
    </div>
  );
}
