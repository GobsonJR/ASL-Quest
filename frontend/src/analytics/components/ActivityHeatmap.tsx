import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { Card, PrimaryButton, SecondaryButton, SectionLabel } from "../../components/AppShell";
import type { HeatmapData, HeatmapRange } from "../api";

const CELL = 13; // px
const GAP = 3;
const LABEL_W = 30; // weekday-label gutter, px
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const LEVEL_CLASS = [
  "bg-[var(--color-surface-sunken)]",
  "bg-[var(--color-primary)]/30",
  "bg-[var(--color-primary)]/55",
  "bg-[var(--color-primary)]/80",
  "bg-[var(--color-primary-dim)]",
];
const LEVEL_TEXT = ["no practice", "1–4 practices", "5–9 practices", "10–19 practices", "20 or more practices"];
const RANGES: Array<{ id: HeatmapRange; label: string }> = [
  { id: "3m", label: "3 months" },
  { id: "6m", label: "6 months" },
  { id: "12m", label: "12 months" },
];

type Cell = HeatmapData["cells"][number];
type Slot = Cell | null; // null = a padding square before the first / after the last day so weeks line up Sunday to Saturday

/** Cells arrive one per calendar day in the learner's timezone; "2026-10-25" is a plain calendar date, so its weekday is read
 * from UTC arithmetic and cannot shift with the browser's own timezone. */
const weekday = (isoDate: string) => {
  const [y, m, d] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).getUTCDay();
};

function describeDate(isoDate: string) {
  const [y, m, d] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", { weekday: "short", year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

export function describeCell(cell: Cell): string {
  if (cell.count === 0) return `${describeDate(cell.date)}: no practice`;
  const parts = [
    cell.letters ? `${cell.letters} letter${cell.letters === 1 ? "" : "s"}` : null,
    cell.words ? `${cell.words} word${cell.words === 1 ? "" : "s"}` : null,
    cell.native ? `${cell.native} native sign${cell.native === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return `${describeDate(cell.date)}: ${cell.count} practice${cell.count === 1 ? "" : "s"} (${parts.join(", ")})`;
}

/** Lays the days out as weeks (columns) by weekday (rows), Sunday first, like a contribution graph. */
export function layoutWeeks(cells: Cell[]): Slot[][] {
  if (cells.length === 0) return [];
  const slots: Slot[] = [...Array<Slot>(weekday(cells[0].date)).fill(null), ...cells];
  while (slots.length % 7 !== 0) slots.push(null);
  const weeks: Slot[][] = [];
  for (let i = 0; i < slots.length; i += 7) weeks.push(slots.slice(i, i + 7));
  return weeks;
}

function monthLabels(weeks: Slot[][]): Array<{ column: number; text: string }> {
  const labels: Array<{ column: number; text: string }> = [];
  let lastMonth = -1;
  let lastColumn = -10;
  weeks.forEach((week, column) => {
    const first = week.find((slot): slot is Cell => slot !== null);
    if (!first) return;
    const month = Number(first.date.slice(5, 7)) - 1;
    if (month !== lastMonth && column - lastColumn >= 3) {
      labels.push({ column, text: MONTHS[month] });
      lastColumn = column;
    }
    lastMonth = month;
  });
  return labels;
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-[8.5rem] flex-1 rounded-2xl border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
      <p className="text-[0.7rem] font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">{label}</p>
      <p className="mt-0.5 font-display text-2xl font-bold text-[var(--color-ink)]">{value}</p>
      {sub && <p className="text-xs text-[var(--color-muted)]">{sub}</p>}
    </div>
  );
}

/** The first thing on the Progress page: a GitHub-style grid of real practice activity (letters, words and native signs) by day,
 * with a legend, month and weekday labels, hover/focus tooltips, keyboard navigation and honest loading/error/empty states. */
export function ActivityHeatmapCard({
  data,
  loading,
  error,
  range,
  onRangeChange,
  onRetry,
  onStart,
}: {
  data: HeatmapData | null;
  loading: boolean;
  error: string | null;
  range: HeatmapRange;
  onRangeChange: (range: HeatmapRange) => void;
  onRetry: () => void;
  onStart: () => void;
}) {
  const weeks = useMemo(() => layoutWeeks(data?.cells ?? []), [data]);
  const labels = useMemo(() => monthLabels(weeks), [weeks]);
  const [active, setActive] = useState<{ column: number; row: number } | null>(null);
  const [focusCell, setFocusCell] = useState<{ column: number; row: number } | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // On a narrow screen the grid scrolls sideways: start at the right-hand end so the most recent weeks (today) are what you see.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [data]);

  const summary = data?.summary;
  const hasActivity = (summary?.total_activity ?? 0) > 0;
  const lastColumn = weeks.length - 1;

  // The cell that holds the single tab stop: the most recent day with a cell (today, normally).
  const defaultFocus = useMemo(() => {
    for (let column = lastColumn; column >= 0; column -= 1) {
      for (let row = 6; row >= 0; row -= 1) if (weeks[column]?.[row]) return { column, row };
    }
    return { column: 0, row: 0 };
  }, [weeks, lastColumn]);
  const tabStop = focusCell ?? defaultFocus;

  function moveFocus(event: KeyboardEvent<HTMLDivElement>, column: number, row: number) {
    const delta: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    let target: { column: number; row: number } | null = null;
    if (delta[event.key]) {
      let c = column;
      let r = row;
      // step in the arrow's direction, skipping padding squares
      for (let guard = 0; guard < 400; guard += 1) {
        c += delta[event.key][0];
        r += delta[event.key][1];
        if (r < 0 || r > 6 || c < 0 || c > lastColumn) break;
        if (weeks[c]?.[r]) {
          target = { column: c, row: r };
          break;
        }
      }
    } else if (event.key === "Home" || event.key === "End") {
      const columns = event.key === "Home" ? [...weeks.keys()] : [...weeks.keys()].reverse();
      const found = columns.find((c) => weeks[c]?.[row]);
      if (found !== undefined) target = { column: found, row };
    }
    if (!target) return;
    event.preventDefault();
    setFocusCell(target);
    setActive(target);
    requestAnimationFrame(() => gridRef.current?.querySelector<HTMLElement>(`[data-cell="${target!.column}-${target!.row}"]`)?.focus());
  }

  const activeCell = active ? weeks[active.column]?.[active.row] ?? null : null;
  const tooltipLeft = active ? LABEL_W + active.column * (CELL + GAP) + CELL / 2 : 0;
  const tooltipTop = active ? 18 + active.row * (CELL + GAP) : 0;

  return (
    <Card className="space-y-4" data-testid="activity-heatmap-card">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <SectionLabel>Your practice activity</SectionLabel>
          <p className="mt-1 text-sm text-[var(--color-ink-soft)]" aria-live="polite">
            {loading
              ? "Loading your activity…"
              : error
                ? "Your activity could not be loaded."
                : hasActivity
                  ? `${summary!.total_activity} practice${summary!.total_activity === 1 ? "" : "s"} on ${summary!.active_days} day${summary!.active_days === 1 ? "" : "s"} in the last ${RANGES.find((r) => r.id === range)?.label.toLowerCase()}`
                  : "No practice recorded in this period yet."}
          </p>
        </div>
        <div role="group" aria-label="Heatmap range" className="flex gap-1.5">
          {RANGES.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-pressed={range === item.id}
              onClick={() => onRangeChange(item.id)}
              className={`min-h-9 rounded-full border-2 px-3 text-xs font-bold ${range === item.id ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]" : "border-[var(--color-border)] text-[var(--color-ink-soft)]"}`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-[var(--color-danger-soft)] px-4 py-3 text-sm font-semibold text-[var(--color-danger)]">
          <span>{error}</span>
          <SecondaryButton onClick={onRetry}>Try again</SecondaryButton>
        </div>
      )}

      {loading && !data && (
        <div aria-busy="true" aria-label="Loading activity grid" className="h-[9.5rem] animate-pulse rounded-2xl bg-[var(--color-surface-soft)]" />
      )}

      {data && (
        <div>
        <div ref={scrollRef} className="overflow-x-auto pb-1">
          <div className="relative" style={{ width: LABEL_W + weeks.length * (CELL + GAP) }}>
            {/* month labels */}
            <div className="relative h-4 text-[0.65rem] font-semibold text-[var(--color-muted)]" aria-hidden="true">
              {labels.map((label) => (
                <span key={`${label.column}-${label.text}`} className="absolute" style={{ left: LABEL_W + label.column * (CELL + GAP) }}>
                  {label.text}
                </span>
              ))}
            </div>
            <div
              ref={gridRef}
              role="grid"
              aria-label={`Practice activity by day, ${data.start} to ${data.end} (${data.timezone}). Use the arrow keys to move between days.`}
              aria-rowcount={7}
              aria-colcount={weeks.length}
              onMouseLeave={() => setActive(null)}
              className="mt-0.5"
            >
              {WEEKDAYS.map((name, row) => (
                <div key={name} role="row" className="flex items-center" style={{ height: CELL, marginBottom: GAP }}>
                  <span role="rowheader" className="text-[0.62rem] font-semibold text-[var(--color-muted)]" style={{ width: LABEL_W, position: "sticky", left: 0, zIndex: 1, background: "var(--color-surface)" }} aria-hidden={row % 2 === 0}>
                    {row % 2 === 1 ? name : ""}
                  </span>
                  {weeks.map((week, column) => {
                    const cell = week[row];
                    if (!cell) return <span key={column} role="gridcell" aria-hidden="true" style={{ width: CELL, height: CELL, marginRight: GAP }} />;
                    const text = describeCell(cell);
                    return (
                      <div
                        key={cell.date}
                        role="gridcell"
                        data-cell={`${column}-${row}`}
                        data-level={cell.intensity}
                        data-date={cell.date}
                        data-count={cell.count}
                        tabIndex={tabStop.column === column && tabStop.row === row ? 0 : -1}
                        aria-label={text}
                        onMouseEnter={() => setActive({ column, row })}
                        onFocus={() => {
                          setFocusCell({ column, row });
                          setActive({ column, row });
                        }}
                        onBlur={() => setActive((current) => (current && current.column === column && current.row === row ? null : current))}
                        onKeyDown={(event) => moveFocus(event, column, row)}
                        className={`rounded-[3px] outline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-ink)] ${LEVEL_CLASS[cell.intensity] ?? LEVEL_CLASS[0]}`}
                        style={{ width: CELL, height: CELL, marginRight: GAP }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>

            {activeCell && (
              <div
                role="tooltip"
                data-testid="heatmap-tooltip"
                className="pointer-events-none absolute z-10 -translate-x-1/2 whitespace-nowrap rounded-lg bg-[var(--color-ink)] px-2.5 py-1.5 text-xs font-semibold text-white shadow-lg"
                style={{ left: Math.max(70, tooltipLeft), top: active && active.row < 2 ? tooltipTop + CELL + 6 : tooltipTop - 34 }}
              >
                {describeCell(activeCell)}
              </div>
            )}
          </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.7rem] text-[var(--color-muted)]">
            <span className="flex items-center gap-1.5" role="img" aria-label={`Legend: ${LEVEL_TEXT.join(", ")}`}>
              Less
              {LEVEL_CLASS.map((cls, level) => (
                <span key={level} title={LEVEL_TEXT[level]} className={`inline-block rounded-[3px] ${cls}`} style={{ width: CELL - 2, height: CELL - 2 }} />
              ))}
              More
            </span>
            <span>Counts letter attempts, word sessions and native-sign attempts · days follow your timezone ({data.timezone})</span>
          </div>
        </div>
      )}

      {data && !hasActivity && !loading && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border-2 border-dashed border-[var(--color-border)] px-4 py-3">
          <p className="text-sm text-[var(--color-ink-soft)]">Nothing here yet. Your first practice lights up today&rsquo;s square.</p>
          <PrimaryButton onClick={onStart}>Start practicing</PrimaryButton>
        </div>
      )}

      {data && summary && (
        <div className="flex flex-wrap gap-3" aria-label="Activity summary">
          <Stat label="Current streak" value={`${summary.current_streak} day${summary.current_streak === 1 ? "" : "s"}`} sub={`Longest ${summary.longest_streak} day${summary.longest_streak === 1 ? "" : "s"}`} />
          <Stat label="Total practice" value={String(summary.total_activity)} sub={`${summary.active_days} active day${summary.active_days === 1 ? "" : "s"}`} />
          <Stat
            label="Letter accuracy"
            value={summary.letter_accuracy != null ? `${summary.letter_accuracy}%` : "—"}
            sub={summary.letter_attempts ? `${summary.letter_attempts} A–Z attempts` : "No A–Z attempts yet"}
          />
        </div>
      )}
    </Card>
  );
}
