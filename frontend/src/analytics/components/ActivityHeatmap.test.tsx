import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { ActivityHeatmapCard, describeCell, layoutWeeks } from "./ActivityHeatmap";
import type { HeatmapData } from "../api";

afterEach(() => cleanup());

type Cell = HeatmapData["cells"][number];

function cell(date: string, count = 0, parts: Partial<Cell> = {}): Cell {
  return { date, count, letters: count, words: 0, native: 0, attempts: count, correct: 0, accuracy: null, xp: 0, intensity: count === 0 ? 0 : count < 5 ? 1 : count < 10 ? 2 : count < 20 ? 3 : 4, ...parts };
}

/** 10 consecutive days starting Wednesday 21 Oct 2026 (a Wednesday), so the grid needs 3 leading padding squares. */
function sample(overrides: Partial<HeatmapData> = {}, cells?: Cell[]): HeatmapData {
  const days = cells ?? Array.from({ length: 10 }, (_, i) => cell(`2026-10-${String(21 + i).padStart(2, "0")}`, 0));
  return {
    range: "3m",
    timezone: "Asia/Kolkata",
    start: days[0].date,
    end: days[days.length - 1].date,
    cells: days,
    summary: { total_activity: 0, active_days: 0, current_streak: 0, longest_streak: 0, letter_accuracy: null, letter_attempts: 0 },
    ...overrides,
  };
}

function renderCard(props: Partial<React.ComponentProps<typeof ActivityHeatmapCard>> = {}) {
  const handlers = { onRangeChange: vi.fn(), onRetry: vi.fn(), onStart: vi.fn() };
  render(<ActivityHeatmapCard data={sample()} loading={false} error={null} range="12m" {...handlers} {...props} />);
  return handlers;
}

describe("layoutWeeks", () => {
  it("aligns days to Sunday-first weeks with padding before the first and after the last day", () => {
    const weeks = layoutWeeks(sample().cells);
    expect(weeks.every((week) => week.length === 7)).toBe(true);
    expect(weeks[0].slice(0, 3)).toEqual([null, null, null]); // 21 Oct 2026 is a Wednesday: Sun, Mon, Tue are padding
    expect(weeks[0][3]?.date).toBe("2026-10-21");
    expect(weeks.flat().filter(Boolean)).toHaveLength(10); // no day invented, none dropped
    expect(weeks[1][0]?.date).toBe("2026-10-25"); // Sunday starts a new column
  });

  it("is empty for no data", () => {
    expect(layoutWeeks([])).toEqual([]);
  });
});

describe("describeCell", () => {
  it("states the date and the real count, with the breakdown", () => {
    expect(describeCell(cell("2026-10-22", 4, { letters: 2, words: 1, native: 1 }))).toMatch(/Oct 22, 2026: 4 practices \(2 letters, 1 word, 1 native sign\)/);
    expect(describeCell(cell("2026-10-22", 1, { letters: 1 }))).toMatch(/1 practice \(1 letter\)/);
    expect(describeCell(cell("2026-10-23", 0))).toMatch(/Oct 23, 2026: no practice/);
  });
});

describe("ActivityHeatmapCard", () => {
  it("renders one square per real day (padding is hidden from assistive tech) with date labels and a legend", () => {
    renderCard();
    const grid = screen.getByRole("grid");
    expect(grid).toHaveAccessibleName(/2026-10-21 to 2026-10-30 \(Asia\/Kolkata\)/);
    expect(grid.querySelectorAll("[data-date]")).toHaveLength(10);
    expect(screen.getByText("Oct")).toBeInTheDocument(); // month label
    expect(screen.getByText("Mon")).toBeInTheDocument(); // weekday labels
    expect(screen.getByText("Wed")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Legend: no practice, 1–4 practices/ })).toBeInTheDocument();
  });

  it("colours squares from the real counts and never fills empty days", () => {
    const cells = [cell("2026-10-24", 0), cell("2026-10-25", 3), cell("2026-10-26", 12), cell("2026-10-27", 30)];
    renderCard({ data: sample({ summary: { total_activity: 45, active_days: 3, current_streak: 3, longest_streak: 3, letter_accuracy: 80, letter_attempts: 45 } }, cells) });
    // the DOM is row-major (one row per weekday), so compare in date order
    const levels = Array.from(screen.getByRole("grid").querySelectorAll("[data-date]"))
      .map((n) => [n.getAttribute("data-date"), n.getAttribute("data-level"), n.getAttribute("data-count")])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])));
    expect(levels).toEqual([["2026-10-24", "0", "0"], ["2026-10-25", "1", "3"], ["2026-10-26", "3", "12"], ["2026-10-27", "4", "30"]]);
  });

  it("shows a tooltip with the date and actual count on hover and on keyboard focus", () => {
    renderCard({ data: sample({}, [cell("2026-10-24", 0), cell("2026-10-25", 7, { letters: 5, words: 2 })]) });
    const square = screen.getByRole("gridcell", { name: /Oct 25, 2026: 7 practices \(5 letters, 2 words\)/ });
    fireEvent.mouseEnter(square);
    expect(screen.getByTestId("heatmap-tooltip")).toHaveTextContent(/Oct 25, 2026: 7 practices/);
    fireEvent.mouseLeave(screen.getByRole("grid"));
    expect(screen.queryByTestId("heatmap-tooltip")).toBeNull();
    fireEvent.focus(square);
    expect(screen.getByTestId("heatmap-tooltip")).toHaveTextContent(/7 practices/);
  });

  it("is a single tab stop and moves between days with the arrow keys", () => {
    renderCard({ data: sample({}, [cell("2026-10-24", 1), cell("2026-10-25", 2), cell("2026-10-26", 3), cell("2026-10-27", 4)]) });
    const byDate = (date: string) => screen.getByRole("grid").querySelector(`[data-date="${date}"]`) as HTMLElement;
    const squares = screen.getAllByRole("gridcell").filter((c) => c.hasAttribute("data-date"));
    expect(squares.filter((s) => s.getAttribute("tabindex") === "0")).toHaveLength(1);
    expect(byDate("2026-10-27")).toHaveAttribute("tabindex", "0"); // starts on the most recent day
    // 25 Oct is a Sunday, so 26 Oct (Monday) is the next row down in the same column
    byDate("2026-10-25").focus();
    fireEvent.keyDown(byDate("2026-10-25"), { key: "ArrowDown" });
    return new Promise<void>((resolve) =>
      requestAnimationFrame(() => {
        expect(document.activeElement).toBe(byDate("2026-10-26"));
        resolve();
      })
    );
  });

  it("shows the streak, total practice and accuracy beside the grid", () => {
    renderCard({
      data: sample({ summary: { total_activity: 42, active_days: 9, current_streak: 4, longest_streak: 6, letter_accuracy: 87.5, letter_attempts: 30 } }),
    });
    const summary = screen.getByLabelText("Activity summary");
    expect(summary).toHaveTextContent("Current streak");
    expect(summary).toHaveTextContent("4 days");
    expect(summary).toHaveTextContent("Longest 6 days");
    expect(summary).toHaveTextContent("Total practice");
    expect(summary).toHaveTextContent("42");
    expect(summary).toHaveTextContent("9 active days");
    expect(summary).toHaveTextContent("87.5%");
    expect(summary).toHaveTextContent("30 A–Z attempts");
  });

  it("explains an empty history honestly and offers a way to start, without inventing activity", () => {
    const { onStart } = renderCard();
    expect(screen.getByText(/No practice recorded in this period yet/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start practicing" }));
    expect(onStart).toHaveBeenCalled();
    expect(screen.getByLabelText("Activity summary")).toHaveTextContent("No A–Z attempts yet");
    expect(Array.from(screen.getByRole("grid").querySelectorAll("[data-count]")).every((n) => n.getAttribute("data-count") === "0")).toBe(true);
  });

  it("a user with a little activity still gets the full grid and a singular-aware summary", () => {
    renderCard({ data: sample({ summary: { total_activity: 1, active_days: 1, current_streak: 1, longest_streak: 1, letter_accuracy: 100, letter_attempts: 1 } }, [cell("2026-10-24", 0), cell("2026-10-25", 1)]) });
    expect(screen.getByText("1 practice on 1 day in the last 12 months")).toBeInTheDocument();
    expect(screen.getByLabelText("Activity summary")).toHaveTextContent("1 day");
    expect(screen.queryByRole("button", { name: "Start practicing" })).toBeNull();
  });

  it("shows a loading skeleton, not an empty grid, while the first request is in flight", () => {
    renderCard({ data: null, loading: true });
    expect(screen.getByLabelText("Loading activity grid")).toHaveAttribute("aria-busy", "true");
    expect(screen.queryByRole("grid")).toBeNull();
    expect(screen.getByText("Loading your activity…")).toBeInTheDocument();
  });

  it("reports a failure with a retry, and keeps showing the last good grid if there is one", () => {
    const { onRetry } = renderCard({ data: null, error: "Couldn't load your activity right now." });
    expect(screen.getByRole("alert")).toHaveTextContent("Couldn't load your activity right now.");
    fireEvent.click(within(screen.getByRole("alert")).getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalled();
    expect(screen.queryByRole("grid")).toBeNull(); // nothing is faked while there is no data
  });

  it("switches the range", () => {
    const { onRangeChange } = renderCard();
    expect(screen.getByRole("button", { name: "12 months" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "3 months" }));
    expect(onRangeChange).toHaveBeenCalledWith("3m");
  });
});
