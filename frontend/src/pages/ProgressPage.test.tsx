import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ProgressPage } from "./ProgressPage";
import * as analytics from "../analytics/api";
import type { HeatmapData } from "../analytics/api";

vi.mock("../analytics/api", async () => {
  const actual = await vi.importActual<typeof import("../analytics/api")>("../analytics/api");
  return { ...actual, fetchHeatmap: vi.fn(), fetchAnalyticsDashboard: vi.fn(), fetchPracticeHistory: vi.fn(), fetchLetterDetail: vi.fn() };
});
vi.mock("../native/api", () => ({ getNativeProgress: vi.fn().mockRejectedValue(new Error("none")) }));
vi.mock("../words/api", () => ({ fetchWord: vi.fn() }));
const startPractice = vi.fn();
vi.mock("../game/GameContext", () => ({ useGame: () => ({ startPractice, startWordPractice: vi.fn() }) }));

const heat = (total = 12): HeatmapData => ({
  range: "12m",
  timezone: "Asia/Kolkata",
  start: "2026-10-23",
  end: "2026-10-25",
  cells: ["2026-10-23", "2026-10-24", "2026-10-25"].map((date, i) => ({
    date, count: i === 2 ? total : 0, letters: i === 2 ? total : 0, words: 0, native: 0, attempts: 0, correct: 0, accuracy: null, xp: 0, intensity: i === 2 ? 3 : 0,
  })),
  summary: { total_activity: total, active_days: total ? 1 : 0, current_streak: total ? 1 : 0, longest_streak: total ? 1 : 0, letter_accuracy: total ? 75 : null, letter_attempts: total },
});

beforeEach(() => {
  vi.mocked(analytics.fetchPracticeHistory).mockResolvedValue({ items: [], page: 1, page_size: 10, total: 0 } as never);
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ProgressPage: heatmap first", () => {
  it("puts the heatmap card before everything else, and shows it while the heavier analytics are still loading", async () => {
    vi.mocked(analytics.fetchHeatmap).mockResolvedValue(heat());
    vi.mocked(analytics.fetchAnalyticsDashboard).mockReturnValue(new Promise(() => {})); // never arrives
    render(<ProgressPage />);

    expect(await screen.findByRole("grid")).toBeInTheDocument();
    expect(screen.getByText("Loading learning analytics...")).toBeInTheDocument();
    const card = screen.getByTestId("activity-heatmap-card");
    const loading = screen.getByText("Loading learning analytics...");
    // DOM order: the heatmap card precedes the rest of the page content
    expect(card.compareDocumentPosition(loading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByLabelText("Activity summary")).toHaveTextContent("75%");
  });

  it("requests the heatmap on its own and re-requests it when the range changes", async () => {
    vi.mocked(analytics.fetchHeatmap).mockResolvedValue(heat());
    vi.mocked(analytics.fetchAnalyticsDashboard).mockReturnValue(new Promise(() => {}));
    render(<ProgressPage />);
    await screen.findByRole("grid");
    expect(analytics.fetchHeatmap).toHaveBeenCalledWith("12m");
    fireEvent.click(screen.getByRole("button", { name: "3 months" }));
    await waitFor(() => expect(analytics.fetchHeatmap).toHaveBeenLastCalledWith("3m"));
    // the heavier dashboard is not re-fetched just because the heatmap range changed
    expect(analytics.fetchAnalyticsDashboard).toHaveBeenCalledTimes(1);
  });

  it("still shows the heatmap (with an honest empty state) for a user who has no analytics yet", async () => {
    vi.mocked(analytics.fetchHeatmap).mockResolvedValue(heat(0));
    vi.mocked(analytics.fetchAnalyticsDashboard).mockResolvedValue({ overview: { has_data: false } } as never);
    render(<ProgressPage />);
    expect(await screen.findByText("Your analytics are waiting for you")).toBeInTheDocument();
    expect(screen.getByRole("grid")).toBeInTheDocument();
    expect(screen.getByText(/No practice recorded in this period yet/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start practicing" }));
    expect(startPractice).toHaveBeenCalled();
  });

  it("a failed heatmap request shows an error and a retry without breaking the rest of the page", async () => {
    vi.mocked(analytics.fetchHeatmap).mockRejectedValueOnce(new Error("boom")).mockResolvedValue(heat());
    vi.mocked(analytics.fetchAnalyticsDashboard).mockResolvedValue({ overview: { has_data: false } } as never);
    render(<ProgressPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your activity right now.");
    expect(await screen.findByText("Your analytics are waiting for you")).toBeInTheDocument(); // the rest of the page still rendered
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("grid")).toBeInTheDocument();
    expect(analytics.fetchHeatmap).toHaveBeenCalledTimes(2);
  });
});
