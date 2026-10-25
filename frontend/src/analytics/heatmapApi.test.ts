import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchHeatmap } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("fetchHeatmap", () => {
  it("sends the range, the learner's timezone and their bearer token", async () => {
    localStorage.setItem("asl-quest-token", "tok");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ cells: [] }) });
    vi.stubGlobal("fetch", fetchMock);

    await fetchHeatmap("6m", "Asia/Kolkata");
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/analytics\/heatmap\?range=6m&tz=Asia%2FKolkata$/);
    expect((init.headers as Headers).get("Authorization")).toBe("Bearer tok");
  });

  it("defaults to the browser's own timezone", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ cells: [] }) });
    vi.stubGlobal("fetch", fetchMock);
    await fetchHeatmap("3m");
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(String(fetchMock.mock.calls[0][0])).toContain(`tz=${encodeURIComponent(zone)}`);
  });

  it("turns a failed response into an error the page can show", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) }));
    await expect(fetchHeatmap("3m", "UTC")).rejects.toThrow(/Failed to load analytics/);
  });
});
