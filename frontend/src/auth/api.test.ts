import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const BACKEND = "https://asl-quest-api.onrender.com";

async function loadApi() {
  vi.resetModules();
  return import("./api");
}

describe("API base URL (config.ts)", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("uses VITE_API_BASE_URL and strips trailing slashes", async () => {
    vi.stubEnv("VITE_API_BASE_URL", `${BACKEND}//`);
    vi.resetModules();
    const { API_BASE } = await import("../config");
    expect(API_BASE).toBe(BACKEND);
  });

  it("falls back to the dev proxy path when unset", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    vi.resetModules();
    const { API_BASE } = await import("../config");
    expect(API_BASE).toBe("/api");
  });
});

describe("auth requests", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", BACKEND);
    localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it("POSTs login as JSON to the backend, not the frontend origin", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ access_token: "t", token_type: "bearer" }), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ id: 1, username: "u", email: "u@example.com", created_at: "2026-01-01T00:00:00Z" }),
          { status: 200 },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    const { loginUser } = await loadApi();
    await loginUser("u@example.com", "secret-pass");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BACKEND}/auth/login`);
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ login: "u@example.com", password: "secret-pass" });
    expect(fetchMock.mock.calls[1][0]).toBe(`${BACKEND}/auth/me`);
  });

  it("shows the backend's message for invalid credentials", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Invalid username/email or password." }), { status: 401 })),
    );
    const { loginUser } = await loadApi();
    await expect(loginUser("a", "b")).rejects.toThrow("Invalid username/email or password.");
  });

  it("handles a non-JSON error body (e.g. a host's plain-text 404) without crashing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("The page could not be found", { status: 404 })));
    const { loginUser } = await loadApi();
    await expect(loginUser("a", "b")).rejects.toThrow("Request failed (404).");
  });

  it("turns a network/CORS failure into a readable error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const { loginUser } = await loadApi();
    await expect(loginUser("a", "b")).rejects.toThrow(/Can't reach the ASL Quest server/);
  });

  it("abandons a login that never answers and reports a retryable timeout", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        })),
      );
      const { loginUser, AUTH_TIMEOUT_MS } = await loadApi();
      const outcome = expect(loginUser("a", "b")).rejects.toThrow(/taking too long.*try again/i);
      await vi.advanceTimersByTimeAsync(AUTH_TIMEOUT_MS + 1);
      await outcome;
    } finally {
      vi.useRealTimers();
    }
  });

  it("abandons a registration that never answers", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        })),
      );
      const { registerUser, AUTH_TIMEOUT_MS } = await loadApi();
      const outcome = expect(registerUser({ username: "u", email: "u@example.com", password: "longenough1" })).rejects.toThrow(/taking too long/i);
      await vi.advanceTimersByTimeAsync(AUTH_TIMEOUT_MS + 1);
      await outcome;
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows the first message of a FastAPI 422 validation error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: "String should have at least 8 characters" }] }), { status: 422 })),
    );
    const { registerUser } = await loadApi();
    await expect(registerUser({ username: "u", email: "u@example.com", password: "x" })).rejects.toThrow(
      "String should have at least 8 characters",
    );
  });
});
