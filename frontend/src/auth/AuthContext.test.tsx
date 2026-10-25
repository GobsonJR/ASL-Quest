import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render } from "@testing-library/react";
import { AuthProvider, useAuth } from "./AuthContext";
import * as api from "./api";
import type { UserProfile } from "./api";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    restoreSession: vi.fn(),
    loginUser: vi.fn(),
    registerUser: vi.fn(),
    logoutUser: vi.fn(),
    fetchMigrationOffer: vi.fn(),
  };
});

const profile = (role: string): UserProfile => ({ id: 7, username: "sam", email: "sam@example.com", role, created_at: "2026-01-01T00:00:00Z" });

let auth!: ReturnType<typeof useAuth>;
function Probe() {
  auth = useAuth();
  return null;
}

async function mount() {
  render(
    <AuthProvider>
      <Probe />
    </AuthProvider>
  );
  await act(async () => {});
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.restoreSession).mockResolvedValue(null);
  vi.mocked(api.fetchMigrationOffer).mockResolvedValue({ should_offer: false, reason: "server_progress_exists" });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("login routing by role", () => {
  it("sends an administrator to the Admin workspace right after signing in (role from the backend account data)", async () => {
    vi.mocked(api.loginUser).mockResolvedValue(profile("admin"));
    await mount();
    expect(auth.landingPage).toBeNull();
    await act(async () => auth.login("sam", "correct-horse-1"));
    expect(auth.user?.role).toBe("admin");
    expect(auth.landingPage).toBe("admin");
    act(() => auth.clearLandingPage());
    expect(auth.landingPage).toBeNull(); // consumed once, so navigating elsewhere is never overridden
  });

  it("keeps a regular learner on the normal learning landing page", async () => {
    vi.mocked(api.loginUser).mockResolvedValue(profile("student"));
    await mount();
    await act(async () => auth.login("sam", "correct-horse-1"));
    expect(auth.user?.role).toBe("student");
    expect(auth.landingPage).toBeNull();
  });

  it("registration never lands anyone in the Admin workspace", async () => {
    vi.mocked(api.registerUser).mockResolvedValue(profile("student"));
    await mount();
    await act(async () => auth.register({ username: "sam", email: "sam@example.com", password: "correct-horse-1" }));
    expect(auth.landingPage).toBeNull();
    // the sign-up call carries exactly username, email and password: there is no way to send a role from the form
    expect(vi.mocked(api.registerUser).mock.calls[0][0]).toEqual({ username: "sam", email: "sam@example.com", password: "correct-horse-1" });
  });

  it("clears the landing page on logout", async () => {
    vi.mocked(api.loginUser).mockResolvedValue(profile("admin"));
    await mount();
    await act(async () => auth.login("sam", "correct-horse-1"));
    act(() => auth.logout());
    expect(auth.user).toBeNull();
    expect(auth.landingPage).toBeNull();
  });

  it("restoring a saved session does not force a landing page (a refresh keeps you where you are)", async () => {
    vi.mocked(api.restoreSession).mockResolvedValue(profile("admin"));
    await mount();
    expect(auth.user?.role).toBe("admin");
    expect(auth.landingPage).toBeNull();
  });
});
