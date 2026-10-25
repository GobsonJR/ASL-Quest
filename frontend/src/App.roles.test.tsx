import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

// The role rules in App.tsx: where an administrator lands after signing in, and that admin pages never render for anyone else.
// Everything heavy is stubbed; the real App routing code runs.
const navigate = vi.fn();
const clearLandingPage = vi.fn();
let mockPage = "home";
let mockRole = "student";
let mockLanding: string | null = null;

vi.mock("./api", () => ({ healthCheck: vi.fn().mockResolvedValue({ model_loaded: true, device: "cpu", gpu_name: null }) }));
vi.mock("./auth/AuthContext", () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useAuth: () => ({
    user: { id: 1, username: "sam", email: "sam@example.com", role: mockRole },
    loading: false,
    authView: "login",
    migrationOffer: false,
    dismissMigration: vi.fn(),
    importLocalProgress: vi.fn(),
    syncStatus: "idle",
    landingPage: mockLanding,
    clearLandingPage,
  }),
}));
vi.mock("./game/GameContext", () => ({
  GameProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useGame: () => ({ page: mockPage, navigate, hydrateState: vi.fn(), practiceKind: "letter" }),
}));
vi.mock("./components/Toast", () => ({ ToastProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>, useToast: () => ({ push: vi.fn() }) }));
vi.mock("./components/AppShell", () => ({ AppShell: ({ children }: { children: React.ReactNode }) => <main>{children}</main> }));
vi.mock("./components/MigrationPrompt", () => ({ MigrationPrompt: () => null }));
vi.mock("./components/signo/SignoCompanion", () => ({ SignoCompanion: () => null }));
vi.mock("./components/signo/SignoContext", () => ({ SignoProvider: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
vi.mock("./pages/AdminPage", () => ({ AdminPage: () => <p>ADMIN WORKSPACE</p> }));
vi.mock("./admin/MentorDashboard", () => ({ MentorDashboard: () => <p>MENTOR DASHBOARD</p> }));
for (const [path, name] of [
  ["./pages/AlphabetPage", "AlphabetPage"], ["./pages/AchievementsPage", "AchievementsPage"], ["./pages/AuthPages", "LoginPage"],
  ["./pages/ChallengesPage", "ChallengesPage"], ["./pages/HomePage", "HomePage"], ["./pages/LearnPage", "LearnPage"],
  ["./native/NativeSignsPage", "NativeSignsPage"], ["./native/NativeSignPracticePage", "NativeSignPracticePage"],
  ["./chatbot/ChatbotPage", "ChatbotPage"], ["./pages/PracticePage", "PracticePage"], ["./pages/WordPracticePage", "WordPracticePage"],
  ["./pages/WordsPage", "WordsPage"], ["./pages/ProgressPage", "ProgressPage"], ["./pages/ProfilePage", "ProfilePage"], ["./pages/SettingsPage", "SettingsPage"],
] as const) {
  vi.doMock(path, () => ({ [name]: () => <p>{name}</p>, RegisterPage: () => null }));
}

async function renderApp() {
  const { default: App } = await import("./App");
  return render(<App />);
}

beforeEach(() => {
  vi.resetModules();
  navigate.mockReset();
  clearLandingPage.mockReset();
  mockPage = "home";
  mockRole = "student";
  mockLanding = null;
});
afterEach(() => cleanup());

describe("login routing", () => {
  it("opens the Admin workspace for an administrator who just signed in, once", async () => {
    mockRole = "admin";
    mockLanding = "admin";
    await renderApp();
    await waitFor(() => expect(navigate).toHaveBeenCalledWith("admin"));
    expect(clearLandingPage).toHaveBeenCalled();
  });

  it("leaves a regular learner on the learning side and never navigates them to admin", async () => {
    mockLanding = null;
    await renderApp();
    expect(navigate).not.toHaveBeenCalledWith("admin");
  });

  it("ignores an 'admin' landing hint for a non-admin (the hint alone grants nothing)", async () => {
    mockRole = "student";
    mockLanding = "admin";
    await renderApp();
    await waitFor(() => expect(clearLandingPage).toHaveBeenCalled());
    expect(navigate).not.toHaveBeenCalledWith("admin");
  });
});

describe("admin pages are not reachable by regular users", () => {
  for (const page of ["admin", "mentor-dashboard"]) {
    it(`does not render ${page} for a learner and sends them home`, async () => {
      mockPage = page;
      mockRole = "student";
      await renderApp();
      expect(screen.queryByText("ADMIN WORKSPACE")).toBeNull();
      expect(screen.queryByText("MENTOR DASHBOARD")).toBeNull();
      await waitFor(() => expect(navigate).toHaveBeenCalledWith("home"));
    });
  }

  it("renders the Admin workspace and the Mentor dashboard for an administrator", async () => {
    mockRole = "admin";
    mockPage = "admin";
    await renderApp();
    expect(await screen.findByText("ADMIN WORKSPACE")).toBeInTheDocument();
    cleanup();
    vi.resetModules();
    mockPage = "mentor-dashboard";
    await renderApp();
    expect(await screen.findByText("MENTOR DASHBOARD")).toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalledWith("home");
  });
});
