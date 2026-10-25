import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

// Practice readiness as App.tsx wires it from the backend's /health answer. The real App code runs; the pages are stubbed
// to print the `ready` / `backendError` props they receive, so this fails if the wiring ever ignores the landmark model again.
vi.setConfig({ testTimeout: 30_000 }); // the first dynamic import of the whole App module is slow on a cold transform cache
const healthCheck = vi.fn();
vi.mock("./api", () => ({ healthCheck: (...args: unknown[]) => healthCheck(...args) }));
vi.mock("./auth/AuthContext", () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useAuth: () => ({
    user: { id: 1, username: "sam", email: "sam@example.com", role: "student" }, loading: false, authView: "login", migrationOffer: false,
    dismissMigration: vi.fn(), importLocalProgress: vi.fn(), syncStatus: "idle", landingPage: null, clearLandingPage: vi.fn(),
  }),
}));
vi.mock("./game/GameContext", () => ({
  GameProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useGame: () => ({ page: "practice", navigate: vi.fn(), hydrateState: vi.fn(), practiceKind: "letter" }),
}));
vi.mock("./components/Toast", () => ({ ToastProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>, useToast: () => ({ push: vi.fn() }) }));
vi.mock("./components/AppShell", () => ({ AppShell: ({ children }: { children: React.ReactNode }) => <main>{children}</main> }));
vi.mock("./components/MigrationPrompt", () => ({ MigrationPrompt: () => null }));
vi.mock("./components/signo/SignoCompanion", () => ({ SignoCompanion: () => null }));
vi.mock("./components/signo/SignoContext", () => ({ SignoProvider: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
vi.mock("./pages/PracticePage", () => ({
  PracticePage: ({ ready, backendError }: { ready: boolean; backendError: string | null }) => (
    <p data-testid="practice">{ready ? "READY" : "NOT READY"} | {backendError ?? "no message"}</p>
  ),
}));

const base = {
  status: "degraded", model_loaded: false, model_path: null, classes: [], error: "No trained model found.", device: "cpu", cuda_available: false,
  gpu_name: null, cuda_version: null, gpu_memory_gb: null, hand_detection_available: false, hand_detection_error: null,
};
const landmark = (loaded: boolean) => ({ loaded, model_file: null, model: null, trained_on: null, error: null });

async function renderApp() {
  const { default: App } = await import("./App");
  return render(<App />);
}

beforeEach(() => {
  vi.resetModules();
  healthCheck.mockReset();
});
afterEach(() => cleanup());

describe("Practice readiness from /health", () => {
  it("is ready when the ResNet is unavailable but the experimental landmark model is loaded", async () => {
    healthCheck.mockResolvedValue({ ...base, experimental_landmark: landmark(true) });
    await renderApp();
    expect(await screen.findByText(/^READY \|/)).toBeInTheDocument();
    expect(screen.getByTestId("practice")).toHaveTextContent("no message");
  });

  it("is NOT ready, with an honest reason, when neither recognizer is loaded", async () => {
    healthCheck.mockResolvedValue({ ...base, experimental_landmark: landmark(false) });
    await renderApp();
    const practice = await screen.findByText(/no sign-recognition model is loaded/);
    expect(practice).toHaveTextContent("NOT READY");
    expect(practice).not.toHaveTextContent(/8000/);
  });

  it("is NOT ready when the API cannot be reached, and does not send a visitor to localhost", async () => {
    healthCheck.mockRejectedValue(new Error("Backend is unavailable."));
    await renderApp();
    const practice = await screen.findByText(/Can't reach the recognition service|Backend unavailable/);
    expect(practice).toHaveTextContent("NOT READY");
  });

  it("says it is checking (not 'start the backend') while /health is still pending", async () => {
    healthCheck.mockReturnValue(new Promise(() => {}));
    await renderApp();
    expect(await screen.findByText(/NOT READY \| Checking the recognition service/)).toBeInTheDocument();
  });
});
