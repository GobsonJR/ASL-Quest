import { useEffect, useRef, useState } from "react";
import { healthCheck, type HealthResponse } from "./api";
import { BACKEND_NOT_CONFIGURED } from "./config";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { MigrationPrompt } from "./components/MigrationPrompt";
import { SignoCompanion } from "./components/signo/SignoCompanion";
import { SignoProvider } from "./components/signo/SignoContext";
import { GameProvider, useGame } from "./game/GameContext";
import { AlphabetPage } from "./pages/AlphabetPage";
import { AchievementsPage } from "./pages/AchievementsPage";
import { LoginPage, RegisterPage } from "./pages/AuthPages";
import { ChallengesPage } from "./pages/ChallengesPage";
import { HomePage } from "./pages/HomePage";
import { LearnPage } from "./pages/LearnPage";
import { NativeSignsPage } from "./native/NativeSignsPage";
import { NativeSignPracticePage } from "./native/NativeSignPracticePage";
import { ChatbotPage } from "./chatbot/ChatbotPage";
import { PracticePage } from "./pages/PracticePage";
import { WordPracticePage } from "./pages/WordPracticePage";
import { WordsPage } from "./pages/WordsPage";
import { ProgressPage } from "./pages/ProgressPage";
import { ToastProvider, useToast } from "./components/Toast";
import { AdminPage } from "./pages/AdminPage";
import { MentorDashboard } from "./admin/MentorDashboard";
import { ProfilePage } from "./pages/ProfilePage";
import { SettingsPage } from "./pages/SettingsPage";
import { ADMIN_PAGES, isAdminRole } from "./auth/roles";

function SyncNotifier() {
  const { syncStatus } = useAuth();
  const { push } = useToast();
  const previous = useRef(syncStatus);

  useEffect(() => {
    if (previous.current === "syncing" && syncStatus === "synced") {
      push("Progress synchronized", "success");
    }
    if (previous.current === "syncing" && syncStatus === "error") {
      push("Could not save progress — we'll retry.", "warning");
    }
    previous.current = syncStatus;
  }, [syncStatus, push]);

  return null;
}

function AuthenticatedApp() {
  const { page, navigate, hydrateState, practiceKind } = useGame();
  const { user, migrationOffer, dismissMigration, importLocalProgress, landingPage, clearLandingPage } = useAuth();
  const isAdmin = isAdminRole(user?.role);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    healthCheck()
      .then((data) => {
        setHealth(data);
        setHealthError(null);
      })
      .catch(() => {
        setHealth(null);
        setHealthError(
          BACKEND_NOT_CONFIGURED
            ? "The recognition service is not configured for this deployment (VITE_API_BASE_URL is not set)."
            : "Backend unavailable. Start FastAPI on port 8000."
        );
      });
  }, []);

  // Right after sign-in an administrator lands in the Admin workspace (once); everyone else lands where they always did.
  useEffect(() => {
    if (!landingPage) return;
    if (landingPage === "admin" && isAdmin) navigate("admin");
    clearLandingPage();
  }, [landingPage, isAdmin, navigate, clearLandingPage]);

  // Admin pages are not rendered for anyone else. (This is only the visible half of the rule: the API refuses them too.)
  const blockedPage = ADMIN_PAGES.includes(page) && !isAdmin;
  useEffect(() => {
    if (blockedPage) navigate("home");
  }, [blockedPage, navigate]);

  const ready = Boolean(health?.model_loaded);
  const backendNote = health
    ? `Ready on ${health.device}${health.gpu_name ? ` · ${health.gpu_name}` : ""}`
    : healthError ?? "Checking backend...";

  async function handleImport() {
    setImporting(true);
    try {
      const imported = await importLocalProgress();
      if (imported) hydrateState(imported);
    } finally {
      setImporting(false);
    }
  }

  return (
    <>
      <SyncNotifier />
      {migrationOffer && (
        <MigrationPrompt onImport={handleImport} onDismiss={dismissMigration} loading={importing} />
      )}
      <SignoProvider>
      <AppShell ready={ready} backendNote={backendNote}>
        {page === "home" && <HomePage />}
        {page === "learn" && <LearnPage />}
        {page === "alphabet" && <AlphabetPage />}
        {page === "words" && <WordsPage />}
        {page === "practice" &&
          (practiceKind === "word" ? (
            <WordPracticePage ready={ready} backendError={healthError} experimentalLandmark={health?.experimental_landmark ?? null} />
          ) : (
            <PracticePage ready={ready} backendError={healthError} experimentalLandmark={health?.experimental_landmark ?? null} />
          ))}
        {page === "native" && <NativeSignsPage />}
        {page === "native-practice" && <NativeSignPracticePage />}
        {page === "challenges" && <ChallengesPage />}
        {page === "progress" && <ProgressPage />}
        {page === "achievements" && <AchievementsPage />}
        {page === "assistant" && <ChatbotPage />}
        {page === "profile" && <ProfilePage />}
        {page === "settings" && <SettingsPage />}
        {page === "admin" && isAdmin && <AdminPage />}
        {page === "mentor-dashboard" && isAdmin && <MentorDashboard />}
      </AppShell>
      <SignoCompanion />
      </SignoProvider>
    </>
  );
}

function AppRouter() {
  const { user, loading, authView } = useAuth();

  if (loading) {
    return (
      <div className="grid min-h-screen place-items-center text-[var(--color-ink-soft)]">
        Loading ASL Quest...
      </div>
    );
  }

  if (!user) {
    return authView === "register" ? <RegisterPage /> : <LoginPage />;
  }

  return (
    <ToastProvider>
      <GameProvider>
        <AuthenticatedApp />
      </GameProvider>
    </ToastProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  );
}
