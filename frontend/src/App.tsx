import { useEffect, useRef, useState } from "react";
import { healthCheck, type HealthResponse } from "./api";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { MigrationPrompt } from "./components/MigrationPrompt";
import { GameProvider, useGame } from "./game/GameContext";
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
  const { page, hydrateState, practiceKind } = useGame();
  const { migrationOffer, dismissMigration, importLocalProgress } = useAuth();
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
        setHealthError("Backend unavailable. Start FastAPI on port 8000.");
      });
  }, []);

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
      <AppShell ready={ready} backendNote={backendNote}>
        {page === "home" && <HomePage />}
        {page === "learn" && <LearnPage />}
        {page === "words" && <WordsPage />}
        {page === "practice" &&
          (practiceKind === "word" ? (
            <WordPracticePage ready={ready} backendError={healthError} />
          ) : (
            <PracticePage ready={ready} backendError={healthError} />
          ))}
        {page === "native" && <NativeSignsPage />}
        {page === "native-practice" && <NativeSignPracticePage />}
        {page === "challenges" && <ChallengesPage />}
        {page === "progress" && <ProgressPage />}
        {page === "achievements" && <AchievementsPage />}
        {page === "assistant" && <ChatbotPage />}
        {page === "profile" && <ProfilePage />}
        {page === "settings" && <SettingsPage />}
        {page === "admin" && <AdminPage />}
        {page === "mentor-dashboard" && <MentorDashboard />}
      </AppShell>
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
