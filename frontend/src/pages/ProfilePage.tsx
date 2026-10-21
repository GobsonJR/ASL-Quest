import { useEffect, useState } from "react";
import { fetchProfile, type FullUserProfile } from "../auth/api";
import {
  Avatar,
  Divider,
  PageHeader,
  PageLayout,
  PrimaryButton,
  SecondaryButton,
  SectionHeader,
  StatGroup,
} from "../components/AppShell";
import { useGame } from "../game/GameContext";

export function ProfilePage() {
  const { navigate, startPractice } = useGame();
  const [profile, setProfile] = useState<FullUserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile()
      .then(setProfile)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="grid min-h-[40vh] place-items-center text-[var(--color-ink-soft)]">Loading profile...</div>;
  if (!profile) return <div className="text-[var(--color-streak)]">Unable to load profile.</div>;

  return (
    <PageLayout>
      <div className="flex flex-col gap-6 md:flex-row md:items-center">
        <Avatar name={profile.avatar_initial || profile.username} size="lg" />
        <div>
          <PageHeader eyebrow="👤 Your profile" title={profile.username} description={profile.email} compact />
          <p className="mt-2 text-sm text-[var(--color-muted)]">
            Member since {new Date(profile.created_at).toLocaleDateString()}
          </p>
        </div>
      </div>

      <StatGroup
        items={[
          { label: "Level", value: profile.level, sub: `${profile.xp} XP` },
          { label: "Streak", value: `${profile.current_streak} days`, sub: `Best ${profile.best_streak} days` },
          { label: "Accuracy", value: profile.overall_accuracy != null ? `${profile.overall_accuracy}%` : "—" },
          { label: "Mastered", value: `${profile.letters_mastered} / 26` },
        ]}
      />

      <Divider />

      <section className="space-y-4">
        <SectionHeader title="Your ASL journey" description="A snapshot of how you've been learning." />
        <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
          {[
            ["Total signs practiced", profile.total_signs_practiced],
            ["Practice sessions", profile.total_practice_sessions],
            ["Active days", profile.total_days_active],
            ["Most practiced", profile.most_practiced_letter ?? "—"],
            ["Strongest letter", profile.strongest_letter ?? "—"],
            ["Weakest letter", profile.weakest_letter ?? "—"],
          ].map(([label, value]) => (
            <div key={String(label)}>
              <p className="text-xs text-[var(--color-muted)]">{label}</p>
              <p className="mt-1 text-lg font-semibold text-[var(--color-ink)]">{value}</p>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-3 pt-2">
          <SecondaryButton onClick={() => navigate("settings")}>Settings</SecondaryButton>
          {profile.weakest_letter && (
            <PrimaryButton onClick={() => startPractice(profile.weakest_letter!)}>
              Practice {profile.weakest_letter}
            </PrimaryButton>
          )}
        </div>
      </section>
    </PageLayout>
  );
}
