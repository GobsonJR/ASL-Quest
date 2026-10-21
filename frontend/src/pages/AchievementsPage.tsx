import { Divider, PageHeader, PageLayout, ProgressBar, SectionHeader, StatusChip } from "../components/AppShell";
import { BADGES } from "../game/constants";
import { useGame } from "../game/GameContext";
import { getBadgeProgress } from "../game/gamification";

export function AchievementsPage() {
  const { state } = useGame();
  const unlockedCount = state.unlockedBadges.length;

  return (
    <PageLayout>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <PageHeader
          eyebrow="🏅 Your collection"
          title="Badges & milestones"
          description="Unlock badges by practicing consistently and completing challenges."
        />
        <StatusChip tone="accent">
          {unlockedCount} / {BADGES.length} unlocked
        </StatusChip>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {BADGES.map((badge) => {
          const unlocked = state.unlockedBadges.includes(badge.id);
          const progress = getBadgeProgress(state, badge.id);

          return (
            <div
              key={badge.id}
              className={`animate-pop flex flex-col items-center gap-3 rounded-[var(--radius-panel)] border-2 p-5 text-center ${
                unlocked
                  ? "border-[var(--color-xp)]/40 bg-[var(--color-xp-soft)]"
                  : "border-[var(--color-border-soft)] bg-[var(--color-surface)]"
              }`}
            >
              <div
                className={`grid h-16 w-16 place-items-center rounded-full text-3xl ${
                  unlocked ? "animate-badge-unlock bg-[var(--color-xp)] shadow-lg" : "bg-[var(--color-locked-soft)] grayscale"
                }`}
                aria-hidden="true"
              >
                {badge.icon}
              </div>
              <div>
                <h3 className="text-base font-bold text-[var(--color-ink)]">{badge.title}</h3>
                <p className="mt-1 text-sm text-[var(--color-ink-soft)]">{badge.description}</p>
              </div>
              <StatusChip tone={unlocked ? "success" : "neutral"}>{unlocked ? "✓ Earned" : "🔒 Locked"}</StatusChip>

              {!unlocked && progress && progress.target > 1 && (
                <div className="mt-1 w-full">
                  <div className="mb-1 flex justify-between text-xs text-[var(--color-muted)]">
                    <span>
                      {progress.current} / {progress.target} {progress.label}
                    </span>
                  </div>
                  <ProgressBar percent={(progress.current / progress.target) * 100} showPercent={false} size="sm" />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Divider />

      <section>
        <SectionHeader title="How to earn more" />
        <ul className="mt-4 space-y-2 text-sm leading-relaxed text-[var(--color-ink-soft)]">
          <li>🔥 Practice daily to build your streak.</li>
          <li>🔤 Sign different letters to reach alphabet milestones.</li>
          <li>⚡ Complete the speed challenge for Speed Signer.</li>
          <li>✨ Finish a daily challenge with zero mistakes for Perfect Practice.</li>
          <li>🧩 Spell complete words with sequential alphabet signs to unlock Word Starter and Word Learner.</li>
        </ul>
      </section>
    </PageLayout>
  );
}
