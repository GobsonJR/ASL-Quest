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
          eyebrow="Achievements"
          title="Badges & milestones"
          description="Unlock badges by practicing consistently and completing challenges."
        />
        <StatusChip tone="accent">
          {unlockedCount} / {BADGES.length} unlocked
        </StatusChip>
      </div>

      <ul className="divide-y divide-[var(--color-line-soft)] rounded-[var(--radius-panel)] border border-[var(--color-line)]">
        {BADGES.map((badge) => {
          const unlocked = state.unlockedBadges.includes(badge.id);
          const progress = getBadgeProgress(state, badge.id);

          return (
            <li
              key={badge.id}
              className={`flex gap-4 px-4 py-4 md:px-5 ${
                unlocked ? "border-l-2 border-l-[var(--color-accent)] bg-[var(--color-accent)]/[0.04]" : "opacity-80"
              }`}
            >
              <div
                className={`grid h-12 w-12 shrink-0 place-items-center rounded-[var(--radius-control)] text-2xl ${
                  unlocked ? "bg-[var(--color-accent)]/12" : "bg-[var(--color-panel-soft)] grayscale"
                }`}
                aria-hidden="true"
              >
                {badge.icon}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-base font-semibold text-[#eef4f0]">{badge.title}</h3>
                  <StatusChip tone={unlocked ? "success" : "neutral"}>{unlocked ? "Earned" : "Locked"}</StatusChip>
                </div>
                <p className="mt-1 text-sm text-[var(--color-mist)]">{badge.description}</p>

                {!unlocked && progress && progress.target > 1 && (
                  <div className="mt-3 max-w-md">
                    <div className="mb-1 flex justify-between text-xs text-[var(--color-muted)]">
                      <span>
                        {progress.current} / {progress.target} {progress.label}
                      </span>
                    </div>
                    <ProgressBar percent={(progress.current / progress.target) * 100} showPercent={false} size="sm" />
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      <Divider />

      <section>
        <SectionHeader title="How to earn more" />
        <ul className="mt-4 space-y-2 text-sm leading-relaxed text-[var(--color-mist)]">
          <li>Practice daily to build your streak.</li>
          <li>Sign different letters to reach alphabet milestones.</li>
          <li>Complete the speed challenge for Speed Signer.</li>
          <li>Finish a daily challenge with zero mistakes for Perfect Practice.</li>
          <li>Spell complete words with sequential alphabet signs to unlock Word Starter and Word Learner.</li>
        </ul>
      </section>
    </PageLayout>
  );
}
