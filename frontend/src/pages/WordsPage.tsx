import { useEffect, useState } from "react";
import {
  PageHeader,
  PageLayout,
  Panel,
  PrimaryButton,
  SectionHeader,
  StatusChip,
} from "../components/AppShell";
import { useGame } from "../game/GameContext";
import { WORD_CATEGORIES, WORD_CATALOG, getWordById, type WordCategory } from "../words/catalog";
import { fetchWords, type WordItem } from "../words/api";

export function WordsPage() {
  const { startWordPractice } = useGame();
  const [category, setCategory] = useState<"All" | WordCategory>("All");
  const [selectedWordId, setSelectedWordId] = useState(WORD_CATALOG[0].id);
  const [remoteWords, setRemoteWords] = useState<WordItem[] | null>(null);

  useEffect(() => {
    fetchWords()
      .then((payload) => setRemoteWords(payload.items))
      .catch(() => setRemoteWords(null));
  }, []);

  const catalog =
    remoteWords ??
    WORD_CATALOG.map((item) => ({
      id: item.id,
      word: item.word,
      category: item.category,
      difficulty: item.difficulty,
      description: item.description,
      letters: item.letters,
      letter_count: item.letterCount,
      estimated_xp: item.estimatedXp,
      tip: item.tip ?? null,
      progress: null,
    }));
  const filtered = catalog.filter((item) => category === "All" || item.category === category);
  const selectedWord = catalog.find((item) => item.id === selectedWordId) ?? catalog[0];
  const localWord = getWordById(selectedWord?.id ?? "") ?? WORD_CATALOG[0];

  return (
    <PageLayout>
      <PageHeader
        eyebrow="Word practice"
        title="Spell words with A–Z signs"
        description="This is letter-by-letter spelling using the existing alphabet classifier. It is not native ASL word recognition."
      />

      <div className="flex flex-wrap gap-2" role="group" aria-label="Word categories">
        {WORD_CATEGORIES.map((item) => (
          <button
            key={item}
            type="button"
            className={`rounded-[var(--radius-control)] px-3 py-1.5 text-sm ${
              category === item
                ? "bg-[var(--color-accent)] text-[var(--color-ink)]"
                : "border border-[var(--color-line)] text-[var(--color-mist)]"
            }`}
            onClick={() => setCategory(item)}
          >
            {item}
          </button>
        ))}
      </div>

      <div className="grid gap-8 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        {selectedWord && (
          <Panel className="space-y-5 xl:sticky xl:top-28 xl:self-start">
            <div>
              <StatusChip tone="neutral">{selectedWord.category}</StatusChip>
              <h3 className="mt-3 font-display text-4xl">{selectedWord.word}</h3>
              <p className="mt-2 text-sm text-[var(--color-mist)]">{selectedWord.description}</p>
            </div>
            <p className="font-display text-2xl tracking-[0.2em] text-[var(--color-accent)]">
              {selectedWord.letters.join("   ")}
            </p>
            <div>
              <p className="text-xs uppercase tracking-[0.12em] text-[var(--color-muted)]">How it works</p>
              <ol className="mt-2 space-y-1 pl-4 text-sm text-[var(--color-mist)] [list-style:decimal]">
                {selectedWord.letters.map((letter, idx) => (
                  <li key={`${letter}-${idx}`}>Make the {letter} sign</li>
                ))}
                <li>Complete the word</li>
              </ol>
            </div>
            {localWord.tip && <p className="text-sm text-[var(--color-muted)]">{localWord.tip}</p>}
            {selectedWord.progress && selectedWord.progress.attempts > 0 && (
              <p className="text-sm text-[var(--color-mist)]">
                {selectedWord.progress.completions} completions · {selectedWord.progress.mastery}% mastery
                {selectedWord.progress.accuracy != null ? ` · ${selectedWord.progress.accuracy}% accuracy` : ""}
              </p>
            )}
            <PrimaryButton onClick={() => startWordPractice(selectedWord.id)}>Start practice</PrimaryButton>
          </Panel>
        )}

        <section>
          <SectionHeader title="Vocabulary" description="Select a word, then spell it one alphabet sign at a time." />
          <ul className="mt-4 divide-y divide-[var(--color-line-soft)] rounded-[var(--radius-panel)] border border-[var(--color-line)]">
            {filtered.map((item) => {
              const active = item.id === selectedWord?.id;
              const mastered = item.progress?.mastered;
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className={`flex w-full items-center justify-between gap-3 px-4 py-3 text-left ${
                      active ? "bg-[var(--color-accent)]/8" : "hover:bg-[var(--color-panel-soft)]"
                    }`}
                    onClick={() => setSelectedWordId(item.id)}
                    aria-pressed={active}
                  >
                    <span>
                      <span className="block font-display text-xl">{item.word}</span>
                      <span className="text-xs text-[var(--color-muted)]">
                        {item.category} · {item.difficulty} · {item.letter_count} letters
                      </span>
                    </span>
                    <span className="flex items-center gap-3">
                      {item.progress && item.progress.attempts > 0 && (
                        <span className="text-xs text-[var(--color-mist)]">{Math.round(item.progress.mastery)}%</span>
                      )}
                      {mastered && <StatusChip tone="success">Mastered</StatusChip>}
                      <span className="text-sm font-medium text-[var(--color-accent)]">Practice</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      </div>
    </PageLayout>
  );
}
