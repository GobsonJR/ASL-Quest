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
        eyebrow="🧩 Word Spelling"
        title="Choose a word to practice"
        description="Spell words letter-by-letter using the same A–Z handshapes you already know. This is not native ASL word recognition."
      />

      <div className="flex flex-wrap gap-2" role="group" aria-label="Word categories">
        {WORD_CATEGORIES.map((item) => (
          <button
            key={item}
            type="button"
            className={`rounded-[var(--radius-pill)] px-4 py-1.5 text-sm font-semibold transition ${
              category === item
                ? "bg-[var(--color-primary)] text-white"
                : "border-2 border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-primary-soft)]"
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
              <StatusChip tone="accent">{selectedWord.category}</StatusChip>
              <h3 className="mt-3 font-display text-4xl font-bold text-[var(--color-ink)]">{selectedWord.word}</h3>
              <p className="mt-2 text-sm text-[var(--color-ink-soft)]">{selectedWord.description}</p>
            </div>
            <p className="font-display text-2xl font-semibold tracking-[0.2em] text-[var(--color-native)]">
              {selectedWord.letters.join("   ")}
            </p>
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--color-muted)]">How it works</p>
              <ol className="mt-2 space-y-1 pl-4 text-sm text-[var(--color-ink-soft)] [list-style:decimal]">
                {selectedWord.letters.map((letter, idx) => (
                  <li key={`${letter}-${idx}`}>Make the {letter} sign</li>
                ))}
                <li>Complete the word</li>
              </ol>
            </div>
            {localWord.tip && <p className="text-sm text-[var(--color-muted)]">💡 {localWord.tip}</p>}
            {selectedWord.progress && selectedWord.progress.attempts > 0 && (
              <p className="text-sm text-[var(--color-ink-soft)]">
                {selectedWord.progress.completions} completions · {selectedWord.progress.mastery}% mastery
                {selectedWord.progress.accuracy != null ? ` · ${selectedWord.progress.accuracy}% accuracy` : ""}
              </p>
            )}
            <PrimaryButton className="w-full" onClick={() => startWordPractice(selectedWord.id)}>
              Start →
            </PrimaryButton>
          </Panel>
        )}

        <section>
          <SectionHeader title="Vocabulary" description="Select a word, then spell it one alphabet sign at a time." />
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {filtered.map((item) => {
              const active = item.id === selectedWord?.id;
              const mastered = item.progress?.mastered;
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`lift-hover flex flex-col items-start gap-2 rounded-[var(--radius-panel)] border-2 p-4 text-left transition ${
                    active
                      ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]"
                      : "border-[var(--color-border-soft)] bg-[var(--color-surface)]"
                  }`}
                  onClick={() => setSelectedWordId(item.id)}
                  aria-pressed={active}
                >
                  <div className="flex w-full items-center justify-between gap-2">
                    <StatusChip tone="neutral">{item.difficulty}</StatusChip>
                    {mastered && <StatusChip tone="success">✓ Mastered</StatusChip>}
                  </div>
                  <span className="font-display text-2xl font-bold text-[var(--color-ink)]">{item.word}</span>
                  <span className="text-sm text-[var(--color-ink-soft)]">
                    {item.letters.join(" → ")} · {item.letter_count} letters
                  </span>
                  {item.progress && item.progress.attempts > 0 && (
                    <span className="text-xs font-semibold text-[var(--color-success)]">
                      {Math.round(item.progress.mastery)}% mastery
                    </span>
                  )}
                  <span className="mt-1 text-sm font-bold text-[var(--color-primary)]">
                    {active ? "Selected" : "Practice →"}
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      </div>
    </PageLayout>
  );
}
