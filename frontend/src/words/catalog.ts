export type WordDifficulty = "Easy" | "Medium" | "Hard";
export type WordCategory = "Beginner" | "Greetings" | "Family" | "Everyday" | "School";

export type WordDefinition = {
  id: string;
  word: string;
  category: WordCategory;
  difficulty: WordDifficulty;
  description: string;
  letters: string[];
  letterCount: number;
  estimatedXp: number;
  tip?: string;
};

export const WORD_CATEGORIES: Array<"All" | WordCategory> = [
  "All",
  "Beginner",
  "Greetings",
  "Family",
  "Everyday",
  "School",
];

const RAW: Array<Omit<WordDefinition, "letters" | "letterCount">> = [
  { id: "beginner-cat", word: "CAT", category: "Beginner", difficulty: "Easy", description: "A small domesticated animal.", estimatedXp: 110, tip: "Spell each letter clearly, then pause before the next sign." },
  { id: "beginner-dog", word: "DOG", category: "Beginner", difficulty: "Easy", description: "A loyal four-legged companion.", estimatedXp: 110, tip: "D, O, and G use very different hand shapes — reset your hand between letters." },
  { id: "beginner-sun", word: "SUN", category: "Beginner", difficulty: "Easy", description: "The star that lights the day.", estimatedXp: 110 },
  { id: "beginner-hat", word: "HAT", category: "Beginner", difficulty: "Easy", description: "A covering worn on the head.", estimatedXp: 110 },
  { id: "beginner-bed", word: "BED", category: "Beginner", difficulty: "Easy", description: "A place to sleep and rest.", estimatedXp: 110 },
  { id: "beginner-bad", word: "BAD", category: "Beginner", difficulty: "Easy", description: "Not good; the opposite of good.", estimatedXp: 110 },
  { id: "beginner-sad", word: "SAD", category: "Beginner", difficulty: "Easy", description: "Feeling unhappy or down.", estimatedXp: 110 },
  { id: "greetings-hi", word: "HI", category: "Greetings", difficulty: "Easy", description: "A short, friendly greeting.", estimatedXp: 90 },
  { id: "greetings-hello", word: "HELLO", category: "Greetings", difficulty: "Medium", description: "A warm way to greet someone.", estimatedXp: 150, tip: "Two L letters in a row — fully finish the first L before repeating it." },
  { id: "greetings-bye", word: "BYE", category: "Greetings", difficulty: "Easy", description: "A short farewell.", estimatedXp: 110 },
  { id: "greetings-thanks", word: "THANKS", category: "Greetings", difficulty: "Medium", description: "A polite way to show gratitude.", estimatedXp: 170 },
  { id: "family-mom", word: "MOM", category: "Family", difficulty: "Easy", description: "A mother or maternal caregiver.", estimatedXp: 110 },
  { id: "family-dad", word: "DAD", category: "Family", difficulty: "Easy", description: "A father or paternal caregiver.", estimatedXp: 110 },
  { id: "family-sister", word: "SISTER", category: "Family", difficulty: "Medium", description: "A female sibling.", estimatedXp: 170 },
  { id: "family-brother", word: "BROTHER", category: "Family", difficulty: "Hard", description: "A male sibling.", estimatedXp: 190 },
  { id: "everyday-home", word: "HOME", category: "Everyday", difficulty: "Easy", description: "The place where you live.", estimatedXp: 130 },
  { id: "everyday-food", word: "FOOD", category: "Everyday", difficulty: "Easy", description: "Something you eat.", estimatedXp: 130 },
  { id: "everyday-water", word: "WATER", category: "Everyday", difficulty: "Medium", description: "The drink we need every day.", estimatedXp: 150 },
  { id: "everyday-friend", word: "FRIEND", category: "Everyday", difficulty: "Medium", description: "Someone you like and trust.", estimatedXp: 170 },
  { id: "everyday-book", word: "BOOK", category: "Everyday", difficulty: "Easy", description: "A set of written pages you can read.", estimatedXp: 130 },
  { id: "school-school", word: "SCHOOL", category: "School", difficulty: "Medium", description: "A place where students learn.", estimatedXp: 170 },
  { id: "school-class", word: "CLASS", category: "School", difficulty: "Medium", description: "A group of learners or a course.", estimatedXp: 150, tip: "Two S letters at the end — hold each S, then sign again." },
  { id: "school-study", word: "STUDY", category: "School", difficulty: "Medium", description: "To learn by reading or practicing.", estimatedXp: 150 },
  { id: "school-teacher", word: "TEACHER", category: "School", difficulty: "Hard", description: "A person who helps others learn.", estimatedXp: 190 },
];

export const WORD_CATALOG: WordDefinition[] = RAW.map((item) => ({
  ...item,
  letters: item.word.split(""),
  letterCount: item.word.length,
}));

export const WORD_BY_ID: Record<string, WordDefinition> = Object.fromEntries(WORD_CATALOG.map((item) => [item.id, item]));

export function getWordById(id: string): WordDefinition | undefined {
  return WORD_BY_ID[id];
}

export function wordsInCategory(category: "All" | WordCategory): WordDefinition[] {
  if (category === "All") return WORD_CATALOG;
  return WORD_CATALOG.filter((item) => item.category === category);
}

export function nextWordId(currentId: string): string {
  const index = WORD_CATALOG.findIndex((item) => item.id === currentId);
  const next = WORD_CATALOG[(index + 1 + WORD_CATALOG.length) % WORD_CATALOG.length];
  return next?.id ?? WORD_CATALOG[0].id;
}
