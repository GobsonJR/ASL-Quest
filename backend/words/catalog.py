from __future__ import annotations

from typing import Any

WORD_CATEGORIES = ("Beginner", "Greetings", "Family", "Everyday", "School")

# Static vocabulary for sequential A–Z spelling practice. Not native ASL word signs.
_WORDS: list[dict[str, Any]] = [
    {
        "id": "beginner-cat",
        "word": "CAT",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "A small domesticated animal.",
        "tip": "Spell each letter clearly, then pause before the next sign.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-dog",
        "word": "DOG",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "A loyal four-legged companion.",
        "tip": "D, O, and G use very different hand shapes — reset your hand between letters.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-sun",
        "word": "SUN",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "The star that lights the day.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-hat",
        "word": "HAT",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "A covering worn on the head.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-bed",
        "word": "BED",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "A place to sleep and rest.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-bad",
        "word": "BAD",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "Not good; the opposite of good.",
        "estimated_xp": 110,
    },
    {
        "id": "beginner-sad",
        "word": "SAD",
        "category": "Beginner",
        "difficulty": "Easy",
        "description": "Feeling unhappy or down.",
        "estimated_xp": 110,
    },
    {
        "id": "greetings-hi",
        "word": "HI",
        "category": "Greetings",
        "difficulty": "Easy",
        "description": "A short, friendly greeting.",
        "estimated_xp": 90,
    },
    {
        "id": "greetings-hello",
        "word": "HELLO",
        "category": "Greetings",
        "difficulty": "Medium",
        "description": "A warm way to greet someone.",
        "tip": "Two L letters in a row — fully finish the first L before repeating it.",
        "estimated_xp": 150,
    },
    {
        "id": "greetings-bye",
        "word": "BYE",
        "category": "Greetings",
        "difficulty": "Easy",
        "description": "A short farewell.",
        "estimated_xp": 110,
    },
    {
        "id": "greetings-thanks",
        "word": "THANKS",
        "category": "Greetings",
        "difficulty": "Medium",
        "description": "A polite way to show gratitude.",
        "estimated_xp": 170,
    },
    {
        "id": "family-mom",
        "word": "MOM",
        "category": "Family",
        "difficulty": "Easy",
        "description": "A mother or maternal caregiver.",
        "estimated_xp": 110,
    },
    {
        "id": "family-dad",
        "word": "DAD",
        "category": "Family",
        "difficulty": "Easy",
        "description": "A father or paternal caregiver.",
        "estimated_xp": 110,
    },
    {
        "id": "family-sister",
        "word": "SISTER",
        "category": "Family",
        "difficulty": "Medium",
        "description": "A female sibling.",
        "estimated_xp": 170,
    },
    {
        "id": "family-brother",
        "word": "BROTHER",
        "category": "Family",
        "difficulty": "Hard",
        "description": "A male sibling.",
        "estimated_xp": 190,
    },
    {
        "id": "everyday-home",
        "word": "HOME",
        "category": "Everyday",
        "difficulty": "Easy",
        "description": "The place where you live.",
        "estimated_xp": 130,
    },
    {
        "id": "everyday-food",
        "word": "FOOD",
        "category": "Everyday",
        "difficulty": "Easy",
        "description": "Something you eat.",
        "estimated_xp": 130,
    },
    {
        "id": "everyday-water",
        "word": "WATER",
        "category": "Everyday",
        "difficulty": "Medium",
        "description": "The drink we need every day.",
        "estimated_xp": 150,
    },
    {
        "id": "everyday-friend",
        "word": "FRIEND",
        "category": "Everyday",
        "difficulty": "Medium",
        "description": "Someone you like and trust.",
        "estimated_xp": 170,
    },
    {
        "id": "everyday-book",
        "word": "BOOK",
        "category": "Everyday",
        "difficulty": "Easy",
        "description": "A set of written pages you can read.",
        "estimated_xp": 130,
    },
    {
        "id": "school-school",
        "word": "SCHOOL",
        "category": "School",
        "difficulty": "Medium",
        "description": "A place where students learn.",
        "estimated_xp": 170,
    },
    {
        "id": "school-class",
        "word": "CLASS",
        "category": "School",
        "difficulty": "Medium",
        "description": "A group of learners or a course.",
        "tip": "Two S letters at the end — hold each S, then sign again.",
        "estimated_xp": 150,
    },
    {
        "id": "school-study",
        "word": "STUDY",
        "category": "School",
        "difficulty": "Medium",
        "description": "To learn by reading or practicing.",
        "estimated_xp": 150,
    },
    {
        "id": "school-teacher",
        "word": "TEACHER",
        "category": "School",
        "difficulty": "Hard",
        "description": "A person who helps others learn.",
        "estimated_xp": 190,
    },
]


def _with_letters(item: dict[str, Any]) -> dict[str, Any]:
    word = str(item["word"]).upper()
    return {
        **item,
        "word": word,
        "letters": list(word),
        "letter_count": len(word),
        "tip": item.get("tip"),
    }


WORD_CATALOG: list[dict[str, Any]] = [_with_letters(item) for item in _WORDS]
WORD_BY_ID: dict[str, dict[str, Any]] = {item["id"]: item for item in WORD_CATALOG}


def list_words(category: str | None = None) -> list[dict[str, Any]]:
    if not category or category.lower() == "all":
        return list(WORD_CATALOG)
    wanted = category.strip().title()
    return [item for item in WORD_CATALOG if item["category"] == wanted]


def get_word(word_id: str) -> dict[str, Any] | None:
    return WORD_BY_ID.get(word_id)


def all_word_ids() -> list[str]:
    return [item["id"] for item in WORD_CATALOG]


def validate_catalog() -> None:
    seen: set[str] = set()
    for item in WORD_CATALOG:
        if item["id"] in seen:
            raise ValueError(f"Duplicate word id: {item['id']}")
        seen.add(item["id"])
        if item["category"] not in WORD_CATEGORIES:
            raise ValueError(f"Unknown category for {item['id']}")
        if not item["word"].isalpha() or not item["word"].isupper():
            raise ValueError(f"Word must be A–Z letters only: {item['word']}")
