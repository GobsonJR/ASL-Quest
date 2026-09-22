"""Curated, reproducible ASL Citizen vocabularies.

Glosses are exact labels from the ASL Citizen source CSVs. None are invented.
"""

from __future__ import annotations

# Stage 1: everyday isolated signs for a student demonstration.
# Selection rules:
# - must exist in official train/val/test splits
# - prefer ~30+ videos and multiple signers
# - one variant per concept (EAT1 not EAT2, DOG1 not DOG2, WANT1 not WANT2)
# - cover greetings, people, basic responses, common actions, simple nouns
ASL_CITIZEN_20_GLOSSES: tuple[str, ...] = (
    "HELLO",
    "THANKYOU",
    "PLEASE",
    "YES",
    "NO",
    "NAME",
    "MOTHER",
    "FATHER",
    "BOY",
    "GIRL",
    "HELP",
    "WANT1",
    "WATER",
    "EAT1",
    "DRINK1",
    "HOUSE",
    "SCHOOL",
    "BOOK",
    "HAPPY",
    "DOG1",
)

ASL_CITIZEN_20_REASONS: dict[str, str] = {
    "HELLO": "Core greeting for a first-lesson demo.",
    "THANKYOU": "High-frequency courtesy sign.",
    "PLEASE": "High-frequency courtesy sign.",
    "YES": "Basic affirmative response.",
    "NO": "Basic negative response; pair with YES.",
    "NAME": "Useful for introductions (WHAT YOUR NAME).",
    "MOTHER": "Core family/people vocabulary.",
    "FATHER": "Core family/people vocabulary; not a duplicate of MOTHER.",
    "BOY": "People vocabulary with stronger val coverage than some family signs.",
    "GIRL": "People vocabulary; pair with BOY.",
    "HELP": "Common actionable request.",
    "WANT1": "Common verb; WANT1 kept, WANT2 excluded as a variant.",
    "WATER": "Everyday noun.",
    "EAT1": "Everyday action; EAT1 kept, EAT2 excluded as a variant.",
    "DRINK1": "Everyday action; DRINK1 kept, DRINK2 excluded as a variant.",
    "HOUSE": "Everyday place noun.",
    "SCHOOL": "Everyday place noun for a student demo.",
    "BOOK": "Everyday object noun.",
    "HAPPY": "Basic emotion, visually distinct from YES/NO.",
    "DOG1": "High-count everyday noun; DOG1 kept, DOG2/3/4 excluded as variants.",
}

ASL_CITIZEN_20_SELECTION_RULE = (
    "Curated 20 isolated glosses from ASL Citizen source labels for a student demo. "
    "Require presence in train/val/test. Prefer everyday greetings, people, responses, "
    "actions, and nouns. One lexical variant per concept. Not top-N by frequency."
)


def stage1_glosses() -> list[str]:
    return list(ASL_CITIZEN_20_GLOSSES)
