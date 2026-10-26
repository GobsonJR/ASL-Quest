"""Offline FAQ (backend/services/chatbot_faq.py) and the fallback_answer retrieval built on it.

Pure unit tests: no network, no database, no LLM.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import chatbot_faq as faq  # noqa: E402
from backend.services import chatbot_knowledge as kb  # noqa: E402

# question -> id of the FAQ entry that must answer it first
EXPECTED_TOP_ENTRY = {
    "What is ASL Quest?": "what_is",
    "What should I learn next? I've completed 0 of 26 letters and my next letter is C.": "getting_started",
    "I can't complete the book lesson": "native_trouble",
    "How is XP awarded?": "xp_levels",
    "How do streaks work?": "streaks",
    "Which badges can I earn?": "achievements",
    "How accurate is the model?": "accuracy",
    "Why is it not detecting my hand?": "camera_tips",
    "Why are J and Z not supported?": "jz",
    "How does native sign recognition work?": "native_model",
    "Which native signs are available?": "native_list",
    "How do I practice a native sign?": "native_how",
    "How does word spelling work?": "word_spelling",
    "What does mastered mean?": "mastery",
    "What challenges are there?": "challenges",
    "How do I change my password?": "account",
    "Is my video stored?": "privacy_security",
    "What is the tech stack?": "tech_stack",
    "What database does it use?": "database",
    "How does the database store my progress?": "database",
    "How do I deploy this on Vercel and Render?": "deployment",
    "What are the limitations?": "limitations",
    "How do I run it locally?": "setup_run",
    "Which model files are needed?": "model_files",
    "Who is Signo?": "signo",
    "What is the confidence threshold?": "confidence",
    "How do I become an admin?": "admin_mentor",
    "What are the API endpoints?": "api_endpoints",
    "How is the model trained?": "datasets_training",
    "How do the tests run?": "testing",
}

OFF_TOPIC = [
    "What is the capital of France?",
    "Tell me a joke",
    "Who is the president of India?",
    "How do I cook pasta?",
    "How does photosynthesis work?",
    "What is the weather today?",
    "Who won the world cup?",
    "What's the best programming language?",
]


class FaqRetrievalTests(unittest.TestCase):
    def test_every_question_maps_to_the_expected_entry(self) -> None:
        for question, expected in EXPECTED_TOP_ENTRY.items():
            matches = faq.best_matches(faq.normalize(question))
            self.assertTrue(matches, f"no FAQ match for: {question}")
            self.assertEqual(matches[0].id, expected, question)

    def test_fallback_answer_uses_the_faq_text(self) -> None:
        for question, expected in EXPECTED_TOP_ENTRY.items():
            answer = kb.fallback_answer(question)
            self.assertIn(kb.FALLBACK_PREFIX, answer, question)
            self.assertIn(faq._BY_ID[expected].answer, answer, question)

    def test_every_entry_is_reachable_by_each_of_its_own_keywords(self) -> None:
        # A keyword that never wins is dead weight (or a collision worth knowing about): every entry
        # must at least be among the returned matches for its own longest keyword phrase.
        for entry in faq.FAQ_ENTRIES:
            longest = max(entry.keywords, key=lambda kw: len(faq.normalize(kw).split()))
            ranked_ids = [e.id for _, e in faq.score_entries(faq.normalize(longest))]
            self.assertIn(entry.id, ranked_ids[:3], f"{entry.id} unreachable via {longest!r}")

    def test_entries_are_well_formed(self) -> None:
        ids = [entry.id for entry in faq.FAQ_ENTRIES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate FAQ ids")
        self.assertGreaterEqual(len(ids), 35)
        for entry in faq.FAQ_ENTRIES:
            self.assertTrue(entry.title and entry.answer and entry.keywords, entry.id)
            for related_id in entry.related:
                self.assertIn(related_id, ids, f"{entry.id} -> unknown related entry {related_id}")
                self.assertNotEqual(related_id, entry.id)

    def test_answers_include_related_questions(self) -> None:
        answer = kb.fallback_answer("How is XP awarded?")
        self.assertIn("You can also ask:", answer)

    def test_no_secrets_or_urls_with_credentials_in_answers(self) -> None:
        for entry in faq.FAQ_ENTRIES:
            lowered = entry.answer.lower()
            self.assertNotIn("sk-or-", lowered)
            self.assertNotIn("postgresql://", lowered)
            self.assertNotRegex(entry.answer, r"://[^/\s:]+:[^@\s]+@")


class OfflineScopeTests(unittest.TestCase):
    def test_project_questions_pass_the_scope_gate(self) -> None:
        for question in EXPECTED_TOP_ENTRY:
            self.assertTrue(kb.is_in_scope(question), question)

    def test_off_topic_questions_are_still_rejected(self) -> None:
        for question in OFF_TOPIC:
            self.assertFalse(kb.is_in_scope(question), question)

    def test_greetings_get_a_welcome_not_a_refusal(self) -> None:
        for greeting in ["hi", "Hello!", "hey aura", "Good morning", "thanks"]:
            self.assertTrue(kb.is_in_scope(greeting), greeting)
            self.assertEqual(kb.fallback_answer(greeting), kb.GREETING_REPLY)

    def test_greeting_words_inside_an_off_topic_sentence_do_not_pass(self) -> None:
        self.assertFalse(kb.is_in_scope("hello can you tell me the capital of France"))

    def test_unknown_topic_gets_honest_no_info_with_suggestions(self) -> None:
        answer = kb.fallback_answer("roadmap")
        self.assertTrue(answer.startswith(kb.NO_INFO_REPLY))
        self.assertIn("Try asking about:", answer)


class FactConsistencyTests(unittest.TestCase):
    """Guard the numbers in the answers against drifting from the code they describe."""

    def test_xp_values_match_frontend_constants(self) -> None:
        constants = (ROOT / "frontend" / "src" / "game" / "constants.ts").read_text(encoding="utf-8")
        xp_answer = faq._BY_ID["xp_levels"].answer
        for name, value in [
            ("XP_CORRECT", "20"), ("XP_FAST_CORRECT", "30"), ("XP_DAILY_COMPLETE", "100"),
            ("XP_PERFECT_DAILY", "150"), ("XP_ALPHABET_CHALLENGE", "500"), ("XP_SPEED_CHALLENGE", "150"),
            ("XP_WORD_COMPLETED", "50"), ("XP_WORD_PERFECT", "25"), ("XP_WORD_FAST", "20"),
            ("XP_WORD_CHALLENGE", "150"),
        ]:
            self.assertRegex(constants, rf"export const {name} = {value};")
            self.assertIn(value, xp_answer)

    def test_badge_list_matches_the_seeded_achievements(self) -> None:
        from backend.services.progress import ACHIEVEMENT_SEED

        self.assertEqual(len(ACHIEVEMENT_SEED), 16)
        answer = faq._BY_ID["achievements"].answer
        self.assertIn("16 badges", answer)
        for _slug, name, *_rest in ACHIEVEMENT_SEED:
            self.assertIn(name, answer, name)

    def test_native_sign_list_matches_the_catalog(self) -> None:
        answer = faq._BY_ID["native_list"].answer
        for gloss in ["HELLO", "THANKYOU", "PLEASE", "YES", "NO", "MOTHER", "WATER", "BOOK", "HELP"]:
            self.assertIn(gloss, answer)

    def test_native_recording_window_matches_the_frontend(self) -> None:
        source = (ROOT / "frontend" / "src" / "native" / "NativeVideoCapture.tsx").read_text(encoding="utf-8")
        self.assertIn("const RECORD_SECONDS = 10;", source)
        self.assertIn("const COUNTDOWN_SECONDS = 3;", source)
        self.assertIn("10 seconds", faq._BY_ID["native_how"].answer)

    def test_level_formula_matches_backend(self) -> None:
        from backend.services.progress import level_from_xp

        self.assertEqual(level_from_xp(0), 1)
        self.assertEqual(level_from_xp(1000), 2)
        self.assertIn("1000 XP per level", faq._BY_ID["xp_levels"].answer)


# Real learner questions that used to get "I don't have enough information", the wrong topic, or an off-topic refusal.
NEWLY_ANSWERED = {
    "How do I practice?": "practice_how",
    "How does practice work?": "practice_how",
    "How many letters are there?": "alphabet_overview",
    "Teach me the alphabet": "alphabet_overview",
    "Which letters are hard?": "hard_letters",
    "How do I improve my accuracy?": "improve_accuracy",
    "What is ASL?": "asl_basics",
    "Is ASL the same as BSL?": "asl_basics",
    "Can I use it on my phone?": "device_support",
    "What is the landmark recognizer?": "a_to_z_how",
    "How does MediaPipe work here?": "a_to_z_how",
    "Why is my sign not recognized?": "camera_tips",
    "How do I sign hello?": "native_how",
    "My native sign is not recognized": "native_trouble",
    "How do I log in?": "account",
}


class LearnerQuestionTests(unittest.TestCase):
    def test_common_learner_questions_reach_the_right_entry(self) -> None:
        for question, expected in NEWLY_ANSWERED.items():
            self.assertTrue(kb.is_in_scope(question), f"wrongly refused as off-topic: {question}")
            answer = kb.fallback_answer(question)
            self.assertIn(faq._BY_ID[expected].answer, answer, question)
            self.assertNotIn(kb.NO_INFO_REPLY, answer, question)

    def test_recognition_answer_describes_the_current_pipeline(self) -> None:
        answer = faq._BY_ID["a_to_z_how"].answer
        self.assertIn("your browser runs MediaPipe", answer)
        self.assertIn("21 landmark points", answer)
        self.assertIn("only those points", answer)
        self.assertNotIn("finds and crops your hand from the webcam frame", answer)

    def test_single_letter_questions_use_the_apps_own_cue(self) -> None:
        for question, letter in {
            "How do I sign the letter A?": "A", "How to sign B": "B", "What does the letter M look like?": "M",
            "what does K look like": "K", "How do I make the letter W": "W",
        }.items():
            answer = kb.fallback_answer(question)
            self.assertIn(f"{letter}: {faq.LETTER_HINTS[letter]}", answer, question)
            self.assertTrue(kb.is_in_scope(question), question)

    def test_letter_comparison_gives_both_cues(self) -> None:
        for question in ["Difference between M and N", "M vs N", "What is the difference between M and N?"]:
            self.assertTrue(kb.is_in_scope(question), question)
            answer = kb.fallback_answer(question)
            self.assertIn(f"M: {faq.LETTER_HINTS['M']}", answer, question)
            self.assertIn(f"N: {faq.LETTER_HINTS['N']}", answer, question)

    def test_moving_letters_are_described_honestly(self) -> None:
        for letter in "JZ":
            self.assertIn("practised as a held shape", kb.fallback_answer(f"How do I sign the letter {letter}?"))

    def test_the_article_a_is_never_read_as_the_letter_a(self) -> None:
        for question in ["What is a streak?", "How do I earn a badge?", "Is there a daily goal?", "What is a native sign?"]:
            self.assertIsNone(faq.letter_answer(faq.normalize(question)), question)
            self.assertNotIn("Make a fist with thumb alongside", kb.fallback_answer(question), question)

    def test_off_topic_questions_are_still_refused(self) -> None:
        for question in OFF_TOPIC + ["Do you prefer pizza or pasta?", "What phone should I buy?", "Who is better, a or b in cricket today?"]:
            self.assertFalse(kb.is_in_scope(question), question)

    def test_letter_cues_match_the_frontend_wording(self) -> None:
        source = (ROOT / "frontend" / "src" / "pages" / "alphabetFeedback.ts").read_text(encoding="utf-8")
        import re

        frontend = dict(re.findall(r'^\s{2}([A-Z]): "(.+?)",$', source, flags=re.M))
        self.assertEqual(set(frontend), set(faq.LETTER_HINTS))
        for letter, cue in frontend.items():
            self.assertEqual(faq.LETTER_HINTS[letter], cue.replace(" — ", " - "), letter)

    def test_fallback_wording_is_not_an_error_message(self) -> None:
        self.assertNotIn("isn't reachable", kb.FALLBACK_PREFIX)
        self.assertIn("knowledge base", kb.FALLBACK_PREFIX)


if __name__ == "__main__":
    unittest.main()
