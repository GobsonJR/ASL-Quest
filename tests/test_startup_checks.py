"""Production safety checks and database URL normalisation (no network, no database)."""

from __future__ import annotations

import unittest

from backend.settings import normalize_database_url
from backend.startup_checks import KNOWN_DEFAULT_SECRET, review

GOOD_SECRET = "x" * 48
GOOD_KWARGS = dict(secret=GOOD_SECRET, database_url="postgresql+psycopg://u@h/db", cors_origins=["https://asl-quest-1.vercel.app"])


class NormalizeDatabaseUrlTests(unittest.TestCase):
    def test_render_style_urls_get_the_psycopg3_driver(self) -> None:
        self.assertEqual(normalize_database_url("postgres://u:p@h/db"), "postgresql+psycopg://u:p@h/db")
        self.assertEqual(normalize_database_url("postgresql://u:p@h/db"), "postgresql+psycopg://u:p@h/db")

    def test_explicit_driver_and_sqlite_are_untouched(self) -> None:
        for url in ("postgresql+psycopg://u:p@h/db", "sqlite:///x.db"):
            self.assertEqual(normalize_database_url(url), url)

    def test_surrounding_whitespace_is_stripped(self) -> None:
        self.assertEqual(normalize_database_url("  sqlite:///x.db \n"), "sqlite:///x.db")


class ReviewTests(unittest.TestCase):
    def test_nothing_is_checked_outside_production(self) -> None:
        found = review(production=False, secret=KNOWN_DEFAULT_SECRET, database_url="sqlite:///x", cors_origins=["http://localhost:5173"])
        self.assertEqual((found.errors, found.warnings), ([], []))

    def test_default_secret_is_an_error_in_production(self) -> None:
        found = review(production=True, **{**GOOD_KWARGS, "secret": KNOWN_DEFAULT_SECRET})
        self.assertEqual(len(found.errors), 1)
        self.assertNotIn(KNOWN_DEFAULT_SECRET, found.errors[0])

    def test_short_secret_warns_but_does_not_block(self) -> None:
        found = review(production=True, **{**GOOD_KWARGS, "secret": "short"})
        self.assertEqual(found.errors, [])
        self.assertEqual(len(found.warnings), 1)

    def test_good_configuration_is_clean(self) -> None:
        found = review(production=True, **GOOD_KWARGS)
        self.assertEqual((found.errors, found.warnings), ([], []))

    def test_sqlite_local_cors_and_debug_flags_warn(self) -> None:
        found = review(
            production=True,
            secret=GOOD_SECRET,
            database_url="sqlite:///x.db",
            cors_origins=["http://localhost:5173"],
            collection_enabled=True,
            diagnostics_enabled=True,
        )
        self.assertEqual(found.errors, [])
        self.assertEqual(len(found.warnings), 4)


if __name__ == "__main__":
    unittest.main()
