"""Phase 2 database-foundation tests: Alembic migrations, new ORM models, and
the words catalog -> `words` table mapping.

These are independent of the existing app test suite (test_auth_db.py,
test_phase4.py, test_phase5.py, ...), which continues to exercise the legacy
tables through the FastAPI app exactly as before. Nothing here modifies
data/asl_quest.db; every test uses its own temporary SQLite file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import models  # noqa: E402,F401
from backend.database import Base, _LEGACY_TABLE_NAMES  # noqa: E402
from backend.models import (  # noqa: E402
    ChatbotConversation,
    ChatbotFeedback,
    ChatbotMessage,
    ModelVersion,
    NativeSign,
    NativeSignProgress,
    NativeSignReference,
    NativeSignSession,
    SignPrediction,
    User,
    UserPreference,
    Word,
    WordProgress,
)
from backend.words.catalog import WORD_CATALOG

MIGRATIONS_DIR = ROOT / "migrations" / "versions"
NEW_TABLE_NAMES = sorted(set(Base.metadata.tables) - set(_LEGACY_TABLE_NAMES))


def _temp_sqlite_url() -> tuple[str, Path]:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    path = Path(handle.name)
    return f"sqlite:///{path.as_posix()}", path


def _alembic_config():
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    return cfg


def _run_alembic(command: str, target_url: str, revision: str = "head") -> None:
    """Invoke `alembic <command> <revision>` as a real subprocess against target_url.

    Using a subprocess (rather than calling alembic.command in-process) sidesteps any
    risk of this test process's already-imported modules leaking state into env.py, and
    exercises the exact same code path a developer running the CLI would use.
    """
    env = dict(os.environ)
    env["DATABASE_URL"] = target_url
    env.pop("ASL_QUEST_DATABASE_URL", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", command, revision],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"alembic {command} {revision} failed:\n{result.stdout}\n{result.stderr}"


class AlembicStructureTests(unittest.TestCase):
    """Migration files exist, form a single linear chain, and the app can load them."""

    def test_migration_files_present(self) -> None:
        names = {p.name for p in MIGRATIONS_DIR.glob("*.py")}
        self.assertIn("0001_baseline_existing_schema.py", names)
        self.assertIn("0002_add_native_sign_chatbot_catalog_tables.py", names)
        self.assertIn("0003_seed_words_catalog.py", names)

    def test_single_linear_head(self) -> None:
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(_alembic_config())
        heads = script.get_heads()
        self.assertEqual(len(heads), 1, f"expected exactly one migration head, got {heads}")
        self.assertEqual(heads[0], "0003_seed_words")

    def test_revision_chain_order(self) -> None:
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(_alembic_config())
        revisions = [rev.revision for rev in script.walk_revisions()]
        self.assertEqual(revisions, ["0003_seed_words", "0002_new_tables", "0001_baseline"])

    def test_baseline_has_no_down_revision(self) -> None:
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(_alembic_config())
        baseline = script.get_revision("0001_baseline")
        self.assertIsNone(baseline.down_revision)


class FreshDatabaseMigrationTests(unittest.TestCase):
    """A brand-new empty SQLite database can be fully migrated up and back down."""

    def setUp(self) -> None:
        self.url, self.path = _temp_sqlite_url()

    def tearDown(self) -> None:
        # On Windows, a file can't be deleted while any engine still holds it open;
        # tests below dispose their engines, but be defensive rather than fail on cleanup.
        try:
            self.path.unlink(missing_ok=True)
        except PermissionError:
            pass

    def test_upgrade_head_creates_all_tables_and_seeds_words(self) -> None:
        _run_alembic("upgrade", self.url)
        engine = sa.create_engine(self.url)
        try:
            inspector = sa.inspect(engine)
            table_names = set(inspector.get_table_names())
            for name in Base.metadata.tables:
                self.assertIn(name, table_names, f"{name} missing after upgrade head")

            with engine.connect() as conn:
                word_count = conn.execute(sa.text("SELECT COUNT(*) FROM words")).scalar_one()
            self.assertEqual(word_count, len(WORD_CATALOG))
        finally:
            engine.dispose()

    def test_upgrade_is_idempotent(self) -> None:
        _run_alembic("upgrade", self.url)
        _run_alembic("upgrade", self.url)  # must not error or duplicate rows
        engine = sa.create_engine(self.url)
        try:
            with engine.connect() as conn:
                word_count = conn.execute(sa.text("SELECT COUNT(*) FROM words")).scalar_one()
            self.assertEqual(word_count, len(WORD_CATALOG))
        finally:
            engine.dispose()

    def test_downgrade_to_baseline_removes_only_new_tables(self) -> None:
        _run_alembic("upgrade", self.url)
        _run_alembic("downgrade", self.url, "0001_baseline")
        engine = sa.create_engine(self.url)
        try:
            inspector = sa.inspect(engine)
            table_names = set(inspector.get_table_names())
            for name in _LEGACY_TABLE_NAMES:
                self.assertIn(name, table_names)
            for name in NEW_TABLE_NAMES:
                self.assertNotIn(name, table_names)
        finally:
            engine.dispose()


class ExistingDataPreservationTests(unittest.TestCase):
    """The core safety guarantee: migrating a database with real rows in it must not
    touch those rows. Simulates the exact procedure used on the live dev database."""

    def setUp(self) -> None:
        self.url, self.path = _temp_sqlite_url()
        # Build only the legacy schema (as it existed before Phase 2) and seed one
        # "existing" user with progress, mirroring the live data/asl_quest.db shape.
        _run_alembic("upgrade", self.url, "0001_baseline")
        engine = sa.create_engine(self.url)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            user = User(username="preexisting", email="preexisting@example.com", password_hash="x")
            session.add(user)
            session.flush()
            session.add(models.LetterProgress(user_id=user.id, letter="A", attempts=3, correct_attempts=2))
            session.commit()
            self.user_id = user.id
        engine.dispose()

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_upgrade_preserves_existing_rows(self) -> None:
        engine = sa.create_engine(self.url)
        with engine.connect() as conn:
            before_users = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar_one()
            before_letters = conn.execute(sa.text("SELECT COUNT(*) FROM letter_progress")).scalar_one()
        engine.dispose()

        _run_alembic("upgrade", self.url)  # -> head (0002 + 0003)

        engine = sa.create_engine(self.url)
        with engine.connect() as conn:
            after_users = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar_one()
            after_letters = conn.execute(sa.text("SELECT COUNT(*) FROM letter_progress")).scalar_one()
            username = conn.execute(
                sa.text("SELECT username FROM users WHERE id = :id"), {"id": self.user_id}
            ).scalar_one()
        engine.dispose()

        self.assertEqual(before_users, after_users)
        self.assertEqual(before_letters, after_letters)
        self.assertEqual(username, "preexisting")


class NewModelCreationTests(unittest.TestCase):
    """New ORM models can be created and related to each other correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.url, cls.path = _temp_sqlite_url()
        engine = sa.create_engine(cls.url)
        Base.metadata.create_all(bind=engine)
        cls.Session = sessionmaker(bind=engine)
        cls.engine = engine

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()
        cls.path.unlink(missing_ok=True)

    def test_native_sign_full_relationship_graph(self) -> None:
        with self.Session() as session:
            user = User(username="signer", email="signer@example.com", password_hash="x")
            sign = NativeSign(gloss="CAT", display_name="Cat", category="Animals", dataset_available=True)
            session.add_all([user, sign])
            session.flush()

            session.add(
                NativeSignReference(
                    native_sign_id=sign.id,
                    source_type="asl_citizen",
                    source_identifier="12345-CAT.mp4",
                    license_note="Microsoft Research License - not redistributed",
                    is_primary=True,
                )
            )
            model_version = ModelVersion(
                model_type="native_sign",
                version="v1",
                checkpoint_path="outputs/native_sign_10/checkpoints/best.pth",
                num_classes=10,
                active=True,
            )
            session.add(model_version)
            session.flush()

            practice_session = NativeSignSession(
                user_id=user.id,
                native_sign_id=sign.id,
                model_version="v1",
                result="correct",
                confidence=0.91,
            )
            session.add(practice_session)
            session.flush()

            session.add(
                NativeSignProgress(
                    user_id=user.id,
                    native_sign_id=sign.id,
                    attempts=1,
                    correct_attempts=1,
                    accuracy=1.0,
                    best_confidence=0.91,
                )
            )
            session.add(
                SignPrediction(
                    user_id=user.id,
                    session_id=practice_session.id,
                    model_version_id=model_version.id,
                    model_type="native_sign",
                    model_version="v1",
                    predicted_label="CAT",
                    expected_label="CAT",
                    confidence=0.91,
                    correct=True,
                )
            )
            session.commit()

            reloaded = session.get(User, user.id)
            self.assertEqual(len(reloaded.native_sign_progress), 1)
            self.assertEqual(len(reloaded.native_sign_sessions), 1)
            self.assertEqual(len(reloaded.sign_predictions), 1)
            reloaded_sign = session.get(NativeSign, sign.id)
            self.assertEqual(len(reloaded_sign.references_), 1)
            self.assertEqual(reloaded_sign.progress[0].accuracy, 1.0)

    def test_chatbot_conversation_thread(self) -> None:
        with self.Session() as session:
            user = User(username="chatuser", email="chatuser@example.com", password_hash="x")
            session.add(user)
            session.flush()

            conversation = ChatbotConversation(user_id=user.id, title="Learning HELLO")
            session.add(conversation)
            session.flush()

            message = ChatbotMessage(conversation_id=conversation.id, role="user", content="How do I sign HELLO?")
            session.add(message)
            session.flush()

            session.add(ChatbotFeedback(message_id=message.id, rating=5, feedback="Helpful"))
            session.commit()

            reloaded = session.get(ChatbotConversation, conversation.id)
            self.assertEqual(len(reloaded.messages), 1)
            self.assertEqual(reloaded.messages[0].feedback[0].rating, 5)

    def test_user_preference_one_to_one(self) -> None:
        with self.Session() as session:
            user = User(username="prefuser", email="prefuser@example.com", password_hash="x")
            session.add(user)
            session.flush()
            session.add(UserPreference(user_id=user.id, theme="dark", language="en"))
            session.commit()

            reloaded = session.get(User, user.id)
            self.assertEqual(reloaded.preference.theme, "dark")

    def test_foreign_keys_declared_correctly(self) -> None:
        inspector = sa.inspect(self.engine)
        fks = {fk["referred_table"] for fk in inspector.get_foreign_keys("native_sign_progress")}
        self.assertEqual(fks, {"users", "native_signs"})
        fks = {fk["referred_table"] for fk in inspector.get_foreign_keys("sign_predictions")}
        self.assertEqual(fks, {"users", "native_sign_sessions", "model_versions"})
        fks = {fk["referred_table"] for fk in inspector.get_foreign_keys("chatbot_messages")}
        self.assertEqual(fks, {"chatbot_conversations"})


class UniqueConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.url, cls.path = _temp_sqlite_url()
        engine = sa.create_engine(cls.url)
        Base.metadata.create_all(bind=engine)
        cls.Session = sessionmaker(bind=engine)
        cls.engine = engine

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()
        cls.path.unlink(missing_ok=True)

    def test_duplicate_native_sign_progress_rejected(self) -> None:
        with self.Session() as session:
            user = User(username="dupuser", email="dupuser@example.com", password_hash="x")
            sign = NativeSign(gloss="DOG", display_name="Dog")
            session.add_all([user, sign])
            session.flush()
            session.add(NativeSignProgress(user_id=user.id, native_sign_id=sign.id))
            session.commit()

            session.add(NativeSignProgress(user_id=user.id, native_sign_id=sign.id))
            with self.assertRaises(IntegrityError):
                session.commit()

    def test_duplicate_gloss_rejected(self) -> None:
        with self.Session() as session:
            session.add(NativeSign(gloss="BIRD", display_name="Bird"))
            session.commit()
            session.add(NativeSign(gloss="BIRD", display_name="Bird (dup)"))
            with self.assertRaises(IntegrityError):
                session.commit()

    def test_duplicate_model_version_rejected(self) -> None:
        with self.Session() as session:
            session.add(ModelVersion(model_type="native_sign", version="v1", checkpoint_path="a.pth"))
            session.commit()
            session.add(ModelVersion(model_type="native_sign", version="v1", checkpoint_path="b.pth"))
            with self.assertRaises(IntegrityError):
                session.commit()

    def test_duplicate_word_id_rejected(self) -> None:
        with self.Session() as session:
            session.add(Word(id="beginner-cat", word="CAT", category="Beginner", difficulty="Easy"))
            session.commit()
            session.add(Word(id="beginner-cat", word="CAT2", category="Beginner", difficulty="Easy"))
            with self.assertRaises(IntegrityError):
                session.commit()


class WordCatalogMappingTests(unittest.TestCase):
    """Every catalog word ends up in the `words` table under its existing ID, and
    pre-existing WordProgress rows (which only ever stored that same string ID) keep
    resolving correctly with zero changes to WordProgress itself."""

    def setUp(self) -> None:
        self.url, self.path = _temp_sqlite_url()
        _run_alembic("upgrade", self.url)
        self.engine = sa.create_engine(self.url)
        self.Session = sessionmaker(bind=self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()
        self.path.unlink(missing_ok=True)

    def test_every_catalog_id_present_as_word_row(self) -> None:
        with self.Session() as session:
            db_ids = {row[0] for row in session.execute(sa.select(Word.id))}
        catalog_ids = {item["id"] for item in WORD_CATALOG}
        self.assertEqual(db_ids, catalog_ids)

    def test_catalog_fields_match(self) -> None:
        with self.Session() as session:
            cat = session.get(Word, "beginner-cat")
        self.assertIsNotNone(cat)
        self.assertEqual(cat.word, "CAT")
        self.assertEqual(cat.category, "Beginner")
        self.assertTrue(cat.spelling_enabled)
        self.assertFalse(cat.native_sign_enabled)

    def test_preexisting_word_progress_resolves_by_id_equality(self) -> None:
        """Simulates a WordProgress row that existed before Phase 2 (word_id is a
        plain unconstrained string, as it always was) and confirms it still resolves
        to the corresponding words row purely by ID equality -- no data migration of
        WordProgress itself was needed or performed."""
        with self.Session() as session:
            user = User(username="worduser", email="worduser@example.com", password_hash="x")
            session.add(user)
            session.flush()
            session.add(WordProgress(user_id=user.id, word_id="beginner-cat", attempts=1, completions=1))
            session.commit()

            progress = session.execute(
                sa.select(WordProgress).where(WordProgress.user_id == user.id)
            ).scalar_one()
            resolved_word = session.get(Word, progress.word_id)
            self.assertIsNotNone(resolved_word)
            self.assertEqual(resolved_word.word, "CAT")


class DatabaseUrlConfigurationTests(unittest.TestCase):
    """backend/settings.py's DATABASE_URL resolution order, tested via subprocess so
    each case gets a clean, unimported module (settings.py computes this constant once
    at import time)."""

    def _resolved_url(self, env_overrides: dict[str, str]) -> str:
        env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "ASL_QUEST_DATABASE_URL")}
        env.update(env_overrides)
        result = subprocess.run(
            [sys.executable, "-c", "from backend.settings import DATABASE_URL; print(DATABASE_URL)"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_database_url_env_var_takes_priority(self) -> None:
        url = self._resolved_url({"DATABASE_URL": "postgresql+psycopg://user:pass@localhost/asl_quest"})
        self.assertEqual(url, "postgresql+psycopg://user:pass@localhost/asl_quest")

    def test_legacy_env_var_used_when_database_url_unset(self) -> None:
        url = self._resolved_url({"ASL_QUEST_DATABASE_URL": "sqlite:///legacy_fallback.db"})
        self.assertEqual(url, "sqlite:///legacy_fallback.db")

    def test_database_url_takes_priority_over_legacy_var(self) -> None:
        url = self._resolved_url(
            {
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/asl_quest",
                "ASL_QUEST_DATABASE_URL": "sqlite:///legacy_fallback.db",
            }
        )
        self.assertEqual(url, "postgresql+psycopg://user:pass@localhost/asl_quest")

    def test_defaults_to_local_sqlite_when_nothing_set(self) -> None:
        url = self._resolved_url({})
        self.assertTrue(url.startswith("sqlite:///"))
        self.assertTrue(url.endswith("asl_quest.db"))


class LegacyCreateAllRestrictionTests(unittest.TestCase):
    """backend.database.migrate_schema() must only ever create the original legacy
    tables via its implicit create_all -- new Phase 2 tables must come from Alembic
    only (see backend/database.py::_LEGACY_TABLE_NAMES).

    Runs in a subprocess (rather than reloading backend.database/backend.settings
    in-process) because those modules are already imported at the top of this test
    file for other tests; reloading backend.database alone would rebind its `Base` to
    a fresh, empty MetaData while backend.models (still cached) stays bound to the
    original one, corrupting the test rather than exercising real app behavior. A
    subprocess is exactly what actually happens in production: one fresh interpreter,
    one import of backend.database, one call to init_db().
    """

    def test_migrate_schema_does_not_create_new_tables(self) -> None:
        url, path = _temp_sqlite_url()
        try:
            path.unlink(missing_ok=True)  # init_db() should create it fresh
            env = dict(os.environ)
            env["ASL_QUEST_DATABASE_URL"] = url
            script = (
                "from backend.database import init_db, engine; "
                "init_db(); "
                "import sqlalchemy as sa; "
                "print(sorted(sa.inspect(engine).get_table_names()))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            table_names = set(eval(result.stdout.strip()))  # noqa: S307 - trusted, our own subprocess output

            for name in _LEGACY_TABLE_NAMES:
                self.assertIn(name, table_names)
            for name in NEW_TABLE_NAMES:
                self.assertNotIn(name, table_names, f"{name} should only be created via Alembic")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
