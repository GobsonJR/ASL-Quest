import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import MetaData, engine_from_config
from sqlalchemy import pool

from alembic import context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import models  # noqa: E402,F401  (registers all tables on Base.metadata)
from backend.database import Base  # noqa: E402
from backend.settings import DEFAULT_DATABASE_PATH  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Mirrors backend/settings.py's DATABASE_URL resolution (env var DATABASE_URL, legacy
# ASL_QUEST_DATABASE_URL, or local SQLite fallback), but reads the environment directly
# here rather than importing backend.settings' module-level constant. backend.settings
# may already be cached in sys.modules with a DATABASE_URL frozen from an earlier import
# in the same process (e.g. in tests that invoke Alembic programmatically against
# multiple temp databases), so re-reading the environment fresh on every Alembic
# invocation is what makes this env.py behave correctly both from the CLI (a fresh
# process each time) and from Python code that calls alembic.command in-process.
_database_url = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("ASL_QUEST_DATABASE_URL")
    or f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"
)
config.set_main_option("sqlalchemy.url", _database_url)

# ALEMBIC_LEGACY_ONLY=1 restricts autogenerate to the tables that existed before Phase 2
# (see backend/database.py::_LEGACY_TABLE_NAMES). This is a one-off authoring aid used to
# (a) generate the 0001 baseline migration's CREATE TABLE DDL against an empty database,
# and (b) re-verify at any time that the baseline migration still matches the live schema
# (autogenerate against the real DB should then produce an empty diff). Not used at
# runtime; normal `alembic upgrade`/`downgrade` always use the full Base.metadata below.
if os.environ.get("ALEMBIC_LEGACY_ONLY") == "1":
    from backend.database import _LEGACY_TABLE_NAMES

    _legacy_metadata = MetaData()
    for _name in _LEGACY_TABLE_NAMES:
        Base.metadata.tables[_name].to_metadata(_legacy_metadata)
    target_metadata = _legacy_metadata
elif os.environ.get("ALEMBIC_NEW_ONLY") == "1":
    # Same idea as ALEMBIC_LEGACY_ONLY above, inverted: restricts autogenerate to the
    # Phase 2 tables that are NOT in _LEGACY_TABLE_NAMES. Used once to author the 0002
    # migration's CREATE TABLE DDL against an empty database.
    from backend.database import _LEGACY_TABLE_NAMES

    _new_metadata = MetaData()
    for _name, _table in Base.metadata.tables.items():
        if _name not in _LEGACY_TABLE_NAMES:
            _table.to_metadata(_new_metadata)
    target_metadata = _new_metadata
else:
    target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite") if url else False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
