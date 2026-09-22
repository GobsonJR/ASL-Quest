"""Explicit, manually-run SQLite -> PostgreSQL data import for ASL Quest.

This script is NEVER run automatically. It exists so that, when you are ready
to move from the local SQLite dev database to a real PostgreSQL database, you
have a safe, auditable way to copy existing rows across.

What it does:
  1. Connects to the SQLite source (read-only intent; no writes are ever made
     to the source database) and the PostgreSQL target you specify.
  2. Refuses to run if the target already has rows in any of these tables,
     unless you pass --force (safety: avoids ever overwriting/duplicating
     data in a target that already has something in it).
  3. Refuses to run if the target schema is not already at the Alembic
     "head" revision (you must run migrations against the target first).
  4. Copies every row, table by table, in foreign-key-safe order (derived
     from SQLAlchemy's metadata, not hand-maintained), preserving primary
     keys exactly as they are in SQLite.
  5. Does the entire copy inside a single transaction on the target. Any
     error rolls back the whole import -- it never leaves the target
     partially populated.
  6. After a successful copy, resets PostgreSQL's auto-increment sequences
     so new rows created by the app afterwards don't collide with imported
     IDs.
  7. Prints a before/after row-count report for every table so you can
     verify nothing was lost or duplicated.

USAGE (manual only -- read this before running):

    # 1. Make sure the target Postgres database exists and is empty, and that
    #    you have already run migrations against it:
    DATABASE_URL="postgresql+psycopg://user:pass@host:5432/asl_quest" \\
        python -m alembic upgrade head

    # 2. Dry run first -- shows what WOULD be copied, writes nothing:
    python scripts/migrate_sqlite_to_postgres.py \\
        --source-url "sqlite:///data/asl_quest.db" \\
        --target-url "postgresql+psycopg://user:pass@host:5432/asl_quest" \\
        --dry-run

    # 3. Real run:
    python scripts/migrate_sqlite_to_postgres.py \\
        --source-url "sqlite:///data/asl_quest.db" \\
        --target-url "postgresql+psycopg://user:pass@host:5432/asl_quest"

--source-url defaults to the project's local SQLite dev database
(data/asl_quest.db) if omitted. --target-url is always required and must be
a postgresql(+psycopg) URL -- this script refuses to "import" into another
SQLite file, since that isn't its purpose.

This script does not delete, reset, or modify data/asl_quest.db in any way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

from backend import models  # noqa: E402,F401  (registers all tables on Base.metadata)
from backend.database import Base  # noqa: E402
from backend.settings import DEFAULT_DATABASE_PATH  # noqa: E402


class MigrationAborted(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-url",
        default=f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}",
        help="SQLAlchemy URL for the SQLite source database (default: the project's data/asl_quest.db).",
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="SQLAlchemy URL for the PostgreSQL target database (must start with postgresql).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts and planned actions without writing anything to the target.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow import into a target that already has rows in one or more tables. Use with care.",
    )
    return parser.parse_args()


def _row_counts(engine: Engine, tables: list[sa.Table]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for table in tables:
            counts[table.name] = conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()
    return counts


def _check_target_schema_at_head(target_engine: Engine) -> None:
    from alembic.script import ScriptDirectory
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    expected_head = script.get_current_head()

    with target_engine.connect() as conn:
        has_version_table = sa.inspect(conn).has_table("alembic_version")
        if not has_version_table:
            raise MigrationAborted(
                "Target database has no alembic_version table. Run "
                "`DATABASE_URL=<target> alembic upgrade head` against the target before importing."
            )
        current = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
    if current != expected_head:
        raise MigrationAborted(
            f"Target database is at Alembic revision {current!r}, expected head {expected_head!r}. "
            "Run `DATABASE_URL=<target> alembic upgrade head` against the target before importing."
        )


def _check_source_is_sqlite(url: str) -> None:
    if not url.startswith("sqlite"):
        raise MigrationAborted(f"--source-url must be a sqlite:// URL, got: {url}")


def _check_target_is_postgres(url: str) -> None:
    if not url.startswith("postgresql"):
        raise MigrationAborted(f"--target-url must be a postgresql(+psycopg):// URL, got: {url}")


def main() -> int:
    args = parse_args()

    try:
        _check_source_is_sqlite(args.source_url)
        _check_target_is_postgres(args.target_url)
    except MigrationAborted as exc:
        print(f"ABORTED: {exc}")
        return 1

    source_engine = sa.create_engine(args.source_url)
    target_engine = sa.create_engine(args.target_url)

    tables = list(Base.metadata.sorted_tables)  # FK-dependency order, computed by SQLAlchemy

    print("=== Source row counts (SQLite) ===")
    source_counts = _row_counts(source_engine, tables)
    for name, count in source_counts.items():
        print(f"  {name}: {count}")

    try:
        _check_target_schema_at_head(target_engine)
    except MigrationAborted as exc:
        print(f"\nABORTED: {exc}")
        return 1

    print("\n=== Target row counts (PostgreSQL) before import ===")
    target_counts_before = _row_counts(target_engine, tables)
    for name, count in target_counts_before.items():
        print(f"  {name}: {count}")

    nonempty = {name: count for name, count in target_counts_before.items() if count > 0}
    if nonempty and not args.force:
        print(f"\nABORTED: target already has data in: {nonempty}. Re-run with --force to import anyway.")
        return 1

    if args.dry_run:
        print("\n--dry-run set: no data was written. Re-run without --dry-run to perform the import.")
        return 0

    print("\n=== Copying data (single transaction; any error rolls back everything) ===")
    with source_engine.connect() as source_conn:
        with target_engine.begin() as target_conn:  # begin() = commit on success, rollback on exception
            try:
                for table in tables:
                    rows = [dict(row._mapping) for row in source_conn.execute(sa.select(table))]
                    if not rows:
                        continue
                    target_conn.execute(table.insert(), rows)
                    print(f"  copied {len(rows)} row(s) into {table.name}")

                if target_engine.dialect.name == "postgresql":
                    _reset_postgres_sequences(target_conn, tables)
            except Exception as exc:  # noqa: BLE001 - intentional: report, let `with` roll back, then re-raise
                print(f"\nERROR during copy, rolling back entire import: {exc}")
                raise

    print("\n=== Target row counts (PostgreSQL) after import ===")
    target_counts_after = _row_counts(target_engine, tables)
    mismatches = []
    for name in source_counts:
        expected = target_counts_before.get(name, 0) + source_counts[name]
        actual = target_counts_after.get(name, 0)
        status = "OK" if actual == expected else "MISMATCH"
        if status == "MISMATCH":
            mismatches.append(name)
        print(f"  {name}: {actual} (expected {expected}) [{status}]")

    if mismatches:
        print(f"\nWARNING: row count mismatches in: {mismatches}. Investigate before trusting this import.")
        return 1

    print("\nImport complete. Source SQLite database was not modified.")
    return 0


def _reset_postgres_sequences(target_conn, tables: list[sa.Table]) -> None:
    for table in tables:
        pk_cols = [c for c in table.primary_key.columns if c.autoincrement is not False]
        for col in pk_cols:
            if not str(col.type).startswith("INTEGER"):
                continue
            target_conn.execute(
                sa.text(
                    "SELECT setval(pg_get_serial_sequence(:table, :col), "
                    "COALESCE((SELECT MAX(\"" + col.name + "\") FROM \"" + table.name + "\"), 1), "
                    "(SELECT MAX(\"" + col.name + "\") FROM \"" + table.name + "\") IS NOT NULL)"
                ),
                {"table": table.name, "col": col.name},
            )


if __name__ == "__main__":
    sys.exit(main())
