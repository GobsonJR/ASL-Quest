"""seed words table from backend/words/catalog.py

Revision ID: 0003_seed_words
Revises: 0002_new_tables
Create Date: 2026-09-16 00:00:00.000000

Data-only migration (no schema changes). Populates the `words` table with the
existing spelling-practice catalog from backend/words/catalog.py, using the
catalog's own string IDs (e.g. "beginner-cat") as the primary key. Those are
exactly the IDs already stored in WordProgress.word_id / WordPracticeSession.word_id,
so existing progress rows keep resolving correctly with no changes to their data.
Idempotent: only inserts rows for catalog IDs not already present.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_seed_words"
down_revision: Union[str, Sequence[str], None] = "0002_new_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

words_table = sa.table(
    "words",
    sa.column("id", sa.String),
    sa.column("word", sa.String),
    sa.column("category", sa.String),
    sa.column("difficulty", sa.String),
    sa.column("description", sa.Text),
    sa.column("spelling_enabled", sa.Boolean),
    sa.column("native_sign_enabled", sa.Boolean),
    sa.column("created_at", sa.DateTime),
)


def _catalog_rows() -> list[dict]:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from backend.words.catalog import WORD_CATALOG

    now = datetime.now(timezone.utc)
    return [
        {
            "id": item["id"],
            "word": item["word"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "description": item.get("description"),
            "spelling_enabled": True,
            "native_sign_enabled": False,
            "created_at": now,
        }
        for item in WORD_CATALOG
    ]


def upgrade() -> None:
    connection = op.get_bind()
    rows = _catalog_rows()
    existing_ids = {
        row[0] for row in connection.execute(sa.select(words_table.c.id))
    }
    to_insert = [row for row in rows if row["id"] not in existing_ids]
    if to_insert:
        op.bulk_insert(words_table, to_insert)


def downgrade() -> None:
    rows = _catalog_rows()
    ids = [row["id"] for row in rows]
    if ids:
        op.execute(words_table.delete().where(words_table.c.id.in_(ids)))
