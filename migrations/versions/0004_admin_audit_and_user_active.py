"""admin audit log and users.is_active

Revision ID: 0004_admin_audit
Revises: 0003_seed_words
Create Date: 2026-10-22 00:00:00.000000

Idempotent: the app's startup migrator (backend/database.py) also creates these on existing databases.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_admin_audit"
down_revision: Union[str, Sequence[str], None] = "0003_seed_words"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "is_active" not in {c["name"] for c in inspector.get_columns("users")}:
        op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    if "admin_audit_log" not in inspector.get_table_names():
        op.create_table(
            "admin_audit_log",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("admin_id", sa.Integer(), nullable=True),
            sa.Column("admin_username", sa.String(length=64), nullable=False),
            sa.Column("action", sa.String(length=48), nullable=False),
            sa.Column("target_type", sa.String(length=32), nullable=False),
            sa.Column("target_id", sa.String(length=128), nullable=False),
            sa.Column("detail", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["admin_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_admin_audit_log_admin_id", "admin_audit_log", ["admin_id"])
        op.create_index("ix_admin_audit_log_action", "admin_audit_log", ["action"])
        op.create_index("ix_admin_audit_log_created_at", "admin_audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("admin_audit_log")
    op.drop_column("users", "is_active")
