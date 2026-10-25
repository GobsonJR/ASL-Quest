"""READ-ONLY deployment self-check: run it inside the environment you want to inspect (e.g. the Render shell).

    python scripts/inspect_deployment_db.py

It only reads. It never creates, alters or inserts anything, never runs Alembic, and never prints a secret: connection strings,
the JWT secret, API keys and the bootstrap e-mail are reported only as set/unset or true/false. Share its output freely.

What it tells you
  * which database engine is in use and whether it is reachable;
  * the Alembic revision, and a plain-language verdict about migration state (including the ambiguous "tables exist but Alembic was never
    initialised" case, where `alembic upgrade head` must NOT be run blindly);
  * whether `users.is_active`, `admin_audit_log` and its indexes exist;
  * how many users / admins / native signs / references / words exist (no names, no e-mails);
  * whether the JWT secret is still the public development default, and which feature flags are on;
  * which model files exist on this machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text  # noqa: E402

from backend.database import Base, engine  # noqa: E402
from backend.settings import (  # noqa: E402
    BOOTSTRAP_ADMIN_EMAIL,
    CORS_ORIGINS,
    DEV_COLLECT,
    DEV_DIAGNOSTICS,
    EXPERIMENTAL_LANDMARK_MODEL,
    RECOGNIZER,
    SECRET_KEY,
)
from backend import models  # noqa: E402,F401  (registers every table on Base.metadata)

KNOWN_DEFAULT_SECRET = "dev-change-me-in-production"
LATEST_REVISION = "0004_admin_audit"
AUDIT_INDEXES = {"ix_admin_audit_log_admin_id", "ix_admin_audit_log_action", "ix_admin_audit_log_created_at"}
ENV_NAMES = [
    "DATABASE_URL", "ASL_QUEST_DATABASE_URL", "ASL_QUEST_SECRET_KEY", "ASL_QUEST_CORS_ORIGINS", "ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL",
    "ASL_QUEST_DEV_COLLECT", "ASL_QUEST_DEV_DIAGNOSTICS", "ASL_QUEST_RECOGNIZER", "ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL",
    "ASL_QUEST_HAND_DETECTION_CONFIDENCE", "OPENROUTER_API_KEY", "CHATBOT_PROVIDER", "PORT",
]


def line(label: str, value: object) -> None:
    print(f"{label:<42}{value}")


def main() -> int:
    print("== ASL Quest deployment self-check (read-only) ==")
    line("database engine", engine.dialect.name)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            line("database reachable", "yes")
            inspector = inspect(connection)
            tables = set(inspector.get_table_names())

            revisions: list[str] = []
            if "alembic_version" in tables:
                revisions = [row[0] for row in connection.execute(text("SELECT version_num FROM alembic_version"))]
            line("alembic_version table", "present" if "alembic_version" in tables else "ABSENT")
            line("alembic revision(s)", revisions or "none")

            expected = set(Base.metadata.tables)
            line("expected tables missing", sorted(expected - tables) or "none")
            legacy_present = {"users", "user_progress", "practice_sessions"} <= tables
            if revisions == [LATEST_REVISION]:
                verdict = f"UP TO DATE (head = {LATEST_REVISION})"
            elif revisions:
                verdict = f"BEHIND: at {revisions[0]}; `alembic upgrade head` applies the later revisions (back up first)"
            elif legacy_present:
                verdict = "AMBIGUOUS: tables exist but Alembic was never initialised. Do NOT run `alembic upgrade head` or `stamp` blindly; follow DATABASE.md"
            else:
                verdict = "EMPTY database (a fresh `alembic upgrade head` is expected)"
            line("migration verdict", verdict)

            user_columns = {c["name"] for c in inspector.get_columns("users")} if "users" in tables else set()
            line("users.role column", "present" if "role" in user_columns else "ABSENT")
            line("users.is_active column", "present" if "is_active" in user_columns else "ABSENT")
            line("admin_audit_log table", "present" if "admin_audit_log" in tables else "ABSENT (admin write actions will fail until 0004 is applied)")
            if "admin_audit_log" in tables:
                found = {i["name"] for i in inspector.get_indexes("admin_audit_log")}
                line("admin_audit_log indexes missing", sorted(AUDIT_INDEXES - found) or "none")

            def count(table: str, where: str = "") -> object:
                if table not in tables:
                    return "table missing"
                return connection.execute(text(f"SELECT COUNT(*) FROM {table} {where}")).scalar()

            line("users (total)", count("users"))
            line("users with role = admin", count("users", "WHERE role = 'admin'"))
            line("users deactivated", count("users", "WHERE is_active = FALSE" if "is_active" in user_columns else "WHERE 1 = 0"))
            line("native_signs (active)", count("native_signs", "WHERE active = TRUE" if engine.dialect.name != "sqlite" else "WHERE active = 1"))
            line("native_sign_references", count("native_sign_references"))
            line("words", count("words"))
    except Exception as error:  # noqa: BLE001 - report the failure type only: messages can contain connection details
        line("database reachable", f"NO ({type(error).__name__})")
        return 1

    print("\n-- security / configuration (booleans only) --")
    line("JWT secret is the public default", "YES  <-- BLOCKER, tokens can be forged" if SECRET_KEY == KNOWN_DEFAULT_SECRET else "no")
    line("JWT secret length >= 32", len(SECRET_KEY) >= 32)
    line("bootstrap admin e-mail configured", bool(BOOTSTRAP_ADMIN_EMAIL))
    line("dataset collection enabled (DEV_COLLECT)", DEV_COLLECT)
    line("dev diagnostics enabled", DEV_DIAGNOSTICS)
    line("default recognizer", RECOGNIZER)
    line("experimental landmark model configured", bool(EXPERIMENTAL_LANDMARK_MODEL))
    line("CORS origins allowed", CORS_ORIGINS)
    print("\n-- environment variables (set / unset, never the value) --")
    for name in ENV_NAMES:
        line(name, "set" if os.environ.get(name, "").strip() else "unset")

    print("\n-- model files on THIS machine --")
    models_dir = ROOT / "models"
    for label, path in [
        ("models/best_model.pth (A-Z ResNet)", models_dir / "best_model.pth"),
        ("models/hand_landmarker.task", models_dir / "hand_landmarker.task"),
        ("I3D backbone weights", models_dir / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"),
        ("native head_best.pt", ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "checkpoints" / "head_best.pt"),
    ]:
        line(label, "present" if path.exists() else "MISSING")
    if EXPERIMENTAL_LANDMARK_MODEL:
        configured = Path(EXPERIMENTAL_LANDMARK_MODEL)
        configured = configured if configured.is_absolute() else ROOT / configured
        line("configured landmark .joblib", "present" if configured.exists() else "MISSING")
    local_videos = ROOT / "dataset" / "videos"
    line("local reference videos (dataset/videos)", f"{len(list(local_videos.glob('*.mp4')))} file(s)" if local_videos.exists() else "none (expected on a server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
