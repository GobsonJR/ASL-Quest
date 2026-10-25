"""Refuse unsafe settings in production, and say so loudly about risky ones. Never prints a secret or a connection string.

Only active where it matters: on Render (it sets RENDER=true) or with ASL_QUEST_ENV=production. On a developer machine every check is
skipped, so local work behaves exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KNOWN_DEFAULT_SECRET = "dev-change-me-in-production"
RECOMMENDED_SECRET_LENGTH = 32


@dataclass
class Findings:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def review(
    *,
    production: bool,
    secret: str,
    database_url: str,
    cors_origins: list[str],
    collection_enabled: bool = False,
    diagnostics_enabled: bool = False,
) -> Findings:
    """Pure function (easy to test): what is wrong with these settings for this environment."""
    found = Findings()
    if not production:
        return found
    if secret == KNOWN_DEFAULT_SECRET:
        found.errors.append(
            "ASL_QUEST_SECRET_KEY is not set (the built-in development secret is public, so anyone could forge login tokens, "
            "including an admin's). Set it to a long random value in the host's environment settings."
        )
    elif len(secret) < RECOMMENDED_SECRET_LENGTH:
        found.warnings.append(f"ASL_QUEST_SECRET_KEY is shorter than {RECOMMENDED_SECRET_LENGTH} characters; use a longer random value.")
    if database_url.startswith("sqlite"):
        found.warnings.append(
            "The database is SQLite on the server's local disk. On Render that disk is wiped on every deploy, so accounts and progress "
            "would be lost. Set DATABASE_URL to the PostgreSQL service."
        )
    if cors_origins and all("localhost" in origin or "127.0.0.1" in origin for origin in cors_origins):
        found.warnings.append("ASL_QUEST_CORS_ORIGINS only lists local origins: the deployed frontend's browser requests will be refused.")
    if collection_enabled:
        found.warnings.append("Dataset collection (ASL_QUEST_DEV_COLLECT) is ON in production: collected images can show identifiable people.")
    if diagnostics_enabled:
        found.warnings.append("Live-request diagnostics (ASL_QUEST_DEV_DIAGNOSTICS) are ON in production and save camera frames to disk.")
    return found


def enforce() -> Findings:
    """Called once at startup. Raises RuntimeError (so the host keeps the previous deploy) if a production error is found."""
    from backend.settings import CORS_ORIGINS, DATABASE_URL, DEV_COLLECT, DEV_DIAGNOSTICS, IS_PRODUCTION, SECRET_KEY

    found = review(
        production=IS_PRODUCTION,
        secret=SECRET_KEY,
        database_url=DATABASE_URL,
        cors_origins=CORS_ORIGINS,
        collection_enabled=DEV_COLLECT,
        diagnostics_enabled=DEV_DIAGNOSTICS,
    )
    for warning in found.warnings:
        print(f"[startup warning] {warning}")
    if found.errors:
        raise RuntimeError("Refusing to start with unsafe production settings: " + " | ".join(found.errors))
    return found
