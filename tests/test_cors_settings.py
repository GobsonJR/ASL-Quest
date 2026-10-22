"""CORS origin configuration: the deployed frontend origin must be parsed exactly as the browser sends it."""
from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

VERCEL = "https://asl-quest-1.vercel.app"


def _reload_settings(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("ASL_QUEST_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("ASL_QUEST_CORS_ORIGINS", value)
    import backend.settings as settings

    return importlib.reload(settings)


def test_default_is_local_dev_plus_deployed_frontend(monkeypatch):
    settings = _reload_settings(monkeypatch, None)
    assert settings.CORS_ORIGINS == ["http://127.0.0.1:5173", "http://localhost:5173", VERCEL]
    assert "*" not in settings.CORS_ORIGINS


def test_blank_env_var_uses_defaults(monkeypatch):
    settings = _reload_settings(monkeypatch, "   ")
    assert VERCEL in settings.CORS_ORIGINS


def test_trailing_slash_and_spaces_are_normalised(monkeypatch):
    settings = _reload_settings(monkeypatch, f" {VERCEL}/ , http://localhost:5173 ,")
    assert settings.CORS_ORIGINS == [VERCEL, "http://localhost:5173"]


def test_preflight_allows_configured_origin_only(monkeypatch):
    settings = _reload_settings(monkeypatch, f"{VERCEL}/")
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/auth/login")
    def login() -> dict:
        return {"ok": True}

    client = TestClient(app)
    headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}

    allowed = client.options("/auth/login", headers={"Origin": VERCEL, **headers})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == VERCEL

    blocked = client.options("/auth/login", headers={"Origin": "https://evil.example.com", **headers})
    assert blocked.status_code == 400


def test_real_app_preflight_for_login(monkeypatch):
    """The actual backend.main app (not a stand-in) answers the browser's login preflight for the Vercel origin."""
    monkeypatch.delenv("ASL_QUEST_CORS_ORIGINS", raising=False)
    import sys

    for name in ("backend.main", "backend.settings"):
        sys.modules.pop(name, None)
    from backend.main import app

    client = TestClient(app)  # no context manager: startup (model load) is not triggered
    headers = {
        "Origin": VERCEL,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/auth/login", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == VERCEL
    assert response.headers["access-control-allow-credentials"] == "true"
