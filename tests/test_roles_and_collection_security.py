"""Roles and authorization: who can become an admin, and that every admin/collection endpoint really checks it on the server."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from jose import jwt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "roles-test-secret"

from backend import models  # noqa: E402,F401
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import AdminAuditLog, User  # noqa: E402
from backend.security import ALGORITHM, hash_password  # noqa: E402
from backend.settings import SECRET_KEY  # noqa: E402  (whichever test module imported the app first fixed it)

PASSWORD = "correct-horse-1"


class RoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    # ---------------------------------------------------------------- helpers
    def register(self, name: str, email: str | None = None, **extra) -> str:
        reply = self.client.post("/auth/register", json={"username": name, "email": email or f"{name}@example.com", "password": PASSWORD, **extra})
        self.assertEqual(reply.status_code, 201, reply.text)
        return reply.json()["access_token"]

    def make_admin(self, name: str = "root") -> str:
        with SessionLocal() as db:
            db.add(User(username=name, email=f"{name}@example.com", password_hash=hash_password(PASSWORD), role="admin"))
            db.commit()
        reply = self.client.post("/auth/login", json={"login": name, "password": PASSWORD})
        self.assertEqual(reply.status_code, 200, reply.text)
        return reply.json()["access_token"]

    @staticmethod
    def auth(token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def me(self, token: str) -> dict:
        return self.client.get("/auth/me", headers=self.auth(token)).json()

    # ---------------------------------------------------------------- registration can never make an admin
    def test_registration_always_creates_a_regular_user(self) -> None:
        token = self.register("alice")
        self.assertEqual(self.me(token)["role"], "student")

    def test_a_role_sent_by_the_browser_is_ignored(self) -> None:
        for index, extra in enumerate([{"role": "admin"}, {"is_admin": True}, {"role": "ADMIN", "admin": True}, {"user": {"role": "admin"}}]):
            token = self.register(f"mallory{index}", **extra)
            self.assertEqual(self.me(token)["role"], "student", extra)

    def test_bootstrap_address_becomes_the_first_admin_only_while_no_admin_exists(self) -> None:
        with patch.dict(os.environ, {"ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL": "boss@example.com"}):
            first = self.register("boss")
            self.assertEqual(self.me(first)["role"], "admin")
            # a different e-mail is an ordinary user, and so is a second sign-up using the bootstrap address once an admin exists
            other = self.register("other")
            self.assertEqual(self.me(other)["role"], "student")
            with SessionLocal() as db:
                db.query(User).filter(User.username == "boss").update({"email": "moved@example.com"})
                db.commit()
            squatter = self.register("squatter", email="boss@example.com")  # admin "boss" still exists (renamed address)
            self.assertEqual(self.me(squatter)["role"], "student")

    def test_bootstrap_address_is_ignored_when_an_admin_already_exists(self) -> None:
        self.make_admin("root")
        with patch.dict(os.environ, {"ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL": "boss@example.com"}):
            token = self.register("boss")
        self.assertEqual(self.me(token)["role"], "student")

    # ---------------------------------------------------------------- the role comes from the database, not from the token or browser
    def test_role_claims_inside_a_token_do_nothing(self) -> None:
        token = self.register("eve")
        user_id = self.me(token)["id"]
        payload = {"sub": str(user_id), "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
        forged_right_key = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)  # even a validly signed extra claim
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(forged_right_key)).status_code, 403)
        forged_wrong_key = jwt.encode(payload, "not-the-secret", algorithm=ALGORITHM)
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(forged_wrong_key)).status_code, 401)
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth("not.a.token")).status_code, 401)

    def test_profile_and_preference_updates_cannot_change_the_role(self) -> None:
        token = self.register("frank")
        self.client.put("/users/me/preferences", headers=self.auth(token), json={"role": "admin", "theme": "dark"})
        self.assertEqual(self.me(token)["role"], "student")
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(token)).status_code, 403)

    def test_role_is_read_fresh_so_demotion_and_deactivation_take_effect_immediately(self) -> None:
        admin_token = self.make_admin("root")
        user_token = self.register("grace")
        with SessionLocal() as db:
            grace = db.query(User).filter(User.username == "grace").one()
            grace.role = "admin"
            db.commit()
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(user_token)).status_code, 200)
        with SessionLocal() as db:
            db.query(User).filter(User.username == "grace").update({"role": "student"})
            db.commit()
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(user_token)).status_code, 403)
        with SessionLocal() as db:
            db.query(User).filter(User.username == "grace").update({"is_active": False})
            db.commit()
        self.assertEqual(self.client.get("/auth/me", headers=self.auth(user_token)).status_code, 403)
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(admin_token)).status_code, 200)

    # ---------------------------------------------------------------- admin-created / promoted admins
    def test_only_an_admin_can_create_an_admin_and_it_is_audited(self) -> None:
        admin_token = self.make_admin("root")
        student_token = self.register("henry")
        body = {"username": "newadmin", "email": "newadmin@example.com", "password": PASSWORD, "role": "admin"}

        self.assertEqual(self.client.post("/admin/users", json=body).status_code, 401)
        self.assertEqual(self.client.post("/admin/users", json=body, headers=self.auth(student_token)).status_code, 403)

        created = self.client.post("/admin/users", json=body, headers=self.auth(admin_token))
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["role"], "admin")
        self.assertNotIn("password", created.text.lower())
        login = self.client.post("/auth/login", json={"login": "newadmin", "password": PASSWORD})
        self.assertEqual(self.me(login.json()["access_token"])["role"], "admin")
        with SessionLocal() as db:
            entry = db.query(AdminAuditLog).filter(AdminAuditLog.action == "user.create").one()
            self.assertEqual(entry.admin_username, "root")
            self.assertNotIn(PASSWORD, entry.detail)

        duplicate = self.client.post("/admin/users", json=body, headers=self.auth(admin_token))
        self.assertEqual(duplicate.status_code, 409)
        weak = self.client.post("/admin/users", json={**body, "username": "x2", "email": "x2@example.com", "password": "short"}, headers=self.auth(admin_token))
        self.assertEqual(weak.status_code, 422)
        bad_role = self.client.post("/admin/users", json={**body, "username": "x3", "email": "x3@example.com", "role": "superuser"}, headers=self.auth(admin_token))
        self.assertEqual(bad_role.status_code, 422)

    def test_admin_can_promote_a_user_but_not_change_their_own_role(self) -> None:
        admin_token = self.make_admin("root")
        user_token = self.register("ivy")
        ivy = self.me(user_token)["id"]
        root = self.me(admin_token)["id"]
        self.assertEqual(self.client.patch(f"/admin/users/{ivy}", json={"role": "admin"}, headers=self.auth(user_token)).status_code, 403)
        self.assertEqual(self.client.patch(f"/admin/users/{ivy}", json={"role": "admin"}, headers=self.auth(admin_token)).status_code, 200)
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(user_token)).status_code, 200)
        self.assertEqual(self.client.patch(f"/admin/users/{root}", json={"role": "student"}, headers=self.auth(admin_token)).status_code, 400)

    # ---------------------------------------------------------------- every admin / collection route checks the role on the server
    def protected_routes(self) -> list[tuple[str, str]]:
        # The OpenAPI document lists every registered path/method (app.routes only holds router wrappers in current FastAPI).
        found = []
        for path, operations in app.openapi()["paths"].items():
            if path.startswith("/admin") or path.startswith("/collect"):
                for method in operations:
                    if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                        found.append((method.upper(), path))
        return found

    def test_every_admin_and_collect_route_rejects_anonymous_and_regular_users(self) -> None:
        routes = self.protected_routes()
        self.assertGreater(len(routes), 25)  # the enumeration really found the admin + collection surface
        student_token = self.register("jack")
        for method, template in routes:
            url = template.replace("{", "").replace("}", "")
            url = "".join(ch if ch.isalnum() or ch in "/_-" else "" for ch in url)
            for token, expected in ((None, 401), (student_token, 403)):
                headers = self.auth(token) if token else {}
                reply = self.client.request(method, url, headers=headers)
                self.assertEqual(reply.status_code, expected, f"{method} {template} as {'student' if token else 'anonymous'} -> {reply.status_code}")

    def test_admin_reaches_the_admin_api_and_collection_is_off_unless_enabled(self) -> None:
        admin_token = self.make_admin("root")
        self.assertEqual(self.client.get("/admin/overview", headers=self.auth(admin_token)).status_code, 200)
        self.assertEqual(self.client.get("/admin/users", headers=self.auth(admin_token)).status_code, 200)
        # authorized admin, but the operator has not switched collection on: 404 (not 401/403, and not data)
        self.assertEqual(self.client.get("/collect/status", headers=self.auth(admin_token)).status_code, 404)


if __name__ == "__main__":
    unittest.main()
