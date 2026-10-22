"""Startup/port-binding regressions (Render "Port scan timeout reached, no open ports detected")."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import connect_args_for  # noqa: E402
from backend.main import bind_address  # noqa: E402


class BindAddressTests(unittest.TestCase):
    def test_uses_platform_port_on_all_interfaces(self) -> None:
        self.assertEqual(bind_address({"PORT": "10000"}), ("0.0.0.0", 10000))
        self.assertEqual(bind_address({"PORT": " 8080 "}), ("0.0.0.0", 8080))

    def test_defaults_to_local_dev_address(self) -> None:
        self.assertEqual(bind_address({}), ("127.0.0.1", 8000))
        self.assertEqual(bind_address({"PORT": ""}), ("127.0.0.1", 8000))

    def test_ignores_invalid_port_values(self) -> None:
        for bad in ("abc", "0", "-5", "70000", "80.5"):
            self.assertEqual(bind_address({"PORT": bad}), ("127.0.0.1", 8000), bad)


class ConnectArgsTests(unittest.TestCase):
    def test_sqlite_keeps_thread_check_off(self) -> None:
        self.assertEqual(connect_args_for("sqlite:///data/asl_quest.db"), {"check_same_thread": False})

    def test_postgres_gets_a_connect_timeout(self) -> None:
        for url in ("postgresql+psycopg://u:p@host/db", "postgresql://u:p@host/db", "postgres://u:p@host/db"):
            self.assertEqual(connect_args_for(url), {"connect_timeout": 15}, url)

    def test_unknown_backends_get_no_extra_args(self) -> None:
        self.assertEqual(connect_args_for("mysql://u:p@host/db"), {})


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Stop exactly the server this test started. On Windows the venv's python.exe is a launcher that
    spawns the real interpreter as a child, so terminating only the launcher would leave the server (and
    its open files) behind."""
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


class RealServerStartupTests(unittest.TestCase):
    def test_python_m_backend_main_listens_on_platform_port(self) -> None:
        """Boot the real app the way a host with only $PORT would and make an HTTP request."""
        port = _free_port()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            env = {
                **os.environ,
                "PORT": str(port),
                "DATABASE_URL": f"sqlite:///{(Path(tmp) / 'startup.db').as_posix()}",
                "PYTHONUNBUFFERED": "1",
            }
            log_path = Path(tmp) / "server.log"
            log_file = open(log_path, "wb")
            proc = subprocess.Popen(
                [sys.executable, "-m", "backend.main"],
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                body = None
                deadline = time.time() + 90
                while time.time() < deadline and proc.poll() is None:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                            self.assertEqual(response.status, 200)
                            body = response.read().decode("utf-8")
                            break
                    except OSError:
                        time.sleep(0.5)
                log_file.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                self.assertIsNotNone(body, f"server never answered on $PORT (exit={proc.poll()}):\n{tail}")
                self.assertIn('"status"', body)
            finally:
                _stop_process_tree(proc)  # only the process this test started
                log_file.close()


if __name__ == "__main__":
    unittest.main()
