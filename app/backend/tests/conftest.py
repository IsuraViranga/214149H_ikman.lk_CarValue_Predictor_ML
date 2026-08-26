"""
conftest.py — shared pytest setup.

Tests run against a REAL Postgres (a separate `carvalue_test` database), not
SQLite, so what CI verifies is the same engine production uses. A SQLite pass
would prove very little: it has different types, different constraint handling
and no JSONB.
"""

import os
import pathlib
import sys

import pytest

# Make app.py / db.py / models.py importable from inside tests/
BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# These MUST be set before importing db.py -- it builds the SQLAlchemy engine
# at import time, so a later change would have no effect.
# ---------------------------------------------------------------------------
# >= 32 bytes: RFC 7518 s3.2 requires the HMAC key to be at least as long
# as the hash output (32 bytes for SHA-256). PyJWT warns below that.
os.environ["JWT_SECRET_KEY"] = "test-secret-only-for-pytest-never-used-in-production"

# Local docker compose -> the `db` service. GitHub Actions overrides this to
# point at the postgres service container on localhost.
_BASE = os.environ.get("TEST_DATABASE_BASE", "postgresql://carvalue:carvalue@db:5432")
TEST_DB_NAME = "carvalue_test"
os.environ["DATABASE_URL"] = f"{_BASE}/{TEST_DB_NAME}"


def _ensure_test_database():
    """CREATE DATABASE carvalue_test if it does not already exist.

    Runs against the maintenance database, and needs AUTOCOMMIT because
    Postgres refuses CREATE DATABASE inside a transaction block.
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    conn = psycopg2.connect(f"{_BASE}/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    conn.close()


_ensure_test_database()

# Safe to import now that DATABASE_URL points at the test database.
import app as app_module          # noqa: E402
from db import get_session        # noqa: E402
from models import User           # noqa: E402


@pytest.fixture(scope="session")
def flask_app():
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture(autouse=True)
def clean_users():
    """Empty the users table around every test.

    autouse means no test has to remember to ask for it, and each test starts
    from a known state regardless of what ran before it.
    """
    with get_session() as session:
        session.query(User).delete()
    yield
    with get_session() as session:
        session.query(User).delete()


# --- helpers -------------------------------------------------------------

@pytest.fixture()
def register(client):
    """Create an account and return its auth header."""
    def _register(email="user@example.com", password="testpassword123"):
        res = client.post("/api/auth/register", json={"email": email, "password": password})
        assert res.status_code == 201, res.get_json()
        return {"Authorization": f"Bearer {res.get_json()['access_token']}"}
    return _register


@pytest.fixture()
def admin_headers(client, register):
    """Create an account, promote it in the database, then log in again.

    The re-login matters: the role is a claim inside the token, so a token
    issued before the promotion still says role=user.
    """
    email, password = "admin@example.com", "adminpassword123"
    register(email, password)

    with get_session() as session:
        session.query(User).filter_by(email=email).update({"role": "admin"})

    res = client.post("/api/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200
    return {"Authorization": f"Bearer {res.get_json()['access_token']}"}


CAR = {
    "brand": "Toyota", "model": "Aqua", "condition": "Used",
    "transmission": "Automatic", "body_type": "Hatchback", "fuel_type": "Hybrid",
    "district": "Colombo", "year": 2018, "mileage_km": 85000,
    "engine_cc": 1500, "has_trim": 1,
}
