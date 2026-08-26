"""
db.py — database engine and session handling.

One table for now (users). See models.py.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Local default matches the `db` service in docker-compose.yml.
# Render supplies the real Neon URL through this same variable.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://carvalue:carvalue@db:5432/carvalue",
)

# Neon (and Heroku) hand out URLs starting with `postgres://`, a scheme
# SQLAlchemy 2.x dropped. Rewrite it rather than making the human do it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class Base(DeclarativeBase):
    """Parent class for every table definition."""


# pool_pre_ping: Neon auto-suspends after ~5 min idle and drops pooled
# connections. Without this the first request after a quiet spell fails on
# a dead socket; with it SQLAlchemy tests and transparently reconnects.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=2,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session():
    """Yield a session, committing on success and rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create any missing tables. Safe to call on every boot."""
    # Importing models registers the mappers on Base before create_all runs.
    import models  # noqa: F401

    try:
        Base.metadata.create_all(engine, checkfirst=True)
    except Exception as exc:
        # Both gunicorn workers boot at once and may race on CREATE TABLE.
        # checkfirst handles the common case; this catches the narrow window
        # where both pass the check and one loses the insert into pg_class.
        print(f"[init_db] table creation skipped: {exc}")
