"""
models.py — table definitions.

users
  id, email, password_hash, role, created_at
"""

from datetime import datetime

import bcrypt
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db import Base

# Work factor for bcrypt. Higher = slower = harder to brute force.
# 12 is the common production default (~250ms per hash on modest hardware).
BCRYPT_ROUNDS = 12

ROLE_USER = "user"
ROLE_ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Stored lowercased and trimmed so Bob@X.com and bob@x.com are one account.
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )

    # bcrypt output is always exactly 60 characters.
    password_hash: Mapped[str] = mapped_column(String(60), nullable=False)

    # Authorization level. No API route can change this -- an admin is made
    # by updating the row directly, so nobody can escalate their own account.
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ROLE_USER, server_default=ROLE_USER
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def set_password(self, raw_password: str) -> None:
        """Hash and store. The plaintext is never written anywhere."""
        # bcrypt silently ignores anything past 72 bytes, so callers must
        # reject longer passwords rather than let them be truncated.
        self.password_hash = bcrypt.hashpw(
            raw_password.encode("utf-8"),
            bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
        ).decode("utf-8")

    def check_password(self, raw_password: str) -> bool:
        """Constant-time comparison -- never use == on hashes."""
        return bcrypt.checkpw(
            raw_password.encode("utf-8"),
            self.password_hash.encode("utf-8"),
        )

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    def to_dict(self) -> dict:
        """Public representation. Deliberately excludes password_hash."""
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<User {self.id} {self.email} role={self.role}>"
