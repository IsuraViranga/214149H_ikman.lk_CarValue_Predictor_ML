"""
auth.py — registration, login, and role-based authorization.

Routes (all public unless marked):
  POST /api/auth/register   create an account, return a token
  POST /api/auth/login      exchange credentials for a token
  GET  /api/auth/me         [token] who the caller is

Also exports role_required(), the decorator used to gate admin routes.
"""

import re
from functools import wraps

import bcrypt
from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt, get_jwt_identity, jwt_required
from sqlalchemy.exc import IntegrityError

from db import get_session
from models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_BYTES = 72  # bcrypt ignores everything past 72 bytes

# Pre-computed hash of a throwaway password. Used on login when the email is
# unknown, so a missing account costs the same time as a wrong password --
# otherwise response timing tells an attacker which emails are registered.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt(rounds=12))


def _read_credentials():
    """Pull and validate email/password from the JSON body.

    Returns (email, password, None) or (None, None, (response, status)).
    """
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return None, None, (jsonify({"error": "Email and password are required"}), 400)

    if not EMAIL_RE.match(email):
        return None, None, (jsonify({"error": "That does not look like a valid email"}), 400)

    if len(password) < MIN_PASSWORD_LEN:
        return None, None, (
            jsonify({"error": f"Password must be at least {MIN_PASSWORD_LEN} characters"}),
            400,
        )

    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        # Refuse rather than let bcrypt silently truncate, which would make
        # two different long passwords authenticate the same account.
        return None, None, (
            jsonify({"error": f"Password must be at most {MAX_PASSWORD_BYTES} bytes"}),
            400,
        )

    return email, password, None


def _issue_token(user):
    """Build a signed JWT carrying the user's id and role."""
    return create_access_token(
        # flask-jwt-extended 4.x requires the subject to be a string.
        identity=str(user.id),
        # Baked into the token so authorization needs no database lookup.
        additional_claims={"role": user.role, "email": user.email},
    )


@auth_bp.route("/register", methods=["POST"])
def register():
    email, password, error = _read_credentials()
    if error:
        return error

    try:
        with get_session() as session:
            user = User(email=email)
            user.set_password(password)   # hashed here; plaintext never stored
            session.add(user)
            session.flush()               # assigns user.id before commit
            token = _issue_token(user)
            payload = user.to_dict()
    except IntegrityError:
        # The UNIQUE index on email rejected it. Checking first then inserting
        # would race between two simultaneous signups; letting the database
        # enforce it is the only way that is actually safe.
        return jsonify({"error": "That email is already registered"}), 409

    return jsonify({"access_token": token, "user": payload}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    email, password, error = _read_credentials()
    if error:
        return error

    with get_session() as session:
        user = session.query(User).filter_by(email=email).one_or_none()

        if user is None:
            # Burn the same time a real bcrypt check would take.
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
            ok = False
        else:
            ok = user.check_password(password)

        if not ok:
            # One message for both cases: saying "no such user" would let an
            # attacker enumerate which emails have accounts.
            return jsonify({"error": "Invalid email or password"}), 401

        token = _issue_token(user)
        payload = user.to_dict()

    return jsonify({"access_token": token, "user": payload}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """Echo back the caller's identity, read from their token."""
    user_id = int(get_jwt_identity())
    with get_session() as session:
        user = session.query(User).get(user_id)
        if user is None:
            # Token is validly signed but the account is gone.
            return jsonify({"error": "Account no longer exists"}), 401
        return jsonify({"user": user.to_dict()}), 200


def role_required(*allowed_roles):
    """Gate a route on the role claim inside the caller's token.

    @jwt_required() answers "are you logged in?"  (authentication -> 401)
    @role_required("admin") answers "may you do this?" (authorization -> 403)
    """
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            role = get_jwt().get("role")
            if role not in allowed_roles:
                return jsonify({
                    "error": "Forbidden",
                    "detail": f"Requires role: {' or '.join(allowed_roles)}",
                    "your_role": role,
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator
