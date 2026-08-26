"""
test_api.py — what CI runs on every push.

Grouped by what each test protects:
  1. the ML artifacts        (does the model still load and behave?)
  2. the public endpoints    (does the app boot and serve?)
  3. authentication          (is the gate on /predict real?)
  4. authorization           (does the role check hold?)
  5. password handling       (is anything stored in the clear?)
"""

import pickle

from conftest import CAR
from models import User


# --- 1. ML artifacts -----------------------------------------------------

def test_model_artifacts_load():
    """The three .pkl files exist and deserialize.

    Guards against a corrupt commit or a scikit-learn version mismatch --
    the reason every dependency in requirements.txt is pinned.
    """
    for name in ("model.pkl", "encoders.pkl", "model_feature_list.pkl"):
        with open(name, "rb") as fh:
            assert pickle.load(fh) is not None


def test_model_predicts_in_a_sane_range(client, register):
    """A 2018 Toyota Aqua must price somewhere believable.

    This is the test a generic web pipeline would not have. Every HTTP test
    below still passes if model.pkl is swapped for a broken one -- only this
    assertion catches it.
    """
    res = client.post("/api/predict", json=CAR, headers=register())
    assert res.status_code == 200
    price = res.get_json()["predicted_price"]
    assert 5_000_000 < price < 25_000_000, f"implausible prediction: {price}"


def test_prediction_is_deterministic(client, register):
    """Same input, same output. A model that drifts between calls is broken."""
    headers = register()
    first = client.post("/api/predict", json=CAR, headers=headers).get_json()
    second = client.post("/api/predict", json=CAR, headers=headers).get_json()
    assert first["predicted_price"] == second["predicted_price"]


# --- 2. public endpoints -------------------------------------------------

def test_health_is_public(client):
    """Render probes this with no credentials -- it must never require auth."""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_options_is_public_and_complete(client):
    res = client.get("/api/options")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["brands"]) == 24
    assert len(data["districts"]) == 21
    assert "Toyota" in data["brands"]


# --- 3. authentication ---------------------------------------------------

def test_predict_requires_a_token(client):
    """The whole point of Step 3: no token, no prediction."""
    assert client.post("/api/predict", json=CAR).status_code == 401


def test_predict_rejects_a_tampered_token(client, register):
    """Flipping a byte in the payload must break signature verification."""
    good = register()["Authorization"].split(" ")[1]
    head, payload, sig = good.split(".")
    forged = f"{head}.{payload[:-4]}AAAA.{sig}"
    res = client.post("/api/predict", json=CAR,
                      headers={"Authorization": f"Bearer {forged}"})
    assert res.status_code == 401


def test_register_then_predict(client, register):
    res = client.post("/api/predict", json=CAR, headers=register())
    assert res.status_code == 200
    assert res.get_json()["success"] is True


def test_duplicate_email_is_rejected(client):
    body = {"email": "dup@example.com", "password": "testpassword123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409


def test_login_errors_do_not_reveal_whether_the_email_exists(client, register):
    """Both failures must return the identical message.

    A different message for 'no such user' would let an attacker enumerate
    which addresses have accounts.
    """
    register("real@example.com", "testpassword123")
    wrong_pw = client.post("/api/auth/login",
                           json={"email": "real@example.com", "password": "nope12345"})
    no_user = client.post("/api/auth/login",
                          json={"email": "ghost@example.com", "password": "nope12345"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.get_json() == no_user.get_json()


def test_short_password_is_rejected(client):
    res = client.post("/api/auth/register",
                      json={"email": "weak@example.com", "password": "short"})
    assert res.status_code == 400


# --- 4. authorization ----------------------------------------------------

def test_normal_user_is_forbidden_from_admin_route(client, register):
    """Authenticated but not authorized -> 403, not 401."""
    res = client.get("/api/admin/users", headers=register())
    assert res.status_code == 403
    assert res.get_json()["your_role"] == "user"


def test_admin_can_list_users(client, admin_headers):
    res = client.get("/api/admin/users", headers=admin_headers)
    assert res.status_code == 200
    assert res.get_json()["count"] >= 1


def test_admin_listing_never_leaks_password_hashes(client, admin_headers):
    res = client.get("/api/admin/users", headers=admin_headers)
    for user in res.get_json()["users"]:
        assert "password_hash" not in user
        assert not any("password" in key for key in user)


# --- 5. password handling ------------------------------------------------

def test_password_is_hashed_not_stored_plaintext():
    user = User(email="hash@example.com")
    user.set_password("testpassword123")
    assert user.password_hash != "testpassword123"
    assert user.password_hash.startswith("$2b$12$")   # bcrypt, cost factor 12
    assert len(user.password_hash) == 60
    assert user.check_password("testpassword123") is True
    assert user.check_password("wrong") is False


def test_same_password_produces_different_hashes():
    """Each hash carries its own random salt, so rainbow tables do not work."""
    a, b = User(email="a@x.com"), User(email="b@x.com")
    a.set_password("identical-password")
    b.set_password("identical-password")
    assert a.password_hash != b.password_hash
