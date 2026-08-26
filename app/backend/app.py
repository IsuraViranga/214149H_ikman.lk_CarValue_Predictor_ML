"""
app.py — Flask Backend for Vehicle Price Prediction
====================================================
Endpoints:
  GET  /api/options      — returns all dropdown options
  POST /api/predict      — accepts car features, returns price + explanation
"""

from datetime import timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, get_jwt, get_jwt_identity, jwt_required
import os, pickle, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

from auth import auth_bp, role_required
from db import get_session, init_db
from models import User

warnings.filterwarnings("ignore")
 
app = Flask(__name__)

# JWT configuration
# The secret signs and verifies every token. Anyone holding it can forge a
# token for any user, so production must set a strong random value in the
# Render dashboard -- never in git.
app.config["JWT_SECRET_KEY"] = os.environ.get(
    "JWT_SECRET_KEY", "dev-secret-not-for-production"
)
# 8 hours: long enough to avoid mid-session logouts without a refresh-token
# flow, short enough that a leaked token is not useful for long.
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=8)
jwt = JWTManager(app)


# Return clean JSON for token problems instead of the library defaults,
# so the frontend can distinguish "log in again" from other failures.
@jwt.expired_token_loader
def _expired(_hdr, _payload):
    return jsonify({"error": "Token expired", "code": "token_expired"}), 401


@jwt.invalid_token_loader
def _invalid(reason):
    return jsonify({"error": "Invalid token", "detail": str(reason)}), 401


@jwt.unauthorized_loader
def _missing(reason):
    return jsonify({"error": "Authorization required", "detail": str(reason)}), 401

# Comma-separated list of allowed frontend origins (set in docker-compose / Render)
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",") if o.strip()
]
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

# Auth routes live in their own blueprint to keep this file focused on the model.
app.register_blueprint(auth_bp)

# Create the users table if it is missing. Safe on every boot.
init_db()

# Load model artifacts 
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("model_feature_list.pkl", "rb") as f:
    FEATURES = pickle.load(f)

# Brand → Model mapping (built from training data) 
BRAND_MODELS = {
    "Audi": ["A1","A3","A4","A5","A6","Q2","Q3","Q4","Q5","Q7","e-tron","Other Model"],
    "BMW": ["218i","225XE","318i","320d","430i","520d","520i","523i","525i","530d","530e","530i","725D","740Le","740Li","740i","M760","X1","X2","X3","X5","X5 M","X6 M","i3","i4","i7","iX3","Other Model"],
    "DFSK": ["Glory","Other Model"],
    "Daihatsu": ["Atrai Wagon","Canbus","Cast Activa","Charade","Hijet","Mira","Move","Rocky","Taft","Terios","Thor","Other Model"],
    "Ford": ["Ecosport","Focus","Kuga","Laser","Mustang","Ranger","Raptor Ranger","Other Model"],
    "Honda": ["Accord","CRV","CRZ","City","Civic","Fit","Fit Aria","Fit Shuttle","Freed","Grace","HR-V","Insight","Jade","N-Box","N-WGN","Vezel","WR-V","ZRV Z","Other Model"],
    "Hyundai": ["Accent","Atos","Creta","Eon","Grand i10","Santa Fe","Tucson","Venue","Other Model"],
    "Kia": ["Carens","Carnival","Cerato","EV5","Picanto","Rio","Seltos","Sorento","Spectra","Sportage","Stonic","Other Model"],
    "Land Rover": ["Defender","Discovery","Discovery Sport","Freelander","Range Rover","Range Rover Evoque","Range Rover PHEV","Range Rover Sport","Range Rover Velar","Other Model"],
    "Lexus": ["GX550","HS250H","LBX","LM 500h","LS500h","LX600","NX300H","RX350","RX450h","Other Model"],
    "MG": ["6","MG4 X","ZS","Other Model"],
    "Mazda": ["2 Skyactive","3","Axela","CX-5","Carol","Familia","Flair","Other Model"],
    "Mercedes Benz": ["A140","A180","A250","C160","C180","C200","C220","C350","CLA 180","CLA 200","CLA 250","CLS","E200","E240","E250","E300","E350","EQB","EQB 300","EQE 300","EQS 450","G400d","GLA 180","GLA 200","GLB","GLE 300D","GLE 400","S300","S350","S400","Vito","Other Model"],
    "Micro": ["Actyon","Almaz","Chery Tiggo Pro4","Kyron","MX 7","Panda","Panda Cross","Rexton","Tivoli","Trend","Other Model"],
    "Mitsubishi": ["4DR","Colt","Delica","EK Custom","Eclipse Cross","Lancer","Mirage","Montero","Outlander","Pajero","Triton GSR","Xpander","eK Wagon","i-MiEV","Other Model"],
    "Nissan": ["AD Wagon","Almera","Aura","Bluebird","Clipper","Dayz","Juke","Leaf","Magnite","March","Navara","Note","Patrol","Qashqai","Roox","Sakura","Serena","Sunny","Sylphy","Teana","Tiida","Wingroad","X-Trail","Other Model"],
    "Other": ["Other Model"],
    "Perodua": ["Axia","Kelisa","Viva Elite","Other Model"],
    "Peugeot": ["3008","407","408","5008","E-2008","Other Model"],
    "Renault": ["KWID","Other Model"],
    "Suzuki": ["A-Star","Alto","Celerio","Ertiga","Escudo","Fronx","Grand Vitara","Hustler","S-Cross","SX4","Spacia","Swift","Vitara","Wagon R","Wagon R Stingray","XBee","Other Model"],
    "Tata": ["Indica","Indigo","Nano","Other Model"],
    "Toyota": ["Allion","Alphard","Aqua","Avanza","Axio","Belta","CHR","Camry","Carina","Corolla","Corona","Crown","Fortuner","Harrier","Hilux","IST","Land Cruiser Prado","Land Cruiser Sahara","Passo","Premio","Prius","RAV4","Raize","Roomy","Rush","Tank","Urban Cruiser","Vellfire","Vios","Vitz","Voxy","Wigo","Yaris","Yaris Ativ","Yaris Cross","Other Model"],
    "Volkswagen": ["Beetle","Golf","ID","ID-4 STYLISH","Passat","Polo","T-Cross","Taigun","Tiguan","Other Model"],
}

CURRENT_YEAR = 2026

# Build model label encoder from training data for inference
MODEL_LE_CLASSES = sorted(set(
    m for models in BRAND_MODELS.values() for m in models
))
le_model_inf = LabelEncoder()
le_model_inf.fit(MODEL_LE_CLASSES)


def encode_model(model_name):
    """Encode model name, fallback to 'Other Model' if unseen."""
    try:
        return int(le_model_inf.transform([model_name])[0])
    except Exception:
        try:
            return int(le_model_inf.transform(["Other Model"])[0])
        except Exception:
            return 0


def feature_contribution(inp_df, feature):
    """Marginal contribution: pred(input) - pred(input with feature=median)."""
    baseline_val = FEATURE_MEDIANS.get(feature, 0)
    modified = inp_df.copy()
    modified[feature] = baseline_val
    pred_full = float(model.predict(inp_df)[0])
    pred_without = float(model.predict(modified)[0])
    return pred_full - pred_without


# Pre-compute feature medians for contribution calculations
_dummy_df = pd.DataFrame([{f: 0 for f in FEATURES}])
FEATURE_MEDIANS = {
    "brand": 11, "model": 50, "condition": 3, "transmission": 0,
    "body_type": 2, "fuel_type": 4, "mileage_km": 30000,
    "engine_cc": 1400, "age": 5, "district": 4, "has_trim": 1
}


# Routes

@app.route("/api/health", methods=["GET"])
def health():
    """Liveness probe — confirms the model artifacts loaded."""
    return jsonify({"status": "ok", "features": len(FEATURES)})


@app.route("/api/options", methods=["GET"])
def get_options():
    """Return all dropdown options for the frontend."""
    return jsonify({
        "brands":        list(encoders["brand"].classes_),
        "brand_models":  BRAND_MODELS,
        "conditions":    list(encoders["condition"].classes_),
        "transmissions": list(encoders["transmission"].classes_),
        "body_types":    list(encoders["body_type"].classes_),
        "fuel_types":    list(encoders["fuel_type"].classes_),
        "districts":     list(encoders["district"].classes_),
        "year_range":    {"min": 1990, "max": 2026},
        "engine_options": [
            {"label": "Electric / N/A (0cc)", "value": 0},
            {"label": "660 cc", "value": 660},
            {"label": "800 cc", "value": 800},
            {"label": "1000 cc", "value": 1000},
            {"label": "1200 cc", "value": 1200},
            {"label": "1300 cc", "value": 1300},
            {"label": "1330 cc", "value": 1330},
            {"label": "1400 cc", "value": 1400},
            {"label": "1490 cc", "value": 1490},
            {"label": "1500 cc", "value": 1500},
            {"label": "1600 cc", "value": 1600},
            {"label": "1800 cc", "value": 1800},
            {"label": "2000 cc", "value": 2000},
            {"label": "2400 cc", "value": 2400},
            {"label": "2500 cc", "value": 2500},
            {"label": "2700 cc", "value": 2700},
            {"label": "3000 cc", "value": 3000},
            {"label": "3200 cc", "value": 3200},
            {"label": "5000 cc", "value": 5000},
        ],
    })


@app.route("/api/predict", methods=["POST"])
# @jwt_required()   <-- DELIBERATELY REMOVED to demonstrate that CI catches a
#                       regression that drops an auth gate. THIS BRANCH MUST
#                       NOT BE MERGED. Restore this line before merging.
def predict():
    try:
        data = request.get_json()

        # Parse & validate inputs
        brand        = str(data.get("brand", "Toyota"))
        model_name   = str(data.get("model", "Other Model"))
        condition    = str(data.get("condition", "Used"))
        transmission = str(data.get("transmission", "Automatic"))
        body_type    = str(data.get("body_type", "Hatchback"))
        fuel_type    = str(data.get("fuel_type", "Petrol"))
        district     = str(data.get("district", "Colombo"))
        year         = int(data.get("year", 2018))
        mileage_km   = int(data.get("mileage_km", 50000))
        engine_cc    = int(data.get("engine_cc", 1500))
        has_trim     = int(data.get("has_trim", 0))

        age = CURRENT_YEAR - year

        # Encode categoricals
        def safe_encode(encoder, val, fallback_idx=0):
            try:
                return int(encoder.transform([val])[0])
            except Exception:
                return fallback_idx

        brand_enc        = safe_encode(encoders["brand"],        brand)
        condition_enc    = safe_encode(encoders["condition"],    condition)
        transmission_enc = safe_encode(encoders["transmission"], transmission)
        body_type_enc    = safe_encode(encoders["body_type"],    body_type)
        fuel_type_enc    = safe_encode(encoders["fuel_type"],    fuel_type)
        district_enc     = safe_encode(encoders["district"],     district)
        model_enc        = encode_model(model_name)

        # Build feature row
        row = {
            "brand":        brand_enc,
            "model":        model_enc,
            "condition":    condition_enc,
            "transmission": transmission_enc,
            "body_type":    body_type_enc,
            "fuel_type":    fuel_type_enc,
            "mileage_km":   mileage_km,
            "engine_cc":    engine_cc,
            "age":          age,
            "district":     district_enc,
            "has_trim":     has_trim,
        }
        inp_df = pd.DataFrame([row])[FEATURES]

        # Predict
        predicted_price = float(model.predict(inp_df)[0])
        predicted_price = max(500_000, predicted_price)  # floor sanity

        # Feature contributions (local explanation)
        contributions = {}
        for feat in FEATURES:
            contributions[feat] = round(feature_contribution(inp_df, feat))

        # Sort by absolute impact
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Confidence band (±10% as proxy)
        low  = predicted_price * 0.90
        high = predicted_price * 1.10

        # Human-readable labels for features
        FEAT_LABELS = {
            "brand": f"Brand ({brand})",
            "model": f"Model ({model_name})",
            "condition": f"Condition ({condition})",
            "transmission": f"Transmission ({transmission})",
            "body_type": f"Body Type ({body_type})",
            "fuel_type": f"Fuel Type ({fuel_type})",
            "mileage_km": f"Mileage ({mileage_km:,} km)",
            "engine_cc": f"Engine ({engine_cc} cc)",
            "age": f"Age ({age} years)",
            "district": f"District ({district})",
            "has_trim": f"Trim Info ({'Yes' if has_trim else 'No'})",
        }

        return jsonify({
            "success": True,
            "predicted_price": round(predicted_price),
            "price_low":  round(low),
            "price_high": round(high),
            "price_formatted": f"Rs {predicted_price:,.0f}",
            "contributions": [
                {
                    "feature":   feat,
                    "label":     FEAT_LABELS.get(feat, feat),
                    "value":     val,
                    "direction": "up" if val >= 0 else "down",
                    "formatted": f"Rs {abs(val):,.0f}",
                }
                for feat, val in sorted_contribs
            ],
            "inputs": {
                "brand": brand, "model": model_name,
                "year": year, "age": age,
                "condition": condition, "transmission": transmission,
                "body_type": body_type, "fuel_type": fuel_type,
                "mileage_km": mileage_km, "engine_cc": engine_cc,
                "district": district, "has_trim": has_trim,
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/users", methods=["GET"])
@role_required("admin")  # AUTHORIZATION: role claim must be admin -> 403 otherwise
def list_users():
    """Every registered account. Admin only.

    Contrast with /api/predict above: both require a valid token, but this one
    additionally checks *what* the caller is allowed to do. A normal user gets
    200 there and 403 here with the exact same token.
    """
    with get_session() as session:
        users = session.query(User).order_by(User.created_at.desc()).all()
        return jsonify({
            "count": len(users),
            "users": [u.to_dict() for u in users],   # to_dict omits password_hash
        })


@app.route("/api/whoami", methods=["GET"])
@jwt_required()
def whoami():
    """Small helper for the viva: shows identity and role straight from the token."""
    return jsonify({
        "user_id": get_jwt_identity(),
        "role": get_jwt().get("role"),
        "email": get_jwt().get("email"),
        "source": "read from the JWT itself - no database query",
    })


if __name__ == "__main__":
    # Local dev only. In Docker/production gunicorn imports `app` directly.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print("🚗  Vehicle Price Predictor — Flask API")
    print(f"    Running on http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
