"""
train.py — XGBoost Model Training & Evaluation
================================================
Vehicle Price Prediction | ikman.lk Sri Lanka Dataset

Produces:
  model.pkl                    — trained XGBoost model
  model_feature_list.pkl       — ordered feature list for inference
  results/metrics_table.csv    — MAE, RMSE, R², MAPE for train/val/test
  results/plot_feature_imp.png — XGBoost feature importance bar chart
  results/plot_learning_curve.png  — R² learning curve
  results/plot_actual_vs_pred.png  — actual vs predicted scatter (test set)
  results/plot_residuals.png   — residual distribution (test set)
  results/training_report.txt  — full training log

Usage:
    pip install xgboost scikit-learn pandas numpy matplotlib
    python train.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV   = "ikman_cars_clean.csv"
OUT_DIR     = "results"
MODEL_PKL   = "model.pkl"
FEAT_PKL    = "model_feature_list.pkl"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 65)
log("  XGBOOST VEHICLE PRICE PREDICTION — TRAINING REPORT")
log("=" * 65)

df = pd.read_csv(INPUT_CSV)
log(f"\n[DATA] Loaded '{INPUT_CSV}': {df.shape[0]} rows × {df.shape[1]} cols")

# Encode 'model' column (still string after preprocessing)
le_model = LabelEncoder()
df["model"] = le_model.fit_transform(df["model"].astype(str))
log(f"       Encoded 'model': {le_model.classes_.shape[0]} unique car models")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "─" * 65)
log("STEP 1 — Train / Validation / Test Split")
log("─" * 65)

TARGET  = "price"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

# Step 1: split off 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# Step 2: split remaining 85% into 70% train + 15% val
#         0.15 / 0.85 ≈ 0.1765 of the temp set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

log(f"  Total samples : {len(X):,}")
log(f"  Train  (70%)  : {len(X_train):,} samples")
log(f"  Val    (15%)  : {len(X_val):,}  samples")
log(f"  Test   (15%)  : {len(X_test):,}  samples")
log(f"\n  Rationale:")
log(f"  • Train  — model learns weights and splits from this set")
log(f"  • Val    — monitors generalisation; drives early stopping")
log(f"  • Test   — held-out; never seen during training or tuning")
log(f"    random_state=42 ensures full reproducibility")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL & HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "─" * 65)
log("STEP 2 — XGBoost Hyperparameters")
log("─" * 65)

PARAMS = dict(
    n_estimators        = 500,    # max trees; early stopping prevents over-growing
    learning_rate       = 0.05,   # shrinkage: slow learning → better generalisation
    max_depth           = 6,      # tree depth; 6 captures complex interactions
    subsample           = 0.8,    # 80% row sampling per tree → reduces variance
    colsample_bytree    = 0.8,    # 80% feature sampling per tree → reduces correlation
    min_child_weight    = 5,      # min samples in leaf → prevents overfitting on noise
    reg_alpha           = 0.1,    # L1 regularisation → sparse feature selection
    reg_lambda          = 1.0,    # L2 regularisation → weight shrinkage
    random_state        = 42,
    n_jobs              = -1,     # use all CPU cores
)

param_justifications = {
    "n_estimators=500"       : "Enough trees for the model to converge; early stopping halts before overfitting",
    "learning_rate=0.05"     : "Small step size forces more trees but yields better generalisation than 0.1–0.3",
    "max_depth=6"            : "Captures brand × age × condition interactions without memorising noise",
    "subsample=0.8"          : "Stochastic boosting — each tree sees 80% of rows, reducing variance",
    "colsample_bytree=0.8"   : "Each tree uses 80% of features, decorrelating individual trees",
    "min_child_weight=5"     : "Prevents leaf nodes forming on fewer than 5 samples (noise guard)",
    "reg_alpha=0.1"          : "L1 penalty drives low-importance feature weights toward zero",
    "reg_lambda=1.0"         : "L2 penalty smooths weights, equivalent to Ridge regularisation",
    "early_stopping_rounds"  : "Training halts if validation RMSE doesn't improve for 50 rounds",
}

log("  Hyperparameter          Value     Justification")
log("  " + "─" * 80)
for k, v in param_justifications.items():
    log(f"  {k:<28s} {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "─" * 65)
log("STEP 3 — Model Training")
log("─" * 65)

model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=50, eval_metric="rmse")

eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

best_iter = model.best_iteration
log(f"  Best iteration (early stopping): {best_iter}")
log(f"  Trees actually used: {best_iter} of {PARAMS['n_estimators']} max")

# Save model and feature list
with open(MODEL_PKL, "wb") as f:
    pickle.dump(model, f)
with open(FEAT_PKL, "wb") as f:
    pickle.dump(FEATURES, f)
log(f"  Model saved → '{MODEL_PKL}'")
log(f"  Feature list saved → '{FEAT_PKL}'")

# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "─" * 65)
log("STEP 4 — Performance Metrics")
log("─" * 65)
log("""
  Metrics used (regression task — NOT classification):
  ┌────────────────────────────────────────────────────────────────────┐
  │ MAE  (Mean Absolute Error)   — average LKR error per prediction   │
  │ RMSE (Root Mean Square Error)— penalises large errors more heavily │
  │ R²   (Coefficient of Det.)   — % variance in price explained       │
  │ MAPE (Mean Abs % Error)      — error as % of actual price          │
  └────────────────────────────────────────────────────────────────────┘
  Note: Accuracy/F1/AUC are classification metrics — not applicable here.
""")

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                          / np.array(y_true))) * 100
    return mae, rmse, r2, mape

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)
y_test_pred  = model.predict(X_test)

results = {}
for split, y_true, y_pred in [
    ("Train",      y_train, y_train_pred),
    ("Validation", y_val,   y_val_pred),
    ("Test",       y_test,  y_test_pred),
]:
    mae, rmse, r2, mape = compute_metrics(y_true, y_pred)
    results[split] = {"MAE (Rs)": mae, "RMSE (Rs)": rmse, "R²": r2, "MAPE (%)": mape}

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(f"{OUT_DIR}/metrics_table.csv")

log("  ┌─────────────┬──────────────────┬──────────────────┬────────┬──────────┐")
log("  │ Split       │ MAE (Rs)         │ RMSE (Rs)        │ R²     │ MAPE (%) │")
log("  ├─────────────┼──────────────────┼──────────────────┼────────┼──────────┤")
for split, m in results.items():
    log(f"  │ {split:<11s} │ {m['MAE (Rs)']:>16,.0f} │ {m['RMSE (Rs)']:>16,.0f} │ "
        f"{m['R²']:>6.4f} │ {m['MAPE (%)']:>8.2f} │")
log("  └─────────────┴──────────────────┴──────────────────┴────────┴──────────┘")

log(f"\n  Interpretation:")
log(f"  • R² = {results['Test']['R²']:.4f} means the model explains "
    f"{results['Test']['R²']*100:.1f}% of price variance on unseen data")
log(f"  • Test MAE = Rs {results['Test']['MAE (Rs)']:,.0f} — "
    f"average prediction is off by this amount")
log(f"  • Train R² ({results['Train']['R²']:.4f}) vs Test R² ({results['Test']['R²']:.4f}) "
    f"— gap of {(results['Train']['R²']-results['Test']['R²']):.4f} indicates "
    + ("minimal overfitting" if results['Train']['R²']-results['Test']['R²'] < 0.05
       else "some overfitting — consider tuning regularisation"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "─" * 65)
log("STEP 5 — Generating Plots")
log("─" * 65)

COLORS = {
    "primary"   : "#2563EB",   # blue
    "secondary" : "#10B981",   # green
    "accent"    : "#F59E0B",   # amber
    "danger"    : "#EF4444",   # red
    "bg"        : "#F8FAFC",
    "grid"      : "#E2E8F0",
}

rs_fmt = FuncFormatter(lambda x, _: f"Rs {x/1e6:.0f}M")

# ── Plot 1: Feature Importance ────────────────────────────────────────────────
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

colors = [COLORS["primary"] if v == fi.max() else
          COLORS["secondary"] if v >= fi.quantile(0.75) else
          "#94A3B8" for v in fi.values]

bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="white",
               linewidth=0.5, height=0.65)

for bar, val in zip(bars, fi.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left",
            fontsize=9, color="#374151", fontweight="bold")

ax.set_xlabel("Feature Importance Score (Gain)", fontsize=11, color="#374151")
ax.set_title("XGBoost Feature Importance\nVehicle Price Prediction — ikman.lk",
             fontsize=13, fontweight="bold", color="#111827", pad=15)
ax.tick_params(colors="#374151", labelsize=10)
ax.set_xlim(0, fi.max() * 1.18)
ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
ax.spines[["top","right","bottom"]].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["primary"],   label="Highest importance"),
    Patch(facecolor=COLORS["secondary"], label="High importance (top 25%)"),
    Patch(facecolor="#94A3B8",           label="Moderate importance"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          framealpha=0.8, edgecolor=COLORS["grid"])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_feature_imp.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"  Saved: plot_feature_imp.png")

# ── Plot 2: Learning Curve ────────────────────────────────────────────────────
train_sizes  = np.linspace(0.1, 1.0, 10)
lc_train_r2  = []
lc_val_r2    = []
lc_train_mae = []
lc_val_mae   = []

for frac in train_sizes:
    n = max(int(frac * len(X_train)), 30)
    m = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    )
    m.fit(X_train.iloc[:n], y_train.iloc[:n], verbose=False)
    lc_train_r2.append(r2_score(y_train.iloc[:n], m.predict(X_train.iloc[:n])))
    lc_val_r2.append(r2_score(y_val, m.predict(X_val)))
    lc_train_mae.append(mean_absolute_error(y_train.iloc[:n], m.predict(X_train.iloc[:n])))
    lc_val_mae.append(mean_absolute_error(y_val, m.predict(X_val)))

n_samples = (train_sizes * len(X_train)).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(COLORS["bg"])

for ax in (ax1, ax2):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)

ax1.plot(n_samples, lc_train_r2, "o-", color=COLORS["primary"],
         lw=2, ms=6, label="Train R²")
ax1.plot(n_samples, lc_val_r2, "s--", color=COLORS["danger"],
         lw=2, ms=6, label="Validation R²")
ax1.fill_between(n_samples, lc_train_r2, lc_val_r2,
                 alpha=0.08, color=COLORS["danger"], label="Generalisation gap")
ax1.set_xlabel("Training Samples", fontsize=11, color="#374151")
ax1.set_ylabel("R² Score", fontsize=11, color="#374151")
ax1.set_title("Learning Curve — R²", fontsize=12, fontweight="bold", color="#111827")
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.tick_params(colors="#374151")

ax2.plot(n_samples, [m/1e6 for m in lc_train_mae], "o-", color=COLORS["primary"],
         lw=2, ms=6, label="Train MAE")
ax2.plot(n_samples, [m/1e6 for m in lc_val_mae], "s--", color=COLORS["danger"],
         lw=2, ms=6, label="Validation MAE")
ax2.set_xlabel("Training Samples", fontsize=11, color="#374151")
ax2.set_ylabel("MAE (Rs Millions)", fontsize=11, color="#374151")
ax2.set_title("Learning Curve — MAE", fontsize=12, fontweight="bold", color="#111827")
ax2.legend(fontsize=10)
ax2.tick_params(colors="#374151")

plt.suptitle("XGBoost Learning Curves — ikman.lk Vehicle Price Prediction",
             fontsize=13, fontweight="bold", color="#111827", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"  Saved: plot_learning_curve.png")

# ── Plot 3: Actual vs Predicted (Test Set) ────────────────────────────────────
y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_test_pred)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

sc = ax.scatter(y_test_arr/1e6, y_pred_arr/1e6,
                alpha=0.55, s=35, color=COLORS["primary"],
                edgecolors="white", linewidths=0.3)

lim_min = min(y_test_arr.min(), y_pred_arr.min()) / 1e6 * 0.95
lim_max = max(y_test_arr.max(), y_pred_arr.max()) / 1e6 * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max],
        "r--", lw=1.8, label="Perfect prediction (y=x)")

# Annotate R² on chart
r2_test = results["Test"]["R²"]
mae_test = results["Test"]["MAE (Rs)"]
ax.text(0.05, 0.92, f"R² = {r2_test:.4f}\nMAE = Rs {mae_test/1e6:.2f}M",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["grid"], alpha=0.9))

ax.set_xlabel("Actual Price (Rs Millions)", fontsize=12, color="#374151")
ax.set_ylabel("Predicted Price (Rs Millions)", fontsize=12, color="#374151")
ax.set_title("Actual vs Predicted Price — Test Set\nXGBoost | ikman.lk Sri Lanka",
             fontsize=13, fontweight="bold", color="#111827", pad=12)
ax.legend(fontsize=10)
ax.grid(color=COLORS["grid"], linewidth=0.7)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(colors="#374151")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"  Saved: plot_actual_vs_pred.png")

# ── Plot 4: Residual Distribution ─────────────────────────────────────────────
residuals = y_test_arr - y_pred_arr

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(COLORS["bg"])

# Histogram
ax = axes[0]
ax.set_facecolor(COLORS["bg"])
ax.hist(residuals/1e6, bins=40, color=COLORS["primary"],
        edgecolor="white", linewidth=0.4, alpha=0.85)
ax.axvline(0, color=COLORS["danger"], lw=2, linestyle="--", label="Zero error")
ax.axvline(np.mean(residuals)/1e6, color=COLORS["accent"], lw=1.5,
           linestyle="-.", label=f"Mean residual: Rs {np.mean(residuals)/1e6:.2f}M")
ax.set_xlabel("Residual (Rs Millions)", fontsize=11, color="#374151")
ax.set_ylabel("Count", fontsize=11, color="#374151")
ax.set_title("Residual Distribution (Test Set)", fontsize=12,
             fontweight="bold", color="#111827")
ax.legend(fontsize=10)
ax.grid(color=COLORS["grid"], linewidth=0.7)
ax.spines[["top","right"]].set_visible(False)

# Residuals vs Predicted
ax = axes[1]
ax.set_facecolor(COLORS["bg"])
ax.scatter(y_pred_arr/1e6, residuals/1e6, alpha=0.5, s=30,
           color=COLORS["secondary"], edgecolors="white", linewidths=0.3)
ax.axhline(0, color=COLORS["danger"], lw=2, linestyle="--")
ax.axhline(np.std(residuals)/1e6, color=COLORS["accent"], lw=1.2,
           linestyle=":", label=f"+1 SD: Rs {np.std(residuals)/1e6:.2f}M")
ax.axhline(-np.std(residuals)/1e6, color=COLORS["accent"], lw=1.2, linestyle=":")
ax.set_xlabel("Predicted Price (Rs Millions)", fontsize=11, color="#374151")
ax.set_ylabel("Residual (Rs Millions)", fontsize=11, color="#374151")
ax.set_title("Residuals vs Predicted Values", fontsize=12,
             fontweight="bold", color="#111827")
ax.legend(fontsize=10)
ax.grid(color=COLORS["grid"], linewidth=0.7)
ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Residual Analysis — Test Set | XGBoost",
             fontsize=13, fontweight="bold", color="#111827", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"  Saved: plot_residuals.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("TRAINING COMPLETE — SUMMARY")
log("=" * 65)
log(f"\n  Dataset         : {len(df):,} rows × {len(FEATURES)} features")
log(f"  Train/Val/Test  : 70% / 15% / 15%")
log(f"  Best iteration  : {best_iter} trees")
log()
log(f"  {'Split':<12} {'MAE (Rs)':>15} {'RMSE (Rs)':>16} {'R²':>8} {'MAPE':>9}")
log(f"  {'─'*12} {'─'*15} {'─'*16} {'─'*8} {'─'*9}")
for split, m in results.items():
    log(f"  {split:<12} {m['MAE (Rs)']:>15,.0f} {m['RMSE (Rs)']:>16,.0f} "
        f"{m['R²']:>8.4f} {m['MAPE (%)']:>8.2f}%")

log(f"""
  Result Interpretation:
  ───────────────────────────────────────────────────────────────
  • R² (Test) = {results['Test']['R²']:.4f}
    The model explains {results['Test']['R²']*100:.1f}% of price variance on
    completely unseen data. For a real-world used car market with
    high negotiation variance, this is a strong result.

  • MAE (Test) = Rs {results['Test']['MAE (Rs)']:,.0f}
    On average, predictions are off by Rs {results['Test']['MAE (Rs)']/1e6:.2f}M
    on a median car price of Rs {float(y.median())/1e6:.2f}M.

  • RMSE > MAE indicates some large errors on outlier-priced
    vehicles (luxury/rare models), which is expected.

  • MAPE (Test) = {results['Test']['MAPE (%)']:.2f}%
    On average, predictions are within {results['Test']['MAPE (%)']:.1f}% of actual price.

  • Overfitting check: Train R² - Test R² =
    {results['Train']['R²']:.4f} - {results['Test']['R²']:.4f} = {results['Train']['R²']-results['Test']['R²']:.4f}
    {"✓ Minimal overfitting — regularisation is working well."
     if results['Train']['R²']-results['Test']['R²'] < 0.07
     else "⚠ Some overfitting — consider reducing max_depth or increasing reg_lambda."}

  Outputs saved:
  • model.pkl                       — trained model for inference
  • model_feature_list.pkl          — feature order for Flask backend
  • results/metrics_table.csv       — metrics for all splits
  • results/plot_feature_imp.png    — feature importance chart
  • results/plot_learning_curve.png — learning curves (R² and MAE)
  • results/plot_actual_vs_pred.png — actual vs predicted scatter
  • results/plot_residuals.png      — residual distribution & spread
""")

with open(f"{OUT_DIR}/training_report.txt", "w") as f:
    f.write("\n".join(log_lines))
log(f"  Report saved → results/training_report.txt")
log("\n DONE")
