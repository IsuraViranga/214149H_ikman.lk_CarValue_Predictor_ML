# 🚗 CarValueLK — Explainable Vehicle Price Predictor
### An Explainable Machine Learning Approach for Used Car Price Prediction in Sri Lanka

> **Index:** 214149H | **Name:** Perera M.I.V.

---

## 📌 Project Overview

This project builds an explainable machine learning model to predict used vehicle market prices in Sri Lanka using listing data scraped from **ikman.lk**. The system includes a full web application allowing users to input vehicle details and receive price predictions with feature-level explanations.

---

## 📁 Folder Structure

```
Desktop/data/new/
├── Scrapper/
│   ├── scraper.py                 ← Web scraper (ikman.lk)
│   ├── ikman_cars_raw.csv         ← Raw scraped data (output)
│   └── scraper.log                ← Scrape run log
│
├── dataFiles/
│   ├── ikman_cars_raw.csv         ← Raw data copy
│   └── ikman_cars_clean_check.csv ← Cleaned data reference
│
├── preprocess/
│   ├── preprocess.py              ← Preprocessing pipeline
│   ├── ikman_cars_raw.csv         ← Raw scraped data
│   ├── ikman_cars_clean.csv       ← Cleaned dataset (output)
│   └── encoders.pkl               ← Label encoders (output)
│
├── train/
│   ├── train.py                   ← XGBoost training script
│   ├── ikman_cars_clean.csv       ← Copy from preprocess/
│   ├── model.pkl                  ← Trained model (output)
│   ├── model_feature_list.pkl     ← Feature list (output)
│   └── results/
│       ├── metrics_table.csv
│       ├── plot_feature_imp.png
│       ├── plot_learning_curve.png
│       ├── plot_actual_vs_pred.png
│       └── plot_residuals.png
│
├── explain/
│   ├── explain.py                 ← XAI explainability script
│   ├── ikman_cars_clean.csv       ← Copy from preprocess/
│   └── results/
│       ├── plot_shap_summary.png
│       ├── plot_shap_waterfall.png
│       ├── plot_permutation_imp.png
│       ├── plot_pdp_grid.png
│       └── plot_local_explain.png
│
├── app/
│   ├── backend/
│   │   ├── app.py                 ← Flask REST API
│   │   ├── requirements.txt
│   │   ├── model.pkl              ← Copy from train/
│   │   ├── encoders.pkl           ← Copy from preprocess/
│   │   └── model_feature_list.pkl ← Copy from train/
│   └── frontend/
│       ├── package.json
│       ├── public/
│       │   └── index.html
│       └── src/
│           ├── index.js
│           └── App.jsx            ← React UI
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Collection | Python, Requests, BeautifulSoup |
| Preprocessing | Pandas, NumPy, Scikit-learn |
| ML Model | XGBoost |
| Explainability | SHAP, Permutation Importance, PDP |
| Backend | Flask, Flask-CORS |
| Frontend | React |

---

## ⚙️ How to Run

### Step 1 — Preprocessing
```bash
cd Desktop/data/preprocess
pip install pandas numpy scikit-learn
python preprocess.py
```
Outputs: `ikman_cars_clean.csv`, `encoders.pkl`

---

### Step 2 — Model Training
```bash
cd Desktop/data/train
pip install xgboost scikit-learn matplotlib
python train.py
```
Outputs: `model.pkl`, `model_feature_list.pkl`, `results/` plots

---

### Step 3 — Explainability
```bash
cd Desktop/data/explain
pip install shap
python explain.py
```
Outputs: SHAP and PDP plots in `results/`

---

### Step 4 — Run the Web App

**Copy these files into `app/backend/` first:**
- `model.pkl` ← from `train/`
- `encoders.pkl` ← from `preprocess/`
- `model_feature_list.pkl` ← from `train/`

**Terminal 1 — Backend:**
```bash
cd Desktop/data/app/backend
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

**Terminal 2 — Frontend:**
```bash
cd Desktop/data/app/frontend
npm install
npm start
# Opens on http://localhost:3000
```

---

## 📊 Model Performance

| Split | MAE (Rs) | RMSE (Rs) | R² | MAPE |
|---|---|---|---|---|
| Train | 2,100,226 | 4,405,335 | 0.9342 | 15.68% |
| Validation | 2,536,258 | 5,694,855 | 0.8817 | 21.28% |
| Test | 2,809,395 | 5,718,479 | **0.8997** | 16.64% |

> The model explains **90% of price variance** on completely unseen test data.

---

## 🔍 Explainability Methods

| Method | Purpose |
|---|---|
| SHAP (TreeExplainer) | Global + local feature attribution |
| SHAP Waterfall | Single prediction step-by-step breakdown |
| Permutation Importance | Model-agnostic feature ranking |
| Partial Dependence Plots | Isolated feature effect on price |
| Local Contributions | Why a specific car was priced this way |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/options` | Returns all dropdown values |
| POST | `/api/predict` | Returns predicted price + explanations |

---

## 📋 Dataset Summary

| Property | Value |
|---|---|
| Source | ikman.lk (scraped Feb 2026) |
| Total Records | 2,609 |
| Features | 11 |
| Target | Price (LKR) |
| Brands | 24 |
| Districts | 21 |

---

## 📚 References

- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*
- ikman.lk — https://ikman.lk
- XGBoost Docs — https://xgboost.readthedocs.io
- Scikit-learn Docs — https://scikit-learn.org
