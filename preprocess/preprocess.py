"""
preprocess.py — ikman.lk Car Dataset Preprocessing Pipeline
============================================================

Steps:
  1.  Drop junk rows (bad price string, bad condition, year < 1990)
  2.  Parse price → numeric (LKR integer)
  3.  Parse mileage → numeric (km integer)
  4.  Parse engine_capacity → numeric (cc integer)
       - Electric cars (fuel_type == Electric) → engine_cc = 0
       - Non-electric with implausible cc (<400) → drop row
  5.  Fix encoding issues (CoupÃ©/Sports → Coupé/Sports)
  6.  Clip outliers: price [500k–150M], mileage [0–500k], year [1990–2026]
  7.  Feature engineering:
       - age = 2026 - year
       - district extracted from location_raw
       - has_trim binary flag (1 if trim present, 0 if not)
  8.  Impute body_type nulls using mode per brand
  9.  Group rare brands (<10 listings) → "Other"
  10. Drop unused columns: trim, location_raw, description, url, scraped_at, title, year
  11. Encode categoricals with LabelEncoder (save encoders for inference)
  12. Save: ikman_cars_clean.csv + encoders.pkl + preprocessing_report.txt

Usage:
    pip install pandas numpy scikit-learn
    python preprocess.py
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder

# Config
INPUT_CSV      = "ikman_cars_raw.csv"
OUTPUT_CSV     = "ikman_cars_clean.csv"
ENCODERS_PKL   = "encoders.pkl"
REPORT_TXT     = "preprocessing_report.txt"
CURRENT_YEAR   = 2026

PRICE_MIN      = 500_000
PRICE_MAX      = 150_000_000
MILEAGE_MAX    = 500_000
YEAR_MIN       = 1990
YEAR_MAX       = 2026
RARE_BRAND_THR = 10          # brands with fewer listings → "Other"
ENGINE_CC_MIN  = 400         # below this, non-EV rows are dropped

CATEGORICAL_COLS = ["brand", "condition", "transmission",
                    "body_type", "fuel_type", "district"]

# Logging helper
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# ═════════════════════
# LOAD
# ═════════════════════
df = pd.read_csv(INPUT_CSV)
original_len = len(df)
log(f"\n{'='*60}")
log(f"IKMAN.LK CAR DATASET — PREPROCESSING REPORT")
log(f"{'='*60}")
log(f"\n[LOAD] {original_len} rows loaded from '{INPUT_CSV}'")
log(f"       Columns: {df.columns.tolist()}")


# ═══════════════════════════
# STEP 1 — Drop junk rows
# ═══════════════════════════
log(f"\n{'─'*60}")
log("STEP 1 — Drop junk rows")
log(f"{'─'*60}")

before = len(df)

# 1a. price_raw that is null or contains only year digits (e.g. "Rs 2017", "Rs 2026")
df['_price_digits'] = df['price_raw'].str.replace(r'[^\d]', '', regex=True)
junk_price_mask = (
    df['price_raw'].isna() |
    df['_price_digits'].astype(str).isin(['2017', '2018', '2019', '2020',
                                          '2021', '2022', '2023', '2024',
                                          '2025', '2026'])
)
df = df[~junk_price_mask].copy()
log(f"  Dropped {before - len(df)} rows with null/junk price_raw")

# 1b. Dirty condition value (scraped nav text)
before = len(df)
valid_conditions = ['Used', 'Brand New', 'Reconditioned', 'Import']
df = df[df['condition'].isin(valid_conditions)].copy()
log(f"  Dropped {before - len(df)} rows with invalid condition values")

# 1c. Year < 1990
before = len(df)
df = df[df['year'].notna()].copy()
df = df[df['year'] >= YEAR_MIN].copy()
log(f"  Dropped {before - len(df)} rows with year < {YEAR_MIN} or null year")

# 1d. Drop null rows for critical columns
before = len(df)
critical = ['brand', 'fuel_type', 'transmission', 'mileage', 'engine_capacity', 'price_raw']
df = df.dropna(subset=critical).copy()
log(f"  Dropped {before - len(df)} rows missing critical fields")

log(f"  → {len(df)} rows remain after Step 1")


# ════════════════════════════════════
# STEP 2 — Parse price → integer LKR
# ════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 2 — Parse price_raw → price (LKR integer)")
log(f"{'─'*60}")

df['price'] = df['price_raw'].str.replace(r'[^\d]', '', regex=True).astype(float).astype(int)
log(f"  price range: Rs {df['price'].min():,} – Rs {df['price'].max():,}")
log(f"  price mean:  Rs {df['price'].mean():,.0f}")


# ══════════════════════════════════════
# STEP 3 — Parse mileage → integer km
# ══════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 3 — Parse mileage → mileage_km (integer)")
log(f"{'─'*60}")

df['mileage_km'] = df['mileage'].str.replace(r'[^\d]', '', regex=True).astype(float).astype(int)
log(f"  mileage_km range: {df['mileage_km'].min():,} – {df['mileage_km'].max():,} km")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Parse engine_capacity → engine_cc (integer)
#          Electric cars → 0, non-electric implausible (<400cc) → drop
# ═══════════════════════════════════════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 4 — Parse engine_capacity → engine_cc")
log(f"{'─'*60}")

df['engine_cc'] = df['engine_capacity'].str.replace(r'[^\d]', '', regex=True).astype(float).astype(int)

# Electric vehicles: set engine_cc = 0 (they have no combustion engine)
ev_mask = df['fuel_type'] == 'Electric'
df.loc[ev_mask, 'engine_cc'] = 0
log(f"  Set engine_cc = 0 for {ev_mask.sum()} Electric vehicles")

# Non-electric with implausible cc → drop
before = len(df)
implausible_mask = (~ev_mask) & (df['engine_cc'] < ENGINE_CC_MIN)
df = df[~implausible_mask].copy()
log(f"  Dropped {before - len(df)} non-EV rows with engine_cc < {ENGINE_CC_MIN}cc")
log(f"  engine_cc range: {df['engine_cc'].min()} – {df['engine_cc'].max():,} cc")


# ═══════════════════════════════
# STEP 5 — Fix encoding issues
# ═══════════════════════════════
log(f"\n{'─'*60}")
log("STEP 5 — Fix encoding issues")
log(f"{'─'*60}")

df['body_type'] = df['body_type'].str.replace('CoupÃ©', 'Coupe', regex=False)
df['body_type'] = df['body_type'].str.replace('Coupé', 'Coupe', regex=False)
df['body_type'] = df['body_type'].str.replace('/', '-', regex=False)  # "SUV / 4x4" → "SUV - 4x4"
log(f"  Fixed: CoupÃ©/Sports → Coupe-Sports")
log(f"  Fixed: SUV / 4x4 → SUV - 4x4 (slash removed for compatibility)")
log(f"  Unique body_type values: {sorted(df['body_type'].dropna().unique().tolist())}")


# ══════════════════════════
# STEP 6 — Clip outliers
# ══════════════════════════
log(f"\n{'─'*60}")
log("STEP 6 — Clip outliers")
log(f"{'─'*60}")

before = len(df)

# Price
df = df[(df['price'] >= PRICE_MIN) & (df['price'] <= PRICE_MAX)].copy()
log(f"  Price filter [{PRICE_MIN:,} – {PRICE_MAX:,}]: dropped {before - len(df)} rows")

before = len(df)

# Mileage (0 is valid for brand new cars)
df = df[df['mileage_km'] <= MILEAGE_MAX].copy()
log(f"  Mileage filter [0 – {MILEAGE_MAX:,} km]: dropped {before - len(df)} rows")

before = len(df)

# Year (already filtered < 1990 in step 1; cap at 2026)
df = df[df['year'] <= YEAR_MAX].copy()
log(f"  Year filter [{YEAR_MIN} – {YEAR_MAX}]: dropped {before - len(df)} rows")

log(f"  → {len(df)} rows remain after Step 6")


# ════════════════════════════════
# STEP 7 — Feature engineering
# ════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 7 — Feature engineering")
log(f"{'─'*60}")

# 7a. age = current year - manufacture year
df['age'] = CURRENT_YEAR - df['year'].astype(int)
log(f"  Created 'age': range {df['age'].min()} – {df['age'].max()} years")

# 7b. district — extract last element from "Area, District"
df['district'] = df['location_raw'].str.split(',').str[-1].str.strip()
# Clean up "Sri Lanka" fallback
df['district'] = df['district'].replace('Sri Lanka', 'Unknown')
log(f"  Created 'district': {df['district'].nunique()} unique districts")
log(f"  Top districts: {df['district'].value_counts().head(5).to_dict()}")

# 7c. has_trim — binary flag (was trim info provided?)
df['has_trim'] = df['trim'].notna().astype(int)
log(f"  Created 'has_trim': {df['has_trim'].sum()} rows have trim info ({df['has_trim'].mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════════
# STEP 8 — Impute body_type nulls using mode per brand
# ═══════════════════════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 8 — Impute body_type nulls (mode per brand)")
log(f"{'─'*60}")

null_before = df['body_type'].isna().sum()

# Compute mode per brand
brand_body_mode = (
    df.dropna(subset=['body_type'])
    .groupby('brand')['body_type']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

# Fill nulls using brand mode
def fill_body_type(row):
    if pd.isna(row['body_type']):
        return brand_body_mode.get(row['brand'], np.nan)
    return row['body_type']

df['body_type'] = df.apply(fill_body_type, axis=1)

# Any remaining nulls → global mode
global_mode = df['body_type'].mode().iloc[0]
df['body_type'] = df['body_type'].fillna(global_mode)
null_after = df['body_type'].isna().sum()

log(f"  body_type nulls before: {null_before}")
log(f"  body_type nulls after:  {null_after}")
log(f"  Global fallback mode: '{global_mode}'")


# ═══════════════════════════════════════
# STEP 9 — Group rare brands → "Other"
# ═══════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 9 — Group rare brands (< {}) → 'Other'".format(RARE_BRAND_THR))
log(f"{'─'*60}")

brand_counts = df['brand'].value_counts()
rare_brands  = brand_counts[brand_counts < RARE_BRAND_THR].index.tolist()
df['brand']  = df['brand'].apply(lambda b: 'Other' if b in rare_brands else b)
log(f"  Grouped {len(rare_brands)} rare brands into 'Other': {rare_brands}")
log(f"  Remaining unique brands: {df['brand'].nunique()}")


# ═══════════════════════════════
# STEP 10 — Drop unused columns
# ═══════════════════════════════
log(f"\n{'─'*60}")
log("STEP 10 — Drop unused columns")
log(f"{'─'*60}")

drop_cols = ['trim', 'location_raw', 'description', 'url',
             'scraped_at', 'title', 'year',
             'price_raw', 'mileage', 'engine_capacity', '_price_digits']

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)
log(f"  Dropped columns: {drop_cols}")
log(f"  Remaining columns: {df.columns.tolist()}")


# ══════════════════════════════════════════════════
# STEP 11 — Encode categoricals with LabelEncoder
# ══════════════════════════════════════════════════
log(f"\n{'─'*60}")
log("STEP 11 — Label encode categorical columns")
log(f"{'─'*60}")

encoders = {}
for col in CATEGORICAL_COLS:
    if col not in df.columns:
        log(f"  SKIP '{col}' — not in dataframe")
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    log(f"  '{col}': {len(le.classes_)} classes → {list(le.classes_)[:8]}{'...' if len(le.classes_) > 8 else ''}")

# Save encoders for use during inference
with open(ENCODERS_PKL, 'wb') as f:
    pickle.dump(encoders, f)
log(f"\n  Encoders saved to '{ENCODERS_PKL}'")


# ════════════════════════════
# FINAL — Save and report
# ════════════════════════════
log(f"\n{'='*60}")
log("FINAL DATASET SUMMARY")
log(f"{'='*60}")

log(f"\n  Shape:     {df.shape[0]} rows × {df.shape[1]} columns")
log(f"  Columns:   {df.columns.tolist()}")
log(f"\n  Rows dropped: {original_len - len(df)} ({(original_len - len(df))/original_len*100:.1f}%)")
log(f"  Rows kept:    {len(df)} ({len(df)/original_len*100:.1f}%)")

log(f"\n  Null counts per column:")
for col, n in df.isnull().sum().items():
    log(f"    {col:20s}: {n}")

log(f"\n  Feature statistics:")
log(df[['price', 'mileage_km', 'engine_cc', 'age']].describe().to_string())

log(f"\n  Target (price) distribution:")
log(f"    Min:    Rs {df['price'].min():>15,}")
log(f"    Q1:     Rs {df['price'].quantile(0.25):>15,.0f}")
log(f"    Median: Rs {df['price'].median():>15,.0f}")
log(f"    Q3:     Rs {df['price'].quantile(0.75):>15,.0f}")
log(f"    Max:    Rs {df['price'].max():>15,}")
log(f"    Mean:   Rs {df['price'].mean():>15,.0f}")

df.to_csv(OUTPUT_CSV, index=False)
log(f"\n Clean dataset saved to '{OUTPUT_CSV}'")

with open(REPORT_TXT, 'w') as f:
    f.write('\n'.join(log_lines))
log(f" Report saved to '{REPORT_TXT}'")
