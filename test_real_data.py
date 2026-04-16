"""Test the MoE 2x1 model on 10,000 real convertible bond observations."""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing.feature_engineering import engineer_features
from architectures.moe_pricer import MoEPricer

SEED = 42
N_SAMPLES = 1_000
MODEL_PATH = "models/moe_2x1_var_bw4.0.pt"
SCALER_X_PATH = "models/scaler_X.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"
DATA_PATH = "real_cb_raw.csv"

# Feature order must match training
FEATURE_COLS = [
    "S", "r", "q", "bs_vol", "credit_spread", "coupon_rate", "frequency",
    "maturity_years", "conversion_ratio", "conversion_price",
    "log_moneyness", "total_vol", "income_advantage", "risky_discount_rate",
    "conversion_premium", "spread_to_vol_ratio", "sqrt_maturity", "parity",
    "total_remaining_income", "real_rate", "credit_vol_product",
    "income_to_optionality", "rate_spread_ratio", "frequency_per_year",
]

# ── Load data ────────────────────────────────────────────────────────────
print("Loading real data...")
raw = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(raw):,}")

# Sample 10k
np.random.seed(SEED)
sample = raw.sample(n=N_SAMPLES, random_state=SEED).copy()

# Report complexity distribution
print(f"\nComplexity flags in sample:")
flags = sample["complexity_flags"].fillna("VANILLA")
print(flags.value_counts().to_string())

# ── Unit conversions ─────────────────────────────────────────────────────
sample["r"] = sample["r_pct"] / 100
sample["q"] = sample["q_pct"] / 100
sample["bs_vol"] = sample["sigma_pct"] / 100
sample["credit_spread"] = sample["credit_spread_bps"] / 10_000
sample["maturity_years"] = sample["maturity"]  # already in years
sample["price_convertible"] = sample["price"]
sample["frequency"] = 2  # semiannual (standard for US convertibles)

# Real data has CR per $1000 bond; training data uses CR per $100 face.
# Evidence: real CP = 1000/CR (e.g. 1000/28.6 ≈ 34.95), training CP = 100/CR.
sample["conversion_ratio"] = sample["conversion_ratio"] / 10
sample["conversion_price"] = sample["redemption"] / sample["conversion_ratio"]

# ── Feature engineering ──────────────────────────────────────────────────
print("\nEngineering features...")
df = engineer_features(sample, tiers=[1, 2, 3])

# Target
y_true_price = df["price_convertible"].values

# Check for any issues
X = df[FEATURE_COLS]
print(f"Feature matrix: {X.shape}")
nan_counts = X.isnull().sum()
if nan_counts.any():
    print(f"WARNING — NaN features:\n{nan_counts[nan_counts > 0]}")
    # Drop rows with NaN
    valid_mask = ~X.isnull().any(axis=1)
    X = X[valid_mask]
    y_true_price = y_true_price[valid_mask.values]
    df = df[valid_mask]
    print(f"After dropping NaN: {len(X):,} rows")

# ── Load scalers and model ───────────────────────────────────────────────
print("\nLoading scalers and model...")
with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)
with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)

pricer = MoEPricer.load(MODEL_PATH, device="cpu")
pricer.expert_pool.eval()
print(f"Model loaded: {MODEL_PATH}")

# ── Inference ────────────────────────────────────────────────────────────
print("\nRunning inference...")
X_raw = torch.tensor(X.values, dtype=torch.float32)
X_scaled = torch.tensor(scaler_X.transform(X), dtype=torch.float32)

with torch.no_grad():
    m_bins, t_bins = pricer.gating.route(X_raw)
    preds_sc = pricer.expert_pool.forward_routed(X_scaled, m_bins, t_bins).numpy()

# Inverse transform: scaler_y → expm1
preds_norm = scaler_y.inverse_transform(preds_sc).ravel()
preds_price = np.expm1(preds_norm)

# ── Results ──────────────────────────────────────────────────────────────
errors = preds_price - y_true_price
abs_errors = np.abs(errors)
pct_errors = abs_errors / np.abs(y_true_price) * 100

print("\n" + "=" * 60)
print(f"  MoE 2x1 (bw=2.0) on {len(X):,} real convertible bonds")
print("=" * 60)

print(f"\n{'Metric':<25} {'Value':>12}")
print("-" * 38)
print(f"{'MAE ($)':<25} {abs_errors.mean():>12.2f}")
print(f"{'Median AE ($)':<25} {np.median(abs_errors):>12.2f}")
print(f"{'RMSE ($)':<25} {np.sqrt((errors**2).mean()):>12.2f}")
print(f"{'Mean Error ($)':<25} {errors.mean():>12.2f}")
print(f"{'MAPE (%)':<25} {pct_errors.mean():>12.2f}")
print(f"{'Median APE (%)':<25} {np.median(pct_errors):>12.2f}")
print(f"{'95th pct AE ($)':<25} {np.percentile(abs_errors, 95):>12.2f}")
print(f"{'99th pct AE ($)':<25} {np.percentile(abs_errors, 99):>12.2f}")
print(f"{'Max AE ($)':<25} {abs_errors.max():>12.2f}")

# Price range context
print(f"\n{'Price range (true)':<25} ${y_true_price.min():.2f} – ${y_true_price.max():.2f}")
print(f"{'Price mean (true)':<25} ${y_true_price.mean():.2f}")
print(f"{'Price mean (pred)':<25} ${preds_price.mean():.2f}")

# Breakdown by complexity
print("\n── Error by Complexity ─────────────────────────────────────")
analysis = pd.DataFrame({
    "abs_error": abs_errors,
    "pct_error": pct_errors,
    "complexity": df["complexity_flags"].fillna("VANILLA").values,
})
grp = analysis.groupby("complexity").agg(
    count=("abs_error", "size"),
    MAE=("abs_error", "mean"),
    MedAE=("abs_error", "median"),
    MAPE=("pct_error", "mean"),
).sort_values("count", ascending=False)
print(grp.to_string())

# Breakdown by moneyness
print("\n── Error by Moneyness ──────────────────────────────────────")
analysis["log_m"] = df["log_moneyness"].values
analysis["moneyness"] = pd.cut(
    analysis["log_m"],
    bins=[-np.inf, -0.5, 0, 0.5, 1.0, np.inf],
    labels=["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"],
)
mon_grp = analysis.groupby("moneyness", observed=True).agg(
    count=("abs_error", "size"),
    MAE=("abs_error", "mean"),
    MedAE=("abs_error", "median"),
    MAPE=("pct_error", "mean"),
)
print(mon_grp.to_string())

# Breakdown by maturity
print("\n── Error by Maturity ───────────────────────────────────────")
analysis["T"] = df["maturity_years"].values
analysis["maturity_bucket"] = pd.cut(
    analysis["T"],
    bins=[0, 1, 3, 5, 10, np.inf],
    labels=["<1y", "1-3y", "3-5y", "5-10y", ">10y"],
)
mat_grp = analysis.groupby("maturity_bucket", observed=True).agg(
    count=("abs_error", "size"),
    MAE=("abs_error", "mean"),
    MedAE=("abs_error", "median"),
    MAPE=("pct_error", "mean"),
)
print(mat_grp.to_string())

# Routing distribution
print("\n── Gating Routing ─────────────────────────────────────────")
for i in range(pricer.gating.M):
    mask = (m_bins == i)
    n = mask.sum().item()
    print(f"  Expert {i} ({pricer.gating.describe_cell(i, 0)['regime']}): "
          f"{n:,} samples ({n/len(X)*100:.1f}%)")

print("\nDone.")
