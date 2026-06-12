"""
MSME Merchant Churn Dataset Generator
======================================
Synthetic dataset inspired by and structurally derived from two public Kaggle datasets:

1. "Churn Modelling" by Shrutimechlearn
   URL: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
   Features borrowed: credit_score, tenure, balance, num_of_products,
                      has_credit_card, active_member, estimated_salary, churn logic

2. "E-Commerce Customer Churn Analysis and Prediction" by Ankitverma2010
   URL: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
   Features borrowed: complaint filed flag, satisfaction score, days since last order,
                      cashback amount patterns, transaction frequency concepts

The raw features from these datasets were re-parameterized and extended with
MSME-specific engineered features (transaction velocity, ticket size variance,
POS usage, working capital utilization, sector mapping) to simulate a realistic
small-business banking portfolio. All merchant IDs, business names, and
financial figures are fully synthetic.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

SECTORS = ["Retail Trader", "Textile & Apparel", "Agri-Allied", "Small Mfg", "Food & Beverages", "Tech Services", "Construction"]
SECTOR_WEIGHTS = [0.28, 0.18, 0.12, 0.15, 0.12, 0.08, 0.07]

# ── Base merchant attributes ──────────────────────────────────────────────────
merchant_id = [f"MRC{str(i).zfill(5)}" for i in range(1, N+1)]
sector = np.random.choice(SECTORS, size=N, p=SECTOR_WEIGHTS)
tenure_months = np.random.randint(1, 120, size=N)
credit_score = np.clip(np.random.normal(650, 80, N).astype(int), 300, 900)
num_products = np.random.choice([1, 2, 3, 4], size=N, p=[0.45, 0.35, 0.15, 0.05])
has_working_capital_loan = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
has_pos_machine = np.random.choice([0, 1], size=N, p=[0.3, 0.7])
estimated_annual_turnover = np.random.lognormal(mean=13.5, sigma=1.2, size=N)  # in INR
account_balance = estimated_annual_turnover * np.random.uniform(0.02, 0.15, N)

# ── Transaction features (90-day vs 30-day window) ────────────────────────────
txn_count_90d = np.random.poisson(lam=45, size=N).clip(1, 300)
# Distressed merchants have fewer recent transactions
distress_flag = np.random.binomial(1, 0.22, N)
txn_count_30d_expected = txn_count_90d / 3
txn_count_30d = np.where(
    distress_flag == 1,
    (txn_count_30d_expected * np.random.uniform(0.2, 0.55, N)).clip(0, 100).astype(int),
    (txn_count_30d_expected * np.random.uniform(0.85, 1.15, N)).clip(0, 100).astype(int)
)

# Transaction velocity drop (%) — key engineered feature
txn_velocity_drop_pct = ((txn_count_90d / 3 - txn_count_30d) / (txn_count_90d / 3 + 1e-9)) * 100
txn_velocity_drop_pct = txn_velocity_drop_pct.clip(-30, 100)

# Average ticket size
avg_ticket_size_90d = estimated_annual_turnover / (txn_count_90d * 4 + 1)
ticket_size_variance_pct = np.where(
    distress_flag == 1,
    np.random.uniform(30, 70, N),
    np.random.uniform(5, 25, N)
)

# Balance drop over last quarter (%)
balance_drop_pct = np.where(
    distress_flag == 1,
    np.random.uniform(15, 60, N),
    np.random.uniform(-10, 15, N)
)

# Working capital utilization (%)
wc_utilization_pct = np.where(
    has_working_capital_loan == 1,
    np.where(distress_flag == 1,
             np.random.uniform(75, 100, N),
             np.random.uniform(30, 75, N)),
    np.zeros(N)
)

# POS transaction count (monthly)
pos_txn_monthly = np.where(
    has_pos_machine == 1,
    np.where(distress_flag == 1,
             np.random.poisson(5, N).clip(0, 50),
             np.random.poisson(30, N).clip(1, 200)),
    np.zeros(N)
)

# Complaints filed (last 12 months)
complaints_12m = np.where(
    distress_flag == 1,
    np.random.poisson(2, N).clip(0, 8),
    np.random.poisson(0.4, N).clip(0, 5)
)

# Days since last transaction
days_since_last_txn = np.where(
    distress_flag == 1,
    np.random.randint(15, 90, N),
    np.random.randint(0, 14, N)
)

# Satisfaction score (1–5)
satisfaction_score = np.where(
    distress_flag == 1,
    np.random.choice([1, 2, 3], size=N, p=[0.4, 0.4, 0.2]),
    np.random.choice([3, 4, 5], size=N, p=[0.2, 0.45, 0.35])
)

# Cashback / reward usage (proxy for engagement)
cashback_utilization = np.where(
    distress_flag == 1,
    np.random.uniform(0, 0.2, N),
    np.random.uniform(0.3, 1.0, N)
)

# ── Churn label (business-rule + probabilistic noise) ─────────────────────────
churn_prob = (
    0.30 * (txn_velocity_drop_pct > 40).astype(float) +
    0.20 * (balance_drop_pct > 30).astype(float) +
    0.15 * (wc_utilization_pct > 85).astype(float) +
    0.10 * (complaints_12m >= 2).astype(float) +
    0.10 * (days_since_last_txn > 30).astype(float) +
    0.05 * (satisfaction_score <= 2).astype(float) +
    0.05 * (credit_score < 550).astype(float) +
    0.05 * distress_flag.astype(float)
)
churn_prob = churn_prob.clip(0, 1)
noise = np.random.uniform(-0.18, 0.18, N)
churn = (churn_prob + noise > 0.35).astype(int)

# Risk tier
def assign_risk(row):
    if row["churn"] == 1 or row["txn_velocity_drop_pct"] > 40:
        return "High"
    elif row["txn_velocity_drop_pct"] > 20 or row["balance_drop_pct"] > 20:
        return "Medium"
    return "Low"

df = pd.DataFrame({
    "merchant_id": merchant_id,
    "sector": sector,
    "tenure_months": tenure_months,
    "credit_score": credit_score,
    "num_products": num_products,
    "has_working_capital_loan": has_working_capital_loan,
    "has_pos_machine": has_pos_machine,
    "estimated_annual_turnover": estimated_annual_turnover.round(2),
    "account_balance": account_balance.round(2),
    "txn_count_90d": txn_count_90d,
    "txn_count_30d": txn_count_30d,
    "txn_velocity_drop_pct": txn_velocity_drop_pct.round(2),
    "avg_ticket_size_90d": avg_ticket_size_90d.round(2),
    "ticket_size_variance_pct": ticket_size_variance_pct.round(2),
    "balance_drop_pct": balance_drop_pct.round(2),
    "wc_utilization_pct": wc_utilization_pct.round(2),
    "pos_txn_monthly": pos_txn_monthly.astype(int),
    "complaints_12m": complaints_12m,
    "days_since_last_txn": days_since_last_txn,
    "satisfaction_score": satisfaction_score,
    "cashback_utilization": cashback_utilization.round(3),
    "churn": churn,
})

df["risk_tier"] = df.apply(assign_risk, axis=1)
df.to_csv("msme_merchant_churn.csv", index=False)
print(f"Dataset saved: {len(df)} rows | Churn rate: {df['churn'].mean()*100:.1f}%")
print(df.describe().round(2))