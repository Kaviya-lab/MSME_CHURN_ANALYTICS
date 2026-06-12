
import os, json
import numpy as np
import pandas as pd
import joblib, shap

os.makedirs("models", exist_ok=True)

# ── Load model + data ─────────────────────────────────────────────────────────
model = joblib.load("xgb_churn_model.pkl")
le    = joblib.load("sector_encoder.pkl")
df    = pd.read_csv("msme_merchant_churn.csv")

FEATURES = [
    "credit_score", "tenure_months", "num_products",
    "has_working_capital_loan", "has_pos_machine",
    "txn_velocity_drop_pct", "avg_ticket_size_90d",
    "ticket_size_variance_pct", "balance_drop_pct",
    "wc_utilization_pct", "pos_txn_monthly",
    "complaints_12m", "days_since_last_txn",
    "satisfaction_score", "cashback_utilization",
    "sector_encoded",
]

df["sector_encoded"] = le.transform(df["sector"])
X = df[FEATURES]

# ── Sample 500 rows (faster SHAP, still representative) ───────────────────────
SAMPLE_SIZE = 500
sample_idx  = np.random.choice(len(X), size=SAMPLE_SIZE, replace=False)
X_sample    = X.iloc[sample_idx].reset_index(drop=True)

# ── Compute SHAP values ───────────────────────────────────────────────────────
explainer  = shap.TreeExplainer(model)
shap_array = explainer.shap_values(X_sample)   # shape: (500, 16)

# ── Global importance: mean |SHAP| per feature ────────────────────────────────
shap_global = {
    feat: float(np.mean(np.abs(shap_array[:, i])))
    for i, feat in enumerate(FEATURES)
}
# Add churn_probability to CSV so app.py can read it
df["churn_probability"] = model.predict_proba(X)[:, 1]
df.to_csv("msme_merchant_churn.csv", index=False)
# ── Save artifacts ────────────────────────────────────────────────────────────
np.save("models/shap_values.npy", shap_array)
X_sample.to_csv("models/shap_sample.csv", index=False)
with open("models/shap_global.json", "w") as f:
    json.dump(shap_global, f, indent=2)

print("✓ models/shap_values.npy",  shap_array.shape)
print("✓ models/shap_sample.csv",  X_sample.shape)
print("✓ models/shap_global.json", len(shap_global), "features")


