"""
MSME Merchant Churn Prediction — ML Training Pipeline
=======================================================
Model: XGBoost Gradient Boosting Classifier
Evaluation: ROC-AUC, F1, Precision, Recall, Confusion Matrix
Outputs: trained model, feature importance chart, predictions CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

# ── Palette ───────────────────────────────────────────────────────────────────
DARK = "#1a1a2e"
MID  = "#16213e"
ACC1 = "#0f3460"
ACC2 = "#e94560"
ACC3 = "#f5a623"
LIGHT = "#eaeaea"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": MID,
    "axes.edgecolor": LIGHT, "axes.labelcolor": LIGHT,
    "xtick.color": LIGHT, "ytick.color": LIGHT,
    "text.color": LIGHT, "grid.color": "#2a2a4a",
    "font.family": "DejaVu Sans",
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("msme_merchant_churn.csv")
print(f"Loaded {len(df)} records | Churn rate: {df['churn'].mean()*100:.1f}%")

# ── Feature engineering ───────────────────────────────────────────────────────
le = LabelEncoder()
df["sector_encoded"] = le.fit_transform(df["sector"])

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

X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Model ──────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, "xgb_churn_model.pkl")
joblib.dump(le, "sector_encoder.pkl")

# ── Figure 1: Feature Importance ─────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

FRIENDLY = {
    "txn_velocity_drop_pct": "Transaction Velocity Drop (%)",
    "days_since_last_txn": "Days Since Last Transaction",
    "balance_drop_pct": "Balance Drop (%)",
    "wc_utilization_pct": "Working Capital Utilization (%)",
    "ticket_size_variance_pct": "Ticket Size Variance (%)",
    "satisfaction_score": "Satisfaction Score",
    "complaints_12m": "Complaints Filed (12m)",
    "cashback_utilization": "Cashback Utilization",
    "credit_score": "Credit Score",
    "tenure_months": "Tenure (Months)",
    "pos_txn_monthly": "Monthly POS Transactions",
    "avg_ticket_size_90d": "Avg Ticket Size (90d)",
    "num_products": "No. of Products",
    "has_working_capital_loan": "Has WC Loan",
    "has_pos_machine": "Has POS Machine",
    "sector_encoded": "Sector",
}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(MID)

colors = [ACC2 if v == importance.iloc[-1] else ACC1 for v in importance.values]
bars = ax.barh([FRIENDLY.get(f, f) for f in importance.index], importance.values, color=colors, edgecolor="none", height=0.7)

# Annotate top feature
top_v = importance.iloc[-1]
ax.annotate(f"  {top_v*100:.1f}% weight",
            xy=(top_v, len(importance)-1), va="center", color=ACC2, fontsize=9, fontweight="bold")

ax.set_xlabel("Feature Importance (Gain)", fontsize=11, labelpad=10)
ax.set_title("XGBoost Feature Importance\nMSME Merchant Churn Prediction", fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(0, top_v * 1.35)
ax.tick_params(labelsize=9)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top","right","bottom","left"]].set_visible(False)

patch1 = mpatches.Patch(color=ACC2, label="Top predictor")
patch2 = mpatches.Patch(color=ACC1, label="Other features")
ax.legend(handles=[patch1, patch2], loc="lower right", framealpha=0.2, fontsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# ── Figure 2: ROC Curve ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(MID)
ax.plot(fpr, tpr, color=ACC2, lw=2.5, label=f"XGBoost (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], "--", color="#555577", lw=1.5, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color=ACC2)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: roc_curve.png")

# ── Figure 3: Churn rate by sector ────────────────────────────────────────────
sector_churn = df.groupby("sector")["churn"].agg(["mean","count"]).reset_index()
sector_churn.columns = ["sector","churn_rate","count"]
sector_churn = sector_churn.sort_values("churn_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(MID)
bar_colors = [ACC2 if r > 0.25 else ACC3 if r > 0.18 else ACC1 for r in sector_churn["churn_rate"]]
bars = ax.barh(sector_churn["sector"], sector_churn["churn_rate"]*100, color=bar_colors, height=0.6, edgecolor="none")
for bar, (_, row) in zip(bars, sector_churn.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{row['churn_rate']*100:.1f}%  (n={int(row['count'])})",
            va="center", fontsize=9, color=LIGHT)
ax.set_xlabel("Churn Rate (%)", fontsize=11)
ax.set_title("Churn Rate by Business Sector", fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(0, 45)
ax.grid(axis="x", alpha=0.25)
ax.spines[["top","right","bottom","left"]].set_visible(False)

p1 = mpatches.Patch(color=ACC2, label="High risk (>25%)")
p2 = mpatches.Patch(color=ACC3, label="Medium risk (18–25%)")
p3 = mpatches.Patch(color=ACC1, label="Low risk (<18%)")
ax.legend(handles=[p1,p2,p3], fontsize=9, framealpha=0.2, loc="lower right")
plt.tight_layout()
plt.savefig("sector_churn_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: sector_churn_rate.png")

# ── Figure 4: Confusion Matrix ────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(MID)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained","Churned"], yticklabels=["Retained","Churned"],
            linewidths=0.5, cbar=False, annot_kws={"size":14,"weight":"bold"})
ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
ax.set_ylabel("Actual", fontsize=11, labelpad=8)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")

# ── Prediction CSV (watchlist) ─────────────────────────────────────────────────
df_test = df.iloc[X_test.index].copy()
df_test["churn_probability"] = y_prob
df_test["predicted_churn"] = y_pred
watchlist = df_test[df_test["churn_probability"] > 0.6].sort_values("churn_probability", ascending=False)
watchlist[["merchant_id","sector","txn_velocity_drop_pct","balance_drop_pct",
           "wc_utilization_pct","complaints_12m","churn_probability","risk_tier"]].to_csv(
    "high_risk_watchlist.csv", index=False
)
print(f"\nHigh-risk watchlist: {len(watchlist)} merchants flagged")
print(f"\nAll outputs saved")