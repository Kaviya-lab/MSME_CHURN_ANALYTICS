# MSME Merchant Churn Analytics & Early Warning System

An Explainable AI-powered banking analytics platform designed to identify MSME merchants at risk of churn and enable proactive retention strategies through predictive modeling, risk monitoring, and actionable business insights.

## Project Overview

Banks rely heavily on MSME merchants for transaction fees, current account balances, POS machine usage, and working capital lending revenue.

When merchants reduce account activity, stop using POS terminals, or migrate to competing banks, financial institutions lose valuable revenue opportunities.

This project predicts merchant churn using machine learning and provides explainable insights through SHAP (SHapley Additive exPlanations), enabling relationship managers to understand *why* a merchant is at risk and take corrective actions.

---

## Key Objectives

* Predict MSME merchant churn using machine learning.
* Detect early warning signals from transaction behavior.
* Generate high-risk merchant watchlists.
* Explain model predictions using SHAP Explainable AI.
* Provide actionable insights for relationship managers.
* Visualize portfolio health through an interactive dashboard.

---

## System Architecture

### Data Layer

Synthetic MSME banking dataset containing:

* Merchant profile information
* Transaction behavior metrics
* Working capital utilization
* Customer engagement indicators
* Merchant sector classifications

### Feature Engineering

Custom banking-focused features include:

* Transaction Velocity Drop (%)
* Average Ticket Size Variance (%)
* Balance Drop (%)
* Working Capital Utilization (%)
* Days Since Last Transaction
* Cashback Utilization
* Complaint Frequency
* POS Transaction Activity

### Machine Learning

**Model:** XGBoost Classifier

Used to predict merchant churn probability based on behavioral and financial indicators.

### Explainable AI

Integrated SHAP for:

* Global Feature Importance
* SHAP Beeswarm Analysis
* Individual Merchant Explanations
* Risk Driver Identification

### Dashboard

Interactive Streamlit dashboard featuring:

* Portfolio KPIs
* Sector Risk Analysis
* Churn Probability Distribution
* High-Risk Merchant Watchlist
* Explainable AI Insights

---

## Key Findings

SHAP analysis revealed that the strongest predictors of churn are:

| Feature                       | Importance |
| ----------------------------- | ---------- |
| Balance Drop (%)              | High       |
| Days Since Last Transaction   | High       |
| Ticket Size Variance (%)      | High       |
| Cashback Utilization          | High       |
| Complaints (12m)              | Medium     |
| Transaction Velocity Drop (%) | Medium     |

### Business Insight

Behavioral signals such as inactivity and declining balances were significantly more predictive of merchant churn than static merchant attributes.

---

## Technology Stack

### Data Processing

* Python
* Pandas
* NumPy

### Machine Learning

* XGBoost
* Scikit-Learn

### Explainable AI

* SHAP

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard

* Streamlit

### Model Persistence

* Joblib

---

## Project Structure

```text
MSME_CHURN_ANALYTICS/
│
├── app.py
├── train_model.py
├── generate_dataset.py
├── generate_shap.py
│
├── models/
│   ├── shap_values.npy
│   ├── shap_sample.csv
│   └── shap_global.json
│
├── msme_merchant_churn.csv
├── xgb_churn_model.pkl
├── sector_encoder.pkl
│
├── feature_importance.png
├── roc_curve.png
├── confusion_matrix.png
├── sector_churn_rate.png
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/Kaviya-lab/MSME_CHURN_ANALYTICS.git

cd MSME_CHURN_ANALYTICS

pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Business Value

This solution helps banks:

* Reduce merchant churn
* Improve customer retention
* Prioritize high-value accounts
* Detect financial distress early
* Enable data-driven relationship management

---

## Future Enhancements

* Real-time transaction monitoring
* Automated retention recommendations
* Revenue-at-risk estimation
* Loan default prediction
* Integration with banking CRM systems

---

## Author

**Kaviya L**

AI • Data Science • Banking Analytics • Explainable AI
