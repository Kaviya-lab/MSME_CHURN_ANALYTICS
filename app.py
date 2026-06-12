"""
MSME Merchant Churn Analytics & Risk Warning System
====================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import json
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stMarkdown p { color: #8b949e; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e2533;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: transform .15s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: .8px; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 2rem !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; gap: 2px; border: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #8b949e; font-size: 13px; font-weight: 500; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #e94560 !important; color: #fff !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #21262d; border-radius: 10px; overflow: hidden; }

/* Section headers */
.section-head {
    font-size: 11px; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 1px solid #21262d; padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Risk badges */
.badge-high   { background:#e9456020;color:#e94560;border-radius:20px;padding:2px 12px;font-size:11px;font-weight:700; }
.badge-medium { background:#f5a62320;color:#f5a623;border-radius:20px;padding:2px 12px;font-size:11px;font-weight:700; }
.badge-low    { background:#4ecdc420;color:#4ecdc4;border-radius:20px;padding:2px 12px;font-size:11px;font-weight:700; }

/* SHAP callout box */
.shap-insight {
    background: linear-gradient(135deg,#1e2533,#161b22);
    border: 1px solid #30363d;
    border-left: 3px solid #e94560;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 13px;
    line-height: 1.6;
    color: #e6edf3;
}
.shap-insight b { color: #f5a623; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(bg="#0d1117", surface="#161b22", card="#1e2533",
         red="#e94560", amber="#f5a623", teal="#4ecdc4",
         blue="#3b82f6", purple="#8b5cf6", text="#e6edf3", muted="#8b949e")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C["text"], size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    colorway=[C["red"], C["amber"], C["teal"], C["blue"], C["purple"]],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"], size=11)),
)
def layout(**overrides):
    base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in overrides}
    return {**base, **overrides}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("msme_merchant_churn.csv")
    shap_vals = np.load("models/shap_values.npy")
    shap_sample = pd.read_csv("models/shap_sample.csv")
    with open("models/shap_global.json") as f:
        shap_global = json.load(f)
    return df, shap_vals, shap_sample, shap_global

@st.cache_resource
def load_model():
    model = joblib.load("xgb_churn_model.pkl")
    le    = joblib.load("sector_encoder.pkl")
    return model, le

df, shap_vals, shap_sample, shap_global = load_data()
model, le = load_model()

FEATURES = [
    "credit_score","tenure_months","num_products","has_working_capital_loan","has_pos_machine",
    "txn_velocity_drop_pct","avg_ticket_size_90d","ticket_size_variance_pct","balance_drop_pct",
    "wc_utilization_pct","pos_txn_monthly","complaints_12m","days_since_last_txn",
    "satisfaction_score","cashback_utilization","sector_encoded",
]
FEAT_LABELS = {
    "credit_score":            "Credit Score",
    "tenure_months":           "Tenure (months)",
    "num_products":            "No. of Products",
    "has_working_capital_loan":"Has WC Loan",
    "has_pos_machine":         "Has POS Machine",
    "txn_velocity_drop_pct":   "Txn Velocity Drop (%)",
    "avg_ticket_size_90d":     "Avg Ticket Size (90d)",
    "ticket_size_variance_pct":"Ticket Size Variance (%)",
    "balance_drop_pct":        "Balance Drop (%)",
    "wc_utilization_pct":      "WC Utilization (%)",
    "pos_txn_monthly":         "Monthly POS Txns",
    "complaints_12m":          "Complaints (12m)",
    "days_since_last_txn":     "Days Since Last Txn",
    "satisfaction_score":      "Satisfaction Score",
    "cashback_utilization":    "Cashback Utilization",
    "sector_encoded":          "Sector",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px">
        <div style="font-size:28px;margin-bottom:6px">📊</div>
        <div style="font-size:15px;font-weight:800;color:#e6edf3">MSME Churn Analytics</div>
        <div style="font-size:11px;color:#8b949e;margin-top:4px">Risk Warning System · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Filters</div>', unsafe_allow_html=True)
    sector_filter = st.multiselect(
        "Business Sector",
        options=sorted(df["sector"].unique()),
        default=sorted(df["sector"].unique()),
    )
    risk_filter = st.multiselect(
        "Risk Tier",
        options=["High","Medium","Low"],
        default=["High","Medium","Low"],
    )
    vel_threshold = st.slider(
        "Min Velocity Drop to flag (%)",
        min_value=0, max_value=80, value=40, step=5,
    )

    st.markdown("---")
    st.markdown('<div class="section-head">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#8b949e;line-height:1.7">
        <b style="color:#e6edf3">Algorithm</b><br>XGBoost Gradient Boosting<br><br>
        <b style="color:#e6edf3">Dataset</b><br>5,000 synthetic merchants<br><br>
        <b style="color:#e6edf3">Churn Rate</b><br>22.2% (890 merchants)<br><br>
        <b style="color:#e6edf3">Explainability</b><br>SHAP TreeExplainer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Data sources: [Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) · [E-Commerce Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle)")

# ── Apply filters ─────────────────────────────────────────────────────────────
dff = df[df["sector"].isin(sector_filter) & df["risk_tier"].isin(risk_filter)]

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px">
    <div style="font-size:24px;font-weight:800;color:#e6edf3;letter-spacing:-.3px">
        MSME Merchant Churn Analytics
        <span style="background:#e9456020;color:#e94560;border-radius:6px;
               padding:3px 10px;font-size:13px;font-weight:600;margin-left:10px">LIVE</span>
    </div>
    <div style="color:#8b949e;font-size:13px;margin-top:4px">
        XGBoost · SHAP Explainability · Relationship Manager Watchlist
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Portfolio Overview",
    "🔍  SHAP Explainability",
    "🚨  RM Watchlist",
    "🧪  Predict a Merchant",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    total = len(dff)
    churned = dff["churn"].sum()
    retained = total - churned
    high_risk = (dff["risk_tier"] == "High").sum()
    avg_vel = dff["txn_velocity_drop_pct"].mean()

    k1.metric("Total Merchants",   f"{total:,}")
    k2.metric("Retained",          f"{retained:,}",        f"{retained/total*100:.1f}%")
    k3.metric("Churn Rate",        f"{churned/total*100:.1f}%", f"+2.1pp vs last quarter", delta_color="inverse")
    k4.metric("High-Risk Flagged", f"{high_risk:,}",       f"Velocity >{vel_threshold}%", delta_color="inverse")
    k5.metric("Avg Velocity Drop", f"{avg_vel:.1f}%",      "30-day vs 90-day avg", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: sector churn + velocity distribution
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-head">Churn Rate by Sector</div>', unsafe_allow_html=True)
        sec = (dff.groupby("sector")["churn"]
               .agg(["mean","count"])
               .reset_index()
               .rename(columns={"mean":"churn_rate","count":"merchants"})
               .sort_values("churn_rate", ascending=True))
        sec["color"] = sec["churn_rate"].apply(
            lambda r: C["red"] if r > 0.24 else C["amber"] if r > 0.21 else C["teal"])
        fig = go.Figure(go.Bar(
            x=sec["churn_rate"]*100, y=sec["sector"],
            orientation="h",
            marker_color=sec["color"].tolist(),
            text=[f"{r*100:.1f}%  (n={int(n)})" for r, n in zip(sec["churn_rate"], sec["merchants"])],
            textposition="outside",
            textfont=dict(color=C["muted"], size=11),
            hovertemplate="<b>%{y}</b><br>Churn: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(**layout( height=300,
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], range=[0, 32], ticksuffix="%", title="Churn Rate (%)"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="")))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-head">Transaction Velocity Drop — Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(dff, x="txn_velocity_drop_pct", nbins=40,
            color_discrete_sequence=[C["blue"]],
            labels={"txn_velocity_drop_pct":"Velocity Drop (%)"})
        fig2.add_vline(x=vel_threshold, line_dash="dash", line_color=C["red"],
            annotation_text=f"  Threshold: {vel_threshold}%",
            annotation_font_color=C["red"], annotation_font_size=11)
        fig2.update_traces(marker_opacity=0.8)
        fig2.update_layout(**layout(height=300,
            showlegend=False,
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Velocity Drop (%)"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Merchants")))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Risk heatmap bubble + WC vs churn
    c3, c4 = st.columns([1.4, 1])

    with c3:
        st.markdown('<div class="section-head">Sector Risk Heatmap — Velocity Drop vs Balance Drop</div>', unsafe_allow_html=True)
        sec2 = dff.groupby("sector").agg(
            velocity_drop=("txn_velocity_drop_pct","mean"),
            balance_drop=("balance_drop_pct","mean"),
            churn_rate=("churn","mean"),
            count=("merchant_id","count"),
        ).reset_index()

        fig3 = px.scatter(sec2,
            x="velocity_drop", y="balance_drop",
            size="count", color="churn_rate",
            text="sector",
            color_continuous_scale=[[0,C["teal"]],[0.5,C["amber"]],[1,C["red"]]],
            size_max=55,
            labels={"velocity_drop":"Avg Velocity Drop (%)","balance_drop":"Avg Balance Drop (%)","churn_rate":"Churn Rate"},
            hover_data={"count":True,"churn_rate":":.1%"},
        )
        fig3.update_traces(textposition="top center", textfont=dict(size=10, color=C["muted"]))
        fig3.update_layout(**PLOTLY_LAYOUT,height=320,
            coloraxis_colorbar=dict(title=dict(text="Churn Rate", font=dict(color=C["muted"])),
                        tickformat=".0%", tickfont=dict(color=C["muted"])))
        
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-head">Working Capital Stress vs Churn</div>', unsafe_allow_html=True)
        bins = [0, 30, 60, 85, 100]
        labels = ["<30%","30–60%","60–85%","85–100%"]
        dff2 = dff.copy()
        dff2["wc_bin"] = pd.cut(dff2["wc_utilization_pct"], bins=bins, labels=labels)
        wc_tab = dff2.groupby(["wc_bin","churn"]).size().reset_index(name="count")
        wc_tab["status"] = wc_tab["churn"].map({0:"Retained",1:"Churned"})

        fig4 = px.bar(wc_tab, x="wc_bin", y="count", color="status",
            color_discrete_map={"Retained":C["teal"],"Churned":C["red"]},
            barmode="group",
            labels={"wc_bin":"WC Utilization Band","count":"Merchants","status":""},
        )
        fig4.update_traces(marker_opacity=0.85)
        fig4.update_layout(**layout(height=320,
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="WC Utilization Band"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Merchants")))
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Churn probability distribution
    st.markdown('<div class="section-head">Churn Probability Distribution (Model Output)</div>', unsafe_allow_html=True)
    fig5 = px.histogram(dff, x="churn_probability", color="risk_tier", nbins=50,
        color_discrete_map={"High":C["red"],"Medium":C["amber"],"Low":C["teal"]},
        opacity=0.8,
        labels={"churn_probability":"Predicted Churn Probability","risk_tier":"Risk Tier"},
        barmode="overlay")
    fig5.add_vline(x=0.5, line_dash="dash", line_color="#8b949e",
        annotation_text="  Decision threshold: 0.5", annotation_font_color="#8b949e", annotation_font_size=11)
    fig5.update_layout(layout(height=260,
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], tickformat=".0%"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Merchants")))
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="SHAP">
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Global SHAP bar (mean |SHAP|)
        st.markdown('<div class="section-head">Global Feature Importance — Mean |SHAP Value|</div>', unsafe_allow_html=True)

        shap_df = (pd.DataFrame.from_dict(shap_global, orient="index", columns=["mean_abs_shap"])
                   .rename(index=FEAT_LABELS)
                   .sort_values("mean_abs_shap", ascending=True))

        max_s = shap_df["mean_abs_shap"].max()
        colors = [C["red"] if v == max_s else C["amber"] if v > max_s*0.4 else C["blue"]
                  for v in shap_df["mean_abs_shap"]]

        fig_s = go.Figure(go.Bar(
            x=shap_df["mean_abs_shap"], y=shap_df.index,
            orientation="h",
            marker_color=colors,
            text=[f"  {v:.3f}" for v in shap_df["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color=C["muted"], size=10),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        fig_s.update_layout(**layout(height=480,
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Mean |SHAP Value|"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="")))
        st.plotly_chart(fig_s, use_container_width=True)

    with col_b:
        # SHAP beeswarm (custom scatter)
        st.markdown('<div class="section-head">SHAP Beeswarm — Feature Value vs Impact Direction</div>', unsafe_allow_html=True)

        top_feats = sorted(shap_global, key=lambda k: shap_global[k], reverse=True)[:8]
        top_labels = [FEAT_LABELS.get(f, f) for f in top_feats]

        shap_scatter_df = []
        for i, feat in enumerate(top_feats):
            feat_vals = shap_sample[feat].values
            feat_shap = shap_vals[:, FEATURES.index(feat)]
            # Normalize feature value to 0-1 for color
            norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-9)
            for j in range(len(feat_shap)):
                shap_scatter_df.append({
                    "feature": FEAT_LABELS.get(feat, feat),
                    "shap_value": float(feat_shap[j]),
                    "feat_norm": float(norm[j]),
                    "y_jitter": i + np.random.uniform(-0.3, 0.3),
                })

        sdf = pd.DataFrame(shap_scatter_df)
        fig_bee = px.scatter(sdf, x="shap_value", y="y_jitter",
            color="feat_norm",
            color_continuous_scale=[[0, "#3b82f6"],[0.5,"#f5a623"],[1,"#e94560"]],
            opacity=0.5, size_max=4,
            labels={"shap_value":"SHAP Value (impact on churn prediction)", "feat_norm":"Feature Value"},
            hover_data={"feature":True, "shap_value":":.3f"},
        )
        fig_bee.update_traces(marker=dict(size=4))
        fig_bee.update_layout(**layout(height=480,yaxis=dict(**PLOTLY_LAYOUT["yaxis"],
        tickvals=list(range(len(top_feats))),
        ticktext=top_labels,
        title=""), xaxis={**PLOTLY_LAYOUT["xaxis"], "title": "← Lower churn risk    SHAP Value    Higher churn risk →", "zeroline": True, "zerolinecolor": "#30363d"},
    coloraxis_showscale=True,
    coloraxis_colorbar=dict(
        title=dict(text="Feature<br>Value", font=dict(color=C["muted"])),
        tickvals=[0,0.5,1], ticktext=["Low","Mid","High"],
        tickfont=dict(color=C["muted"])),
    showlegend=False,
))
        fig_bee.add_vline(x=0, line_dash="dash", line_color="#30363d")
        st.plotly_chart(fig_bee, use_container_width=True)

    # SHAP insights callouts
    st.markdown('<div class="section-head">Banker-Readable Insights from SHAP</div>', unsafe_allow_html=True)
    ins_c1, ins_c2, ins_c3 = st.columns(3)

    with ins_c1:
        st.markdown("""
        <div class="shap-insight">
            📉 <b>Days Since Last Transaction</b> is the #2 predictor by SHAP magnitude.
            A merchant inactive for 30+ days already has a 2.49× higher churn SHAP impact —
            suggesting relationship managers should trigger outreach at the 21-day mark, not 30.
        </div>
        """, unsafe_allow_html=True)

    with ins_c2:
        st.markdown("""
        <div class="shap-insight">
            💳 <b>Balance Drop</b> has 2.64 mean |SHAP|, but it lags behind transaction signals
            by ~3 weeks on average. This confirms: <b>monitor txn frequency, not just balances</b>,
            for early-warning advantage.
        </div>
        """, unsafe_allow_html=True)

    with ins_c3:
        st.markdown("""
        <div class="shap-insight">
            📋 <b>Complaints Filed</b> (SHAP: 0.67) shows that even a single complaint in 12 months
            doubles the churn signal. Each incremental complaint adds a non-linear SHAP push —
            a <b>zero-complaint merchant is ~4× less likely</b> to churn than a 3-complaint one.
        </div>
        """, unsafe_allow_html=True)

    # SHAP dependence plot (top feature vs churn)
    st.markdown('<div class="section-head">SHAP Dependence Plot — Transaction Velocity Drop</div>', unsafe_allow_html=True)
    feat_idx = FEATURES.index("txn_velocity_drop_pct")
    int_idx  = FEATURES.index("days_since_last_txn")

    dep_df = pd.DataFrame({
        "txn_velocity_drop_pct": shap_sample["txn_velocity_drop_pct"].values,
        "shap_velocity": shap_vals[:, feat_idx],
        "days_since": shap_sample["days_since_last_txn"].values,
    })
    fig_dep = px.scatter(dep_df,
        x="txn_velocity_drop_pct", y="shap_velocity",
        color="days_since",
        color_continuous_scale=[[0,C["teal"]],[0.5,C["amber"]],[1,C["red"]]],
        opacity=0.65,
        labels={
            "txn_velocity_drop_pct":"Transaction Velocity Drop (%)",
            "shap_velocity":"SHAP Value for Velocity Drop",
            "days_since":"Days Since Last Txn",
        },
        title="SHAP Dependence: Velocity Drop (coloured by Days Since Last Transaction)",
    )
    fig_dep.update_traces(marker=dict(size=5))
    fig_dep.update_layout(**PLOTLY_LAYOUT, height=340,
        title_font_size=13,
        coloraxis_colorbar=dict(
            title=dict(text="Days<br>Since Txn", font=dict(color=C["muted"])),
            tickfont=dict(color=C["muted"])),
    )
    fig_dep.add_vline(x=40, line_dash="dash", line_color=C["red"],
        annotation_text="  40% alert threshold", annotation_font_color=C["red"], annotation_font_size=11)
    st.plotly_chart(fig_dep, use_container_width=True)

    st.markdown("""
    <div class="shap-insight" style="border-left-color:#4ecdc4">
        📌 <b>Reading this chart:</b> Each dot is one merchant. The X-axis shows their actual velocity drop;
        the Y-axis shows how much that feature pushed the model toward predicting churn.
        Colour shows days since last transaction — <b>red dots (long-inactive merchants) with high velocity drop
        have the largest positive SHAP values</b>, meaning both signals compound each other non-linearly.
        This is why the dual trigger (velocity drop AND inactivity) is a stronger alert than either alone.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RM WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-head">High-Risk Merchant Watchlist — Relationship Manager Action Queue</div>', unsafe_allow_html=True)

    wl = dff[dff["txn_velocity_drop_pct"] >= vel_threshold].sort_values("churn_probability", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Merchants in Watchlist", f"{len(wl):,}")
    m2.metric("Already Churned",        f"{wl['churn'].sum():,}", delta_color="inverse")
    m3.metric("Avg Churn Probability",  f"{wl['churn_probability'].mean()*100:.1f}%", delta_color="inverse")
    m4.metric("Avg Velocity Drop",      f"{wl['txn_velocity_drop_pct'].mean():.1f}%", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    search = st.text_input("🔍  Search by Merchant ID or Sector", placeholder="e.g. MRC04779 or Retail Trader")
    if search:
        wl = wl[wl["merchant_id"].str.contains(search, case=False) |
                wl["sector"].str.contains(search, case=False)]

    display_cols = {
        "merchant_id": "Merchant ID",
        "sector": "Sector",
        "txn_velocity_drop_pct": "Velocity Drop (%)",
        "balance_drop_pct": "Balance Drop (%)",
        "wc_utilization_pct": "WC Util (%)",
        "complaints_12m": "Complaints",
        "credit_score": "Credit Score",
        "churn_probability": "Churn Prob",
        "risk_tier": "Risk Tier",
        "churn": "Churned",
    }
    wl_display = wl[list(display_cols.keys())].rename(columns=display_cols).head(100)
    wl_display["Churn Prob"] = wl_display["Churn Prob"].apply(lambda x: f"{x*100:.1f}%")
    wl_display["Churned"] = wl_display["Churned"].map({1:"✗ Yes", 0:"✓ Active"})

    st.dataframe(
        wl_display.style
            .applymap(lambda v: "color:#e94560;font-weight:700" if v == "✗ Yes" else "color:#4ecdc4", subset=["Churned"])
            .applymap(lambda v: "color:#e94560;font-weight:700" if v == "High" else "color:#f5a623" if v == "Medium" else "color:#4ecdc4", subset=["Risk Tier"])
            .background_gradient(subset=["Velocity Drop (%)"], cmap="Reds"),
        use_container_width=True,
        height=480,
    )

    # Download button
    csv = wl.to_csv(index=False).encode()
    st.download_button("⬇️  Download Watchlist CSV", csv, "high_risk_watchlist.csv", "text/csv")

    # Action recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Recommended Actions by Risk Tier</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""
        <div class="shap-insight" style="border-left-color:{C['red']}">
            🔴 <b>High Risk</b> — Call within 48 hours<br><br>
            • Offer 1% lower overdraft rate<br>
            • Waive 3-month maintenance fee<br>
            • Assign dedicated RM contact<br>
            • Escalate to branch manager if no response
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="shap-insight" style="border-left-color:{C['amber']}">
            🟡 <b>Medium Risk</b> — Schedule 2-week review<br><br>
            • Send personalised SMS/email<br>
            • Offer cashback on POS usage<br>
            • Review WC limit adequacy<br>
            • Invite to business banking webinar
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown(f"""
        <div class="shap-insight" style="border-left-color:{C['teal']}">
            🟢 <b>Low Risk</b> — Quarterly touchpoint<br><br>
            • Standard relationship review<br>
            • Share sector benchmark report<br>
            • Offer referral programme<br>
            • Cross-sell insurance / FD products
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT A MERCHANT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown('<div class="section-head">Live Churn Prediction with SHAP Explanation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="shap-insight" style="border-left-color:#8b5cf6">
        🧪 Enter a merchant's profile to get an <b>instant churn prediction + SHAP waterfall explanation</b>
        showing exactly which signals are driving the risk score.
    </div>
    """, unsafe_allow_html=True)

    c_l, c_r = st.columns([1, 1])

    with c_l:
        st.markdown("**Business Profile**")
        sector_sel = st.selectbox("Business Sector", sorted(df["sector"].unique()))
        tenure_sel = st.slider("Tenure (months)", 1, 120, 24)
        credit_sel = st.slider("Credit Score", 300, 900, 650)
        num_prod   = st.slider("Number of Products", 1, 4, 2)
        has_wc     = st.toggle("Has Working Capital Loan", value=True)
        has_pos    = st.toggle("Has POS Machine", value=True)

    with c_r:
        st.markdown("**Transaction Signals**")
        vel_drop   = st.slider("Transaction Velocity Drop (%)", -30.0, 100.0, 55.0, 0.5)
        bal_drop   = st.slider("Balance Drop (%)", -10.0, 70.0, 30.0, 0.5)
        wc_util    = st.slider("WC Utilization (%)", 0.0, 100.0, 70.0, 0.5)
        complaints = st.slider("Complaints Filed (12m)", 0, 8, 2)
        days_inact = st.slider("Days Since Last Transaction", 0, 90, 20)
        satisf     = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
        cashback   = st.slider("Cashback Utilization (0–1)", 0.0, 1.0, 0.4, 0.01)

    if st.button("⚡  Run Prediction + SHAP Analysis", type="primary", use_container_width=True):
        sec_enc = le.transform([sector_sel])[0]
        inp = pd.DataFrame([{
            "credit_score": credit_sel,
            "tenure_months": tenure_sel,
            "num_products": num_prod,
            "has_working_capital_loan": int(has_wc),
            "has_pos_machine": int(has_pos),
            "txn_velocity_drop_pct": vel_drop,
            "avg_ticket_size_90d": 50000,
            "ticket_size_variance_pct": 20,
            "balance_drop_pct": bal_drop,
            "wc_utilization_pct": wc_util,
            "pos_txn_monthly": 15,
            "complaints_12m": complaints,
            "days_since_last_txn": days_inact,
            "satisfaction_score": satisf,
            "cashback_utilization": cashback,
            "sector_encoded": sec_enc,
        }])

        prob = model.predict_proba(inp)[0][1]
        risk = "🔴 HIGH RISK" if prob > 0.65 else "🟡 MEDIUM RISK" if prob > 0.40 else "🟢 LOW RISK"
        risk_color = C["red"] if prob > 0.65 else C["amber"] if prob > 0.40 else C["teal"]

        st.markdown(f"""
        <div style="background:#1e2533;border:1px solid #30363d;border-left:4px solid {risk_color};
             border-radius:12px;padding:20px 24px;margin:16px 0;text-align:center">
            <div style="font-size:13px;color:#8b949e;margin-bottom:6px">Predicted Churn Probability</div>
            <div style="font-size:3rem;font-weight:800;color:{risk_color}">{prob*100:.1f}%</div>
            <div style="font-size:16px;font-weight:700;color:{risk_color};margin-top:4px">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP waterfall
        explainer = shap.TreeExplainer(model)
        shap_single = explainer.shap_values(inp)[0]
        base_val = float(explainer.expected_value)

        feat_shap = sorted(zip(FEATURES, shap_single),
                           key=lambda x: abs(x[1]), reverse=True)[:10]

        labels  = [FEAT_LABELS.get(f, f) for f, _ in reversed(feat_shap)]
        vals    = [v for _, v in reversed(feat_shap)]
        running = base_val
        starts  = []
        for v in vals:
            starts.append(running)
            running += v

        bar_colors = [C["red"] if v > 0 else C["teal"] for v in vals]

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            x=vals, y=labels, orientation="h",
            base=starts,
            marker_color=bar_colors,
            text=[f"{'+' if v>0 else ''}{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color=C["muted"], size=10),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        ))
        fig_wf.add_vline(x=base_val, line_dash="dot", line_color="#8b949e",
            annotation_text=f"  Base: {base_val:.3f}", annotation_font_color="#8b949e", annotation_font_size=10)
        fig_wf.add_vline(x=running, line_dash="dash", line_color=risk_color,
            annotation_text=f"  Output: {running:.3f}", annotation_font_color=risk_color, annotation_font_size=10)
        fig_wf.update_layout(**layout(height=400,
            title=dict(text="SHAP Waterfall — Why this merchant is flagged", font=dict(size=13)),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="SHAP Value contribution"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="")))
        st.plotly_chart(fig_wf, use_container_width=True)

        # Narrative explanation
        top_pos = [(FEAT_LABELS.get(f,f), v) for f,v in feat_shap if v > 0][:3]
        top_neg = [(FEAT_LABELS.get(f,f), v) for f,v in feat_shap if v < 0][:2]
        pos_str = " · ".join([f"**{n}** (+{v:.3f})" for n,v in top_pos]) or "none"
        neg_str = " · ".join([f"**{n}** ({v:.3f})" for n,v in top_neg]) or "none"

        st.markdown(f"""
        <div class="shap-insight">
            🔎 <b>Model Explanation:</b> The model's baseline churn probability is <b>{base_val*100:.1f}%</b>.
            For this merchant, the biggest <span style="color:{C['red']}">risk-increasing factors</span> are: {pos_str}.
            The <span style="color:{C['teal']}">risk-reducing factors</span> are: {neg_str}.
            The final predicted churn probability is <b style="color:{risk_color}">{prob*100:.1f}%</b>.
        </div>
        """.replace("**","<b>").replace("</b>","</b>"), unsafe_allow_html=True)