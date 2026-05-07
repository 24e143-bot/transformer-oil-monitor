"""
=============================================================
  TRANSFORMER OIL DEGRADATION — STREAMLIT DASHBOARD v3
  Scale: REAL-WORLD (months, 0–180 months = 15 years)
  Run with: streamlit run dashboard.py
=============================================================
Install dependencies first:
  pip install streamlit pandas numpy scikit-learn matplotlib plotly
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ─── PAGE CONFIG ──────────────────────────────────────────

st.set_page_config(
    page_title="Transformer Oil Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.sub-title  { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.alert-critical {
    background: #fff0f0;
    border-left: 4px solid #d32f2f;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    color: #b71c1c;
}
.alert-warning {
    background: #fffde7;
    border-left: 4px solid #f9a825;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    color: #e65100;
}
.alert-ok {
    background: #f1f8e9;
    border-left: 4px solid #558b2f;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA ─────────────────────────────────────────────────
# Time axis: 0–180 months (15 years) — real-world scale
# Remaining_Life: 0–180 months
# Zone boundaries:
#   Good     : 0  – ~32 months
#   Moderate : 32 – ~72 months
#   Bad      : 72 – 180 months

@st.cache_data
def load_data():
    # 51 points linearly spaced over 180 months
    time_months = [round(i * 180 / 50, 1) for i in range(51)]  # 0.0 to 180.0

    data = {
        "Time_months":    time_months,
        "Rb_MOhm":        [1200,1192,1185,1170,1158,1145,1128,1105,1080,1055,1030,1005,980,
                           955,930,900,875,850,820,790,760,730,700,670,640,610,580,550,520,
                           490,460,435,410,385,360,335,310,290,270,250,230,210,195,180,165,
                           150,135,120,110,100,90],
        "Rct_Ohm":        [950,942,935,920,905,890,875,855,835,815,795,770,750,730,710,690,
                           670,650,630,610,590,570,550,530,510,490,470,450,430,410,390,370,
                           350,330,310,290,270,255,240,225,210,195,180,165,150,140,130,120,
                           110,100,90],
        "Impedance_Ohm":  [500,502,504,508,512,518,525,535,548,560,575,590,610,630,650,670,
                           690,710,735,760,785,810,840,870,900,930,960,990,1020,1050,1080,
                           1110,1140,1170,1200,1230,1260,1290,1320,1350,1380,1410,1440,1470,
                           1500,1530,1560,1590,1620,1650,1680],
        "Freq_Response":  [0.020,0.021,0.022,0.024,0.026,0.028,0.031,0.035,0.039,0.044,
                           0.048,0.052,0.057,0.062,0.068,0.073,0.079,0.085,0.092,0.100,
                           0.108,0.117,0.127,0.138,0.150,0.163,0.177,0.192,0.208,0.225,
                           0.243,0.262,0.282,0.303,0.325,0.348,0.372,0.397,0.423,0.450,
                           0.478,0.507,0.537,0.568,0.600,0.633,0.667,0.702,0.738,0.775,0.813],
        "Health_Status":  ["Good"]*9+["Moderate"]*11+["Bad"]*31,
        # Remaining_Life in months: 180 months → 0 months
        "Remaining_Life": [round(i * 180 / 50, 1) for i in range(50, -1, -1)],
    }
    return pd.DataFrame(data)


@st.cache_resource
def train_models(df):
    FEATURES = ["Rb_MOhm", "Rct_Ohm", "Impedance_Ohm", "Freq_Response"]
    le = LabelEncoder()
    y_cls = le.fit_transform(df["Health_Status"])
    y_reg = df["Remaining_Life"]
    X = df[FEATURES]
    X_tr, _, y_c_tr, _ = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    X_tr2, _, y_r_tr, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_c_tr)
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr2, y_r_tr)
    return clf, reg, le, FEATURES


def health_index(rb, rct, imp, freq):
    rb_n  = (rb   - 90) / (1200 - 90)
    rct_n = (rct  - 90) / (950  - 90)
    imp_n = 1 - (imp  - 500)  / (1680 - 500)
    frq_n = 1 - (freq - 0.020) / (0.813 - 0.020)
    hi    = (rb_n*0.30 + rct_n*0.30 + imp_n*0.20 + frq_n*0.20) * 100
    return round(float(np.clip(hi, 0, 100)), 1)


# ─── LOAD ─────────────────────────────────────────────────

df = load_data()
clf, reg, le, FEATURES = train_models(df)

color_map = {"Good": "#3B6D11", "Moderate": "#BA7517", "Bad": "#A32D2D"}

# Zone boundaries in months
GOOD_END = 18 / 100 * 180   # ≈ 32.4 months
MOD_END  = 40 / 100 * 180   # ≈ 72.0 months

# ─── SIDEBAR ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Input Parameters")
    st.caption("Adjust values to get real-time prediction")

    rb   = st.slider("Rb — Bulk Resistance (MΩ)",    90, 1200, 850, step=1)
    rct  = st.slider("Rct — Charge Transfer Res (Ω)", 90,  950, 660, step=1)
    imp  = st.slider("Impedance (Ω)",                500, 1680, 710, step=1)
    freq = st.slider("Frequency Response",          0.020, 0.813, 0.085, step=0.001)

    st.divider()
    st.caption("📌 Real-world scale")
    st.info(
        "Transformer oil typically lasts **10–15 years (120–180 months)** "
        "under normal operating conditions.\n\n"
        "⚠ **Warning threshold**: 24 months remaining\n\n"
        "🚨 **Critical threshold**: 6 months remaining"
    )

    st.divider()
    if st.button("🔁 Load Good Sample"):
        rb, rct, imp, freq = 1145, 890, 518, 0.028
    if st.button("⚠️ Load Critical Sample"):
        rb, rct, imp, freq = 150, 140, 1530, 0.633


# ─── PREDICTIONS ──────────────────────────────────────────

X_in = pd.DataFrame([{"Rb_MOhm": rb, "Rct_Ohm": rct, "Impedance_Ohm": imp, "Freq_Response": freq}])
health_pred  = le.inverse_transform(clf.predict(X_in[FEATURES]))[0]
life_months  = max(0, round(float(reg.predict(X_in[FEATURES])[0]), 1))
life_years   = round(life_months / 12, 1)
hi           = health_index(rb, rct, imp, freq)
proba        = dict(zip(le.classes_, clf.predict_proba(X_in[FEATURES])[0]))


# ─── HEADER ───────────────────────────────────────────────

st.markdown('<p class="main-title">⚡ Transformer Oil Degradation Monitor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">EIS-based condition monitoring | ML prediction system | Real-world scale: 0–180 months</p>', unsafe_allow_html=True)

# Alert — thresholds in months
if life_months <= 6:
    st.markdown(
        f'<div class="alert-critical">🚨 CRITICAL: Oil will reach failure in approximately '
        f'<strong>{life_months} months</strong>. Replace immediately.</div>',
        unsafe_allow_html=True
    )
elif life_months <= 24:
    st.markdown(
        f'<div class="alert-warning">⚠️ WARNING: Oil is degrading. Plan maintenance within '
        f'<strong>{life_months} months ({life_years} years)</strong>.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="alert-ok">✅ STATUS OK: Oil condition is acceptable. Estimated '
        f'<strong>{life_months} months ({life_years} years)</strong> remaining.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ─── KPI CARDS ────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("🩺 Health Status",   health_pred)
c2.metric("💯 Health Index",    f"{hi} / 100")
c3.metric("⏳ Remaining Life",  f"{life_months} months ({life_years} yrs)")
c4.metric("📉 Degradation",     f"{round(100-hi, 1)}%")

st.markdown("---")


# ─── GAUGE + PROBABILITIES ────────────────────────────────

col_g, col_p = st.columns([1, 1])

with col_g:
    st.subheader("Health Index Gauge")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hi,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color_map.get(health_pred, "#555")},
            "steps": [
                {"range": [0,  50], "color": "#fff0f0"},
                {"range": [50, 75], "color": "#fffde7"},
                {"range": [75, 100], "color": "#f1f8e9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.8,
                "value": 30
            }
        },
        title={"text": "Health Score (0 = Failed, 100 = New)"}
    ))
    fig_g.update_layout(height=260, margin=dict(t=40, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with col_p:
    st.subheader("Classification Confidence")
    fig_p = go.Figure(go.Bar(
        x=list(proba.keys()),
        y=[round(v*100, 1) for v in proba.values()],
        marker_color=[color_map[k] for k in proba.keys()],
        text=[f"{round(v*100,1)}%" for v in proba.values()],
        textposition="outside"
    ))
    fig_p.update_layout(
        yaxis=dict(range=[0, 110], title="Probability (%)"),
        xaxis_title="Health Class",
        height=260,
        margin=dict(t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown("---")


# ─── TREND CHARTS ─────────────────────────────────────────

st.subheader("📊 Parameter Trends & Degradation Zones")
tab1, tab2, tab3, tab4 = st.tabs(["Rb (MΩ)", "Rct (Ω)", "Impedance (Ω)", "Frequency Response"])

def make_trend(col, ylabel, current_val):
    fig = go.Figure()
    # Zone shading in months
    fig.add_vrect(x0=0,        x1=GOOD_END, fillcolor="#3B6D11", opacity=0.07,
                  line_width=0, annotation_text="Good Zone",     annotation_position="top left")
    fig.add_vrect(x0=GOOD_END, x1=MOD_END,  fillcolor="#BA7517", opacity=0.07,
                  line_width=0, annotation_text="Moderate Zone", annotation_position="top left")
    fig.add_vrect(x0=MOD_END,  x1=180,      fillcolor="#A32D2D", opacity=0.07,
                  line_width=0, annotation_text="Bad Zone",      annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=df["Time_months"], y=df[col],
        mode="lines+markers",
        line=dict(color="#185FA5", width=2),
        marker=dict(size=4),
        name=ylabel
    ))
    fig.add_hline(y=current_val, line_dash="dash", line_color="orange",
                  annotation_text=f"Current: {current_val}")
    fig.update_layout(
        xaxis_title="Time (months)",
        yaxis_title=ylabel,
        height=300,
        margin=dict(t=20, b=20)
    )
    return fig

with tab1:
    st.plotly_chart(make_trend("Rb_MOhm", "Rb (MΩ)", rb), use_container_width=True)
with tab2:
    st.plotly_chart(make_trend("Rct_Ohm", "Rct (Ω)", rct), use_container_width=True)
with tab3:
    st.plotly_chart(make_trend("Impedance_Ohm", "Impedance (Ω)", imp), use_container_width=True)
with tab4:
    st.plotly_chart(make_trend("Freq_Response", "Frequency Response", freq), use_container_width=True)

st.markdown("---")


# ─── REMAINING LIFE PROJECTION ────────────────────────────

st.subheader("⏳ Remaining Life Projection (months)")
fig_l = go.Figure()
fig_l.add_trace(go.Scatter(
    x=df["Time_months"], y=df["Remaining_Life"],
    mode="lines", fill="tozeroy",
    line=dict(color="#534AB7", width=2),
    fillcolor="rgba(83,74,183,0.10)",
    name="Remaining Life (months)"
))

# Estimate current age from remaining life
est_age_months = max(0, 180 - life_months)
fig_l.add_vline(
    x=est_age_months, line_dash="dash", line_color="orange",
    annotation_text=f"You are here (~{round(est_age_months)} months / {round(est_age_months/12, 1)} yrs)"
)
fig_l.add_hline(
    y=24, line_dash="dot", line_color="#BA7517",
    annotation_text="⚠ Warning: 24 months remaining"
)
fig_l.add_hline(
    y=6, line_dash="dot", line_color="red",
    annotation_text="🚨 Critical: 6 months remaining"
)
fig_l.update_layout(
    xaxis_title="Time (months)",
    yaxis_title="Remaining Life (months)",
    height=320,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_l, use_container_width=True)

st.markdown("---")
st.caption(
    "🎓 Project: Transformer Oil Condition Monitoring | "
    "Dataset: 51 samples (0–180 months real-world scale) | "
    "Models: Random Forest Classifier + Regressor | "
    "Standard: IEC 60422 transformer oil maintenance"
)
