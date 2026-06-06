# ╔══════════════════════════════════════════════════════════════════╗
# ║          HeartGuard AI  –  by Rahul Thakur                      ║
# ║  Advanced Heart Disease Prediction & Analytics Dashboard        ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import export_text
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI · Rahul Thakur",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "HeartGuard AI — Built by Rahul Thakur"
    }
)

# ─────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:      #07090f;
    --card:    #0f1622;
    --card2:   #131c2e;
    --red:     #e63946;
    --red2:    #b5131f;
    --green:   #06d6a0;
    --gold:    #ffd166;
    --blue:    #4895ef;
    --text:    #dde3f0;
    --muted:   #6b7a99;
    --border:  rgba(230,57,70,0.18);
    --border2: rgba(255,255,255,0.06);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }

/* ── HEADER ── */
.hero {
    background: linear-gradient(120deg, #100516 0%, #1a0510 30%, #0f1622 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.2rem 2.8rem 1.8rem;
    margin: -1rem -1rem 1.8rem -1rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content:'';
    position: absolute;
    top:-60px; right:-60px;
    width:340px; height:340px;
    background: radial-gradient(circle, rgba(230,57,70,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
    letter-spacing: -1.5px;
}
.hero h1 span { color: var(--red); }
.hero p { color: var(--muted); margin: .4rem 0 0; font-size: .95rem; }
.hero-badge {
    position: absolute; top: 1.4rem; right: 2rem;
    background: rgba(230,57,70,0.12);
    border: 1px solid var(--border);
    border-radius: 2rem;
    padding: .4rem 1rem;
    font-size: .75rem;
    color: var(--gold);
    font-weight: 600;
    letter-spacing: .5px;
}

/* ── KPIS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom:1.5rem; }
.kpi {
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 1rem;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
}
.kpi:hover { transform: translateY(-3px); border-color: var(--border); }
.kpi::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--red), transparent);
}
.kpi-val { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#fff; }
.kpi-lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:.1rem; }
.kpi-icon { position:absolute; top:1rem; right:1.2rem; font-size:1.6rem; opacity:.4; }

/* ── SECTION TITLES ── */
.st-hdr {
    font-family:'Syne',sans-serif;
    font-size:1.35rem; font-weight:800;
    color:#fff;
    margin:2rem 0 .9rem;
    display:flex; align-items:center; gap:.7rem;
}
.st-hdr::after { content:''; flex:1; height:1px; background:var(--border2); }

/* ── CARDS ── */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius:1.2rem;
    padding:1.5rem;
    margin-bottom:1rem;
}

/* ── RESULT BOXES ── */
.result-pos {
    background: linear-gradient(135deg,rgba(230,57,70,.12),rgba(157,2,8,.07));
    border: 1.5px solid var(--red);
    border-radius:1.4rem;
    padding:2rem 1.5rem;
    text-align:center;
    box-shadow: 0 0 30px rgba(230,57,70,.2);
}
.result-neg {
    background: linear-gradient(135deg,rgba(6,214,160,.12),rgba(0,120,80,.07));
    border: 1.5px solid var(--green);
    border-radius:1.4rem;
    padding:2rem 1.5rem;
    text-align:center;
    box-shadow: 0 0 30px rgba(6,214,160,.2);
}
.result-label { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; margin:.6rem 0 .3rem; }
.result-prob  { font-size:3rem; font-weight:700; font-family:'Syne',sans-serif; }
.result-sub   { color:var(--muted); font-size:.85rem; margin-top:.4rem; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1120 0%,#07090f 100%) !important;
    border-right: 1px solid var(--border2);
}
.sidebar-logo {
    text-align:center;
    padding: 1rem 0 1.2rem;
    border-bottom: 1px solid var(--border2);
    margin-bottom: 1rem;
}
.sidebar-logo .name { color:var(--gold); font-weight:700; font-size:.85rem; letter-spacing:1px; }

/* ── BUTTONS ── */
.stButton>button {
    background: linear-gradient(135deg, var(--red), var(--red2));
    color:#fff; border:none;
    border-radius:.75rem;
    padding:.7rem 1.8rem;
    font-family:'Outfit',sans-serif;
    font-weight:600; font-size:.95rem;
    letter-spacing:.3px;
    width:100%;
    box-shadow: 0 4px 18px rgba(230,57,70,.3);
    transition: all .2s;
}
.stButton>button:hover {
    transform:translateY(-2px);
    box-shadow: 0 8px 24px rgba(230,57,70,.5);
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--card); border-radius:.9rem;
    gap:.25rem; padding:.28rem;
    border: 1px solid var(--border2);
}
.stTabs [data-baseweb="tab"] { color:var(--muted); border-radius:.65rem; font-weight:500; font-size:.88rem; }
.stTabs [aria-selected="true"] { background:var(--red) !important; color:#fff !important; }

/* ── RISK TIMELINE ── */
.risk-item { display:flex; align-items:center; gap:.8rem; margin:.5rem 0; }
.risk-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.risk-text { font-size:.85rem; color:var(--text); }
.risk-val  { margin-left:auto; font-size:.82rem; color:var(--muted); }

/* ── RECOMMENDATION CARD ── */
.rec-card {
    background: var(--card2);
    border-left: 3px solid;
    border-radius: 0 .8rem .8rem 0;
    padding: .9rem 1.2rem;
    margin: .5rem 0;
    font-size: .88rem;
}

/* ── TABLE STYLING ── */
.stDataFrame { border-radius: .8rem; overflow: hidden; }

/* ── FOOTER ── */
.footer {
    text-align:center;
    padding:2rem 0 1rem;
    border-top:1px solid var(--border2);
    margin-top:3rem;
    font-size:.76rem;
    color:var(--muted);
}
.footer span { color:var(--gold); font-weight:600; }

/* ── INPUT TWEAKS ── */
.stSlider>div>div>div>div { background:var(--red) !important; }
.stSelectbox label, .stSlider label, .stNumberInput label {
    color:var(--muted) !important;
    font-size:.8rem;
}
div[data-testid="stMetricValue"] { color: var(--red); font-family:'Syne',sans-serif; }

/* ── UPLOAD ZONE ── */
.uploadedFile { border-radius:.8rem !important; }

hr { border-color:var(--border2) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load('heart_disease_model.pkl')

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('heart.csv')

@st.cache_data(show_spinner=False)
def compute_metrics(_model, df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_pred  = _model.predict(X_test)
    y_proba = _model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    cv5     = cross_val_score(_model, X, y, cv=5, scoring='accuracy')
    fi      = dict(zip(X.columns, _model.feature_importances_))
    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'cm':        cm,
        'fpr':       fpr, 'tpr': tpr,
        'roc_auc':   roc_auc,
        'cv5':       cv5,
        'fi':        fi,
        'y_test':    y_test.values,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'X_test':    X_test,
    }

model = load_model()
df    = load_data()
mets  = compute_metrics(model, df)

FEATURES = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']
FEAT_LABELS = {
    'age':'Age','sex':'Sex','cp':'Chest Pain','trestbps':'Blood Pressure',
    'chol':'Cholesterol','fbs':'Fasting BS','restecg':'ECG',
    'thalach':'Max Heart Rate','exang':'Exercise Angina',
    'oldpeak':'ST Depression','slope':'ST Slope','ca':'Major Vessels','thal':'Thalassemia'
}

# Plotly shared layout
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,22,40,0.6)',
    font=dict(color='#dde3f0', family='Outfit'),
    title_font_family='Syne',
    margin=dict(t=48, b=32, l=24, r=24),
)

# ─────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🏆 Created by RAHUL THAKUR</div>
  <h1>🫀 Heart<span>Guard</span> AI</h1>
  <p>Clinical-grade Heart Disease Risk Assessment · Decision Tree · UCI Heart Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  SIDEBAR  — Patient Input
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-size:2rem;">🫀</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#fff;">HeartGuard AI</div>
      <div class="name">RAHUL THAKUR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🩺 Patient Parameters")

    age      = st.slider("Age", 20, 80, 52, help="Patient age in years")
    sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female 👩" if x==0 else "Male 👨")
    cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                            format_func=lambda x:{0:"Typical Angina",1:"Atypical Angina",
                                                  2:"Non-anginal Pain",3:"Asymptomatic"}[x])
    trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 125)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 212)
    fbs      = st.radio("Fasting BS > 120 mg/dl", [0,1],
                        format_func=lambda x:"No" if x==0 else "Yes", horizontal=True)
    restecg  = st.selectbox("Resting ECG", [0,1,2],
                            format_func=lambda x:{0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    thalach  = st.slider("Max Heart Rate", 60, 220, 168)
    exang    = st.radio("Exercise Angina", [0,1],
                        format_func=lambda x:"No" if x==0 else "Yes", horizontal=True)
    oldpeak  = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    slope    = st.selectbox("ST Slope", [0,1,2],
                            format_func=lambda x:{0:"Downsloping",1:"Flat",2:"Upsloping"}[x])
    ca       = st.select_slider("Major Vessels (CA)", [0,1,2,3])
    thal     = st.selectbox("Thalassemia", [0,1,2,3],
                            format_func=lambda x:{0:"Unknown",1:"Normal",
                                                  2:"Fixed Defect",3:"Reversible Defect"}[x])

    st.markdown("---")
    predict_btn = st.button("🔬 Analyze Patient", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📂 Upload Custom Data")
    uploaded_file = st.file_uploader("Upload CSV (same schema)", type=["csv"])
    if uploaded_file:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(custom_df)} rows")
            df = custom_df
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("⚠️ For educational use only. Not a substitute for medical diagnosis.")


# ─────────────────────────────────────────────────────
#  KPI STRIP (always visible)
# ─────────────────────────────────────────────────────
total   = len(df)
disease = int(df['target'].sum())
healthy = total - disease
acc_pct = round(mets['accuracy'] * 100, 1)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-icon">👥</div>
    <div class="kpi-val">{total}</div>
    <div class="kpi-lbl">Total Patients</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">❤️‍🔥</div>
    <div class="kpi-val" style="color:#e63946">{disease}</div>
    <div class="kpi-lbl">Heart Disease</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">💚</div>
    <div class="kpi-val" style="color:#06d6a0">{healthy}</div>
    <div class="kpi-lbl">Healthy</div>
  </div>
  <div class="kpi">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-val">{acc_pct}%</div>
    <div class="kpi-lbl">Model Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict & Explain", "📊 Data Explorer",
                "📈 Visual Analytics", "🧠 Model Report", "💡 Health Guide", "ℹ️ About"])

tab_pred, tab_data, tab_vis, tab_model, tab_guide, tab_about = tabs


# ══════════════════════════════════════════════════════
#  TAB 1 — PREDICT & EXPLAIN
# ══════════════════════════════════════════════════════
with tab_pred:

    input_arr = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if predict_btn:
        prediction = model.predict(input_arr)[0]
        prob_arr   = model.predict_proba(input_arr)[0]
        risk_pct   = round(prob_arr[1] * 100, 1)
        safe_pct   = round(prob_arr[0] * 100, 1)

        # ── Result ──
        st.markdown('<div class="st-hdr">🧬 Prediction Result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-pos">
                  <div style="font-size:3rem">⚠️</div>
                  <div class="result-label" style="color:#e63946">Heart Disease Risk</div>
                  <div class="result-prob" style="color:#e63946">{risk_pct}%</div>
                  <div class="result-sub">Risk probability detected<br>Recommend specialist consultation</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-neg">
                  <div style="font-size:3rem">✅</div>
                  <div class="result-label" style="color:#06d6a0">Low Risk — Healthy</div>
                  <div class="result-prob" style="color:#06d6a0">{safe_pct}%</div>
                  <div class="result-sub">Parameters within acceptable range<br>Continue healthy lifestyle</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            # Probability donut
            fig_donut = go.Figure(go.Pie(
                values=[risk_pct, safe_pct],
                labels=["Disease Risk", "Healthy"],
                hole=.65,
                marker_colors=['#e63946', '#06d6a0'],
                textinfo='none',
            ))
            fig_donut.add_annotation(
                text=f"{risk_pct}%<br><span style='font-size:10px'>Risk</span>",
                x=.5, y=.5, showarrow=False,
                font=dict(size=18, color='#fff', family='Syne')
            )
            fig_donut.update_layout(**PLOT_LAYOUT, height=220,
                showlegend=True,
                legend=dict(orientation='h', y=-.15, font=dict(size=10)))
            st.plotly_chart(fig_donut, use_container_width=True)

        with r3:
            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x':[0,1],'y':[0,1]},
                number={'suffix':'%','font':{'color':'#fff','size':28,'family':'Syne'}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#6b7a99',
                            'tickfont':{'color':'#6b7a99','size':9}},
                    'bar':{'color':'#e63946' if prediction==1 else '#06d6a0','thickness':.28},
                    'bgcolor':'#0f1622',
                    'bordercolor':'rgba(0,0,0,0)',
                    'steps':[
                        {'range':[0,33], 'color':'rgba(6,214,160,0.1)'},
                        {'range':[33,66],'color':'rgba(255,209,102,0.1)'},
                        {'range':[66,100],'color':'rgba(230,57,70,0.12)'},
                    ],
                },
                title={'text':'Risk Score','font':{'size':12,'color':'#6b7a99','family':'Outfit'}}
            ))
            fig_gauge.update_layout(**PLOT_LAYOUT, height=220)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature contribution (manual LIME-style) ──
        st.markdown('<div class="st-hdr">🔎 Feature Contribution to Prediction</div>', unsafe_allow_html=True)

        fi_vals  = model.feature_importances_
        contrib  = fi_vals * input_arr[0]
        feat_df  = pd.DataFrame({
            'Feature': [FEAT_LABELS[f] for f in FEATURES],
            'Importance': fi_vals,
            'Your Value': input_arr[0],
            'Contribution': contrib
        }).sort_values('Contribution', ascending=True)

        fig_contrib = go.Figure(go.Bar(
            y=feat_df['Feature'],
            x=feat_df['Contribution'],
            orientation='h',
            marker_color=['#e63946' if v > 0 else '#06d6a0' for v in feat_df['Contribution']],
            text=[f"{v:.3f}" for v in feat_df['Contribution']],
            textposition='outside',
        ))
        fig_contrib.update_layout(**PLOT_LAYOUT, height=380,
            xaxis_title='Contribution Score',
            title='How each feature influenced this prediction')
        st.plotly_chart(fig_contrib, use_container_width=True)

        # ── Patient Summary Table ──
        st.markdown('<div class="st-hdr">📋 Patient Summary Report</div>', unsafe_allow_html=True)

        summary_data = {
            "Parameter": ["Age","Sex","Chest Pain Type","Resting BP","Cholesterol",
                           "Fasting Blood Sugar","Resting ECG","Max Heart Rate",
                           "Exercise Angina","ST Depression","ST Slope","Major Vessels","Thalassemia"],
            "Your Value": [
                age,
                "Male" if sex==1 else "Female",
                {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}[cp],
                f"{trestbps} mm Hg", f"{chol} mg/dl",
                "Yes" if fbs else "No",
                {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[restecg],
                f"{thalach} bpm", "Yes" if exang else "No",
                f"{oldpeak}",
                {0:"Downsloping",1:"Flat",2:"Upsloping"}[slope],
                ca,
                {0:"Unknown",1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[thal]
            ],
            "Dataset Avg": [
                round(df['age'].mean(),1),
                "Male" if round(df['sex'].mean()) else "Female",
                "—", f"{df['trestbps'].mean():.1f} mm Hg",
                f"{df['chol'].mean():.1f} mg/dl","—","—",
                f"{df['thalach'].mean():.1f} bpm","—",
                f"{df['oldpeak'].mean():.2f}","—",
                f"{df['ca'].mean():.1f}","—"
            ],
            "Feature Weight": [f"{fi_vals[i]:.3f}" for i in range(len(FEATURES))]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # ── Download report ──
        report_csv = pd.DataFrame(summary_data).to_csv(index=False)
        st.download_button(
            label="⬇️ Download Patient Report (CSV)",
            data=report_csv,
            file_name=f"heartguard_report_age{age}.csv",
            mime="text/csv"
        )

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;background:var(--card);border-radius:1.2rem;border:1px solid var(--border2);">
          <div style="font-size:4rem;margin-bottom:1rem;">🫀</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#fff;">
            Ready to Analyze
          </div>
          <div style="color:var(--muted);margin-top:.5rem;font-size:.9rem;">
            Adjust parameters in the sidebar and click <b style="color:#e63946">Analyze Patient</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show some sample similar patients
        st.markdown('<div class="st-hdr">🔬 Sample Patients from Dataset</div>', unsafe_allow_html=True)
        sample = df.sample(5, random_state=7).copy()
        sample['Diagnosis'] = sample['target'].map({0:'✅ Healthy', 1:'⚠️ Disease'})
        st.dataframe(sample, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="st-hdr">📊 Dataset Explorer</div>', unsafe_allow_html=True)

    d1, d2 = st.columns([3,1])
    with d1:
        search_query = st.text_input("🔍 Filter rows (e.g. age > 55)", "")
        show_n = st.slider("Rows to display", 5, 100, 15)
        display_df = df.copy()
        if search_query:
            try:
                display_df = df.query(search_query)
                st.caption(f"Showing {len(display_df)} matching rows")
            except:
                st.warning("Invalid filter. Try: `age > 55` or `target == 1`")
        display_df_show = display_df.head(show_n).copy()
        display_df_show['Diagnosis'] = display_df_show['target'].map({0:'✅ Healthy','1':'⚠️ Disease',1:'⚠️ Disease'})
        st.dataframe(display_df_show, use_container_width=True)
    with d2:
        st.markdown("#### 📌 Dataset Info")
        st.metric("Rows",    df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Null Values", int(df.isnull().sum().sum()))
        st.metric("Duplicates",  int(df.duplicated().sum()))
        st.metric("Disease %", f"{round(df['target'].mean()*100,1)}%")

    # Download filtered
    csv_dl = display_df.to_csv(index=False)
    st.download_button("⬇️ Download Filtered Data", csv_dl,
                       "filtered_heart_data.csv", "text/csv")

    st.markdown('<div class="st-hdr">📐 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown('<div class="st-hdr">📦 Feature Distribution</div>', unsafe_allow_html=True)
    col_sel, style_sel = st.columns(2)
    feat_pick  = col_sel.selectbox("Feature", FEATURES, format_func=lambda x: FEAT_LABELS[x])
    chart_type = style_sel.selectbox("Chart Type", ["Histogram","Box Plot","Violin","Strip"])

    df_plot = df.copy()
    df_plot['Diagnosis'] = df_plot['target'].map({0:'Healthy', 1:'Heart Disease'})
    colors = {'Healthy':'#06d6a0','Heart Disease':'#e63946'}

    if chart_type == "Histogram":
        fig = px.histogram(df_plot, x=feat_pick, color='Diagnosis',
                           color_discrete_map=colors, barmode='overlay',
                           template='plotly_dark',
                           labels={feat_pick: FEAT_LABELS[feat_pick]})
    elif chart_type == "Box Plot":
        fig = px.box(df_plot, x='Diagnosis', y=feat_pick, color='Diagnosis',
                     color_discrete_map=colors, template='plotly_dark',
                     labels={feat_pick: FEAT_LABELS[feat_pick]})
    elif chart_type == "Violin":
        fig = px.violin(df_plot, x='Diagnosis', y=feat_pick, color='Diagnosis',
                        color_discrete_map=colors, box=True,
                        template='plotly_dark',
                        labels={feat_pick: FEAT_LABELS[feat_pick]})
    else:
        fig = px.strip(df_plot, x='Diagnosis', y=feat_pick, color='Diagnosis',
                       color_discrete_map=colors, template='plotly_dark',
                       labels={feat_pick: FEAT_LABELS[feat_pick]})
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
#  TAB 3 — VISUAL ANALYTICS
# ══════════════════════════════════════════════════════
with tab_vis:
    df_vis = df.copy()
    df_vis['Diagnosis'] = df_vis['target'].map({0:'Healthy', 1:'Heart Disease'})
    CMAP = {'Healthy':'#06d6a0','Heart Disease':'#e63946'}

    st.markdown('<div class="st-hdr">🔢 Disease Overview</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)

    with v1:
        # Donut breakdown
        cnts = df_vis['Diagnosis'].value_counts().reset_index()
        cnts.columns = ['Diagnosis', 'count']
        fig_pie = px.pie(cnts, values='count', names='Diagnosis', hole=.55,
                         color='Diagnosis', color_discrete_map=CMAP,
                         template='plotly_dark', title='Healthy vs Heart Disease')
        fig_pie.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with v2:
        # Age distribution
        fig_age = px.histogram(df_vis, x='age', color='Diagnosis',
                               color_discrete_map=CMAP, nbins=20,
                               template='plotly_dark', barmode='overlay',
                               title='Age Distribution by Diagnosis')
        fig_age.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig_age, use_container_width=True)

    # Scatter matrix (4 key features)
    st.markdown('<div class="st-hdr">🌐 Multi-feature Scatter Matrix</div>', unsafe_allow_html=True)
    scatter_feats = st.multiselect(
        "Select features (3–5 recommended)",
        FEATURES, default=['age','thalach','chol','oldpeak'],
        format_func=lambda x: FEAT_LABELS[x]
    )
    if len(scatter_feats) >= 2:
        fig_matrix = px.scatter_matrix(
            df_vis, dimensions=scatter_feats, color='Diagnosis',
            color_discrete_map=CMAP, template='plotly_dark',
            labels={f: FEAT_LABELS[f] for f in FEATURES}
        )
        fig_matrix.update_traces(diagonal_visible=False, marker_size=3)
        fig_matrix.update_layout(**PLOT_LAYOUT, height=550)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Bubble chart
    st.markdown('<div class="st-hdr">🫧 Age · BP · Cholesterol Bubble Chart</div>', unsafe_allow_html=True)
    fig_bubble = px.scatter(
        df_vis, x='age', y='trestbps', size='chol', color='Diagnosis',
        color_discrete_map=CMAP, template='plotly_dark',
        hover_data=['thalach','cp'], size_max=22,
        labels={'age':'Age','trestbps':'Resting BP','chol':'Cholesterol'},
        title='Bubble size = Cholesterol level'
    )
    fig_bubble.update_layout(**PLOT_LAYOUT, height=420)
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="st-hdr">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df.corr(numeric_only=True).round(2)
    fig_heat = px.imshow(corr, text_auto=True, template='plotly_dark',
                          color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                          title='Feature Correlation Matrix')
    fig_heat.update_layout(**PLOT_LAYOUT, height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Chest pain & gender
    st.markdown('<div class="st-hdr">📊 Clinical Breakdowns</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        df_cp = df_vis.copy()
        df_cp['Chest Pain'] = df_cp['cp'].map(
            {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal",3:"Asymptomatic"})
        fig_cp = px.histogram(df_cp, x='Chest Pain', color='Diagnosis',
                              color_discrete_map=CMAP, barmode='group',
                              template='plotly_dark', title='Chest Pain Type vs Diagnosis')
        fig_cp.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig_cp, use_container_width=True)
    with c2:
        df_sex = df_vis.copy()
        df_sex['Gender'] = df_sex['sex'].map({0:'Female',1:'Male'})
        fig_sex = px.histogram(df_sex, x='Gender', color='Diagnosis',
                               color_discrete_map=CMAP, barmode='group',
                               template='plotly_dark', title='Gender Distribution')
        fig_sex.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig_sex, use_container_width=True)

    # 3D Scatter
    st.markdown('<div class="st-hdr">🧊 3D Feature Space</div>', unsafe_allow_html=True)
    ax_x = st.selectbox("X axis", FEATURES, index=0, format_func=lambda x:FEAT_LABELS[x], key='3dx')
    ax_y = st.selectbox("Y axis", FEATURES, index=7, format_func=lambda x:FEAT_LABELS[x], key='3dy')
    ax_z = st.selectbox("Z axis", FEATURES, index=9, format_func=lambda x:FEAT_LABELS[x], key='3dz')
    fig_3d = px.scatter_3d(df_vis, x=ax_x, y=ax_y, z=ax_z,
                            color='Diagnosis', color_discrete_map=CMAP,
                            template='plotly_dark', opacity=.75,
                            labels={ax_x:FEAT_LABELS[ax_x],
                                    ax_y:FEAT_LABELS[ax_y],
                                    ax_z:FEAT_LABELS[ax_z]})
    fig_3d.update_layout(**PLOT_LAYOUT, height=500)
    st.plotly_chart(fig_3d, use_container_width=True)


# ══════════════════════════════════════════════════════
#  TAB 4 — MODEL REPORT
# ══════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="st-hdr">🎯 Performance Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in zip(
        [m1, m2, m3, m4],
        ["Accuracy","Precision","Recall","F1 Score"],
        [mets['accuracy'], mets['precision'], mets['recall'], mets['f1']],
        ['#4895ef','#ffd166','#e63946','#06d6a0']
    ):
        col.markdown(f"""
        <div class="kpi" style="border-top: 2px solid {color}">
          <div class="kpi-val" style="color:{color}">{val:.1%}</div>
          <div class="kpi-lbl">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="st-hdr">📉 ROC Curve</div>', unsafe_allow_html=True)
    roc1, roc2 = st.columns(2)

    with roc1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=mets['fpr'], y=mets['tpr'],
            mode='lines', name=f"AUC = {mets['roc_auc']:.3f}",
            line=dict(color='#e63946', width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='#6b7a99', dash='dash', width=1),
            name='Random Baseline', showlegend=True
        ))
        fig_roc.update_layout(**PLOT_LAYOUT, height=340,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title=f'ROC Curve — AUC: {mets["roc_auc"]:.3f}')
        st.plotly_chart(fig_roc, use_container_width=True)

    with roc2:
        # Confusion matrix
        cm = mets['cm']
        cm_labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Predicted: Healthy','Predicted: Disease'],
            y=['Actual: Healthy','Actual: Disease'],
            colorscale=[[0,'#0f1622'],[0.5,'#7d1128'],[1,'#e63946']],
            text=cm, texttemplate="%{text}",
            showscale=False
        ))
        fig_cm.update_layout(**PLOT_LAYOUT, height=340, title='Confusion Matrix')
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    st.markdown('<div class="st-hdr">⚖️ Feature Importance</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({
        'Feature': [FEAT_LABELS[f] for f in FEATURES],
        'Importance': [mets['fi'][f] for f in FEATURES]
    }).sort_values('Importance', ascending=True)

    fig_fi = go.Figure(go.Bar(
        y=fi_df['Feature'], x=fi_df['Importance'],
        orientation='h',
        marker=dict(
            color=fi_df['Importance'],
            colorscale=[[0,'#131c2e'],[0.5,'#7d1128'],[1,'#e63946']],
            showscale=False
        ),
        text=[f"{v:.3f}" for v in fi_df['Importance']],
        textposition='outside'
    ))
    fig_fi.update_layout(**PLOT_LAYOUT, height=380, xaxis_title='Gini Importance',
                          title='Feature Importance (Gini Criterion)')
    st.plotly_chart(fig_fi, use_container_width=True)

    # Cross-validation
    st.markdown('<div class="st-hdr">🔁 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    cv5 = mets['cv5']
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)], y=cv5,
        marker_color=['#e63946' if v == cv5.min() else '#4895ef' for v in cv5],
        text=[f"{v:.2%}" for v in cv5], textposition='outside'
    ))
    fig_cv.add_hline(y=cv5.mean(), line_dash='dash',
                     line_color='#ffd166', annotation_text=f"Mean: {cv5.mean():.2%}")
    fig_cv.update_layout(**PLOT_LAYOUT, height=300,
                          yaxis=dict(range=[0,1], tickformat='.0%'),
                          title='Cross-validation Accuracy per Fold')
    st.plotly_chart(fig_cv, use_container_width=True)

    cv_col1, cv_col2, cv_col3 = st.columns(3)
    cv_col1.metric("CV Mean", f"{cv5.mean():.2%}")
    cv_col2.metric("CV Std",  f"±{cv5.std():.2%}")
    cv_col3.metric("Best Fold", f"{cv5.max():.2%}")

    # Decision Tree text rules
    st.markdown('<div class="st-hdr">🌳 Decision Tree Rules (Top Depth)</div>', unsafe_allow_html=True)
    try:
        tree_rules = export_text(model, feature_names=FEATURES, max_depth=3)
        st.code(tree_rules, language='text')
    except Exception as e:
        st.info(f"Tree export: {e}")

    # Model params table
    st.markdown('<div class="st-hdr">⚙️ Model Hyperparameters</div>', unsafe_allow_html=True)
    params = model.get_params()
    params_df = pd.DataFrame(list(params.items()), columns=['Parameter','Value'])
    st.dataframe(params_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  TAB 5 — HEALTH GUIDE
# ══════════════════════════════════════════════════════
with tab_guide:
    st.markdown('<div class="st-hdr">💡 Understand Your Risk Factors</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#e63946;margin-top:0">🩺 Key Risk Factors</h4>

  <div class="rec-card" style="border-color:#e63946">
    <b>Cholesterol > 240 mg/dl</b><br>
    High LDL cholesterol deposits plaque in coronary arteries, raising heart attack risk significantly.
  </div>
  <div class="rec-card" style="border-color:#ffd166">
    <b>Blood Pressure > 140 mm Hg</b><br>
    Chronic hypertension damages artery walls and forces the heart to work harder over time.
  </div>
  <div class="rec-card" style="border-color:#e63946">
    <b>Fasting Blood Sugar > 120 mg/dl</b><br>
    Elevated glucose (diabetes marker) doubles the risk of developing coronary artery disease.
  </div>
  <div class="rec-card" style="border-color:#4895ef">
    <b>Thalassemia (Reversible Defect)</b><br>
    Indicates reduced blood flow during stress — one of the strongest predictors in this dataset.
  </div>
  <div class="rec-card" style="border-color:#ffd166">
    <b>ST Depression (Oldpeak > 2.0)</b><br>
    Exercise-induced ST depression suggests myocardial ischemia — needs cardiology review.
  </div>
</div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#06d6a0;margin-top:0">✅ Heart-Healthy Habits</h4>

  <div class="rec-card" style="border-color:#06d6a0">
    <b>🏃 Regular Aerobic Exercise</b><br>
    At least 150 min/week of moderate activity reduces cardiovascular risk by up to 35%.
  </div>
  <div class="rec-card" style="border-color:#06d6a0">
    <b>🥗 Mediterranean Diet</b><br>
    Rich in olive oil, fish, nuts, and vegetables — lowers LDL and systemic inflammation.
  </div>
  <div class="rec-card" style="border-color:#06d6a0">
    <b>🚭 No Smoking</b><br>
    Quitting smoking reduces coronary disease risk by half within 1 year of cessation.
  </div>
  <div class="rec-card" style="border-color:#06d6a0">
    <b>😴 7–9 Hours Sleep</b><br>
    Sleep deprivation raises blood pressure and inflammation markers significantly.
  </div>
  <div class="rec-card" style="border-color:#06d6a0">
    <b>🧘 Stress Management</b><br>
    Chronic cortisol elevation accelerates arterial plaque formation and arrhythmias.
  </div>
</div>
        """, unsafe_allow_html=True)

    # Normal ranges table
    st.markdown('<div class="st-hdr">📏 Clinical Normal Ranges</div>', unsafe_allow_html=True)
    ranges_df = pd.DataFrame({
        'Parameter': ['Blood Pressure','Cholesterol','Fasting Blood Sugar',
                      'Heart Rate (Resting)','Max Heart Rate','BMI'],
        'Normal Range': ['< 120/80 mm Hg','< 200 mg/dl','70–99 mg/dl',
                         '60–100 bpm','220 – Age (bpm)','18.5–24.9'],
        'At Risk':      ['120–139 / 80–89','200–239 mg/dl','100–125 mg/dl',
                         '> 100 bpm','< 85% Max HR','25–29.9'],
        'Danger Zone':  ['≥ 140/90 mm Hg','≥ 240 mg/dl','≥ 126 mg/dl',
                         '> 120 bpm (rest)','< 70% Max HR','≥ 30']
    })
    st.dataframe(ranges_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  TAB 6 — ABOUT
# ══════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="st-hdr">ℹ️ About This Project</div>', unsafe_allow_html=True)

    ab1, ab2 = st.columns([2, 1])
    with ab1:
        st.markdown("""
**HeartGuard AI** is a full-stack machine learning web application for predicting
heart disease risk using clinical parameters from the UCI Heart Disease dataset.

Built as a portfolio project demonstrating end-to-end ML deployment, interactive
data visualization, and clinical decision-support system design.

---

### 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn Decision Tree |
| Web Framework | Streamlit |
| Visualization | Plotly (interactive) |
| Data Processing | Pandas · NumPy |
| Model Persistence | Joblib |
| Evaluation | ROC-AUC · F1 · CV |

### 🧠 Model Configuration
| Hyperparameter | Value |
|----------------|-------|
| Algorithm | Decision Tree Classifier |
| Criterion | Entropy (information gain) |
| Max Depth | 6 |
| Min Samples Split | 4 |
| Min Samples Leaf | 4 |
| Test Accuracy | 81.97% |
| ROC-AUC | 0.826 |
| CV Mean (5-fold) | 75.2% |

### 📦 Dataset
- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **Patients:** 303
- **Features:** 13 clinical attributes
- **Target:** Binary (0 = No Disease, 1 = Disease)
- **Prevalence:** 54.5% disease positive

### ⚠️ Disclaimer
This application is **strictly for educational and portfolio demonstration purposes**.
It should never be used as a substitute for professional medical evaluation or clinical diagnosis.
        """)

    with ab2:
        st.markdown(f"""
<div style="background:linear-gradient(145deg,#100a1f,#1a0510);border:1px solid rgba(230,57,70,.3);
     border-radius:1.4rem;padding:2rem;text-align:center;">
  <div style="font-size:3rem">🫀</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#fff;margin:.4rem 0">
    HeartGuard AI
  </div>
  <div style="color:#6b7a99;font-size:.8rem">v2.0.0 · Advanced Edition</div>
  <hr style="border-color:rgba(255,255,255,.08);margin:1.2rem 0">
  <div style="color:#6b7a99;font-size:.75rem;letter-spacing:1px;text-transform:uppercase">
    Created by
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
       color:#ffd166;margin:.3rem 0;letter-spacing:.5px">
    RAHUL THAKUR
  </div>
  <hr style="border-color:rgba(255,255,255,.08);margin:1.2rem 0">
  <div style="font-size:.78rem;color:#6b7a99;line-height:2">
    🎯 Test Accuracy: <b style="color:#e63946">81.97%</b><br>
    📈 ROC-AUC: <b style="color:#4895ef">{mets['roc_auc']:.3f}</b><br>
    🔁 CV Mean: <b style="color:#06d6a0">{mets['cv5'].mean():.2%}</b><br>
    📋 F1 Score: <b style="color:#ffd166">{mets['f1']:.3f}</b><br>
    👥 Dataset: <b style="color:#fff">303 patients</b>
  </div>
</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🫀 HeartGuard AI &nbsp;·&nbsp; Built with Python & Streamlit
  &nbsp;·&nbsp; Created by <span>RAHUL THAKUR</span>
  &nbsp;·&nbsp; Educational use only — not a medical device
</div>
""", unsafe_allow_html=True)