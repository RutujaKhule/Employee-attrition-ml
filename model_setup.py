"""
Employee Attrition Detection - Enhanced Streamlit Web App
=========================================================
Run with: streamlit run model_setup.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import hashlib
import time
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ · Employee Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# USER DATABASE (demo credentials)
# ─────────────────────────────────────────────
USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Admin User",
        "role": "Administrator",
        "avatar": "👑",
    },
    "analyst": {
        "password": hashlib.sha256("analyst123".encode()).hexdigest(),
        "name": "Data Analyst",
        "role": "HR Analyst",
        "avatar": "📊",
    },
    "demo": {
        "password": hashlib.sha256("demo".encode()).hexdigest(),
        "name": "Demo User",
        "role": "Guest",
        "avatar": "🎯",
    },
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-deep:     #06080f;
    --bg-mid:      #0d1117;
    --bg-card:     #111827;
    --bg-glass:    rgba(17,24,39,0.85);
    --border:      rgba(56,189,248,0.15);
    --border-hot:  rgba(56,189,248,0.45);
    --indigo:      #6366f1;
    --sky:         #38bdf8;
    --rose:        #fb7185;
    --emerald:     #34d399;
    --amber:       #fbbf24;
    --text-1:      #f1f5f9;
    --text-2:      #94a3b8;
    --text-3:      #475569;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: var(--bg-deep);
    color: var(--text-1);
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 100%, rgba(56,189,248,0.08) 0%, transparent 55%),
                var(--bg-deep);
}

/* ── LOGIN PAGE ─── */
.login-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
}
.login-card {
    background: linear-gradient(145deg, rgba(17,24,39,0.95), rgba(13,17,23,0.98));
    border: 1px solid var(--border-hot);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    max-width: 440px;
    width: 100%;
    box-shadow: 0 0 60px rgba(56,189,248,0.08), 0 32px 64px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.login-card::before {
    content: '';
    position: absolute;
    top: -2px; left: 10%; right: 10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--sky), var(--indigo), transparent);
    border-radius: 100%;
}
.login-logo {
    font-size: 3.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}
.login-title {
    text-align: center;
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--sky) 0%, var(--indigo) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.login-sub {
    text-align: center;
    color: var(--text-3);
    font-size: 0.9rem;
    margin-bottom: 2rem;
}
.demo-hint {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: var(--text-2);
    font-family: 'Space Mono', monospace;
}
.demo-hint strong { color: var(--sky); }

/* ── HERO ─── */
.hero-bar {
    background: linear-gradient(100deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.8rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.hero-bar::after {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%236366f1' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-left h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, var(--sky) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-left p {
    color: var(--text-2);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 100px;
    padding: 0.4rem 1.1rem;
    font-size: 0.8rem;
    color: var(--sky);
    font-weight: 600;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

/* ── USER BADGE ─── */
.user-pill {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(56,189,248,0.1));
    border: 1px solid var(--border-hot);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
}
.user-pill .uname {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-1);
}
.user-pill .urole {
    font-size: 0.75rem;
    color: var(--sky);
    font-weight: 500;
    letter-spacing: 0.4px;
}

/* ── SECTION TITLE ─── */
.stitle {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--sky);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 2rem 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.stitle-num {
    background: linear-gradient(135deg, var(--indigo), var(--sky));
    color: #fff;
    border-radius: 6px;
    width: 26px; height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 800;
    flex-shrink: 0;
}

/* ── METRIC CARDS ─── */
.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.mcard {
    flex: 1;
    min-width: 130px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.mcard:hover { border-color: var(--border-hot); }
.mcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, var(--indigo));
    border-radius: 100% 100% 0 0;
}
.mcard .ml { font-size: 0.7rem; color: var(--text-3); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.4rem; }
.mcard .mv { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--text-1); line-height: 1; }
.mcard .ms { font-size: 0.75rem; color: var(--text-2); margin-top: 0.2rem; }

/* ── PRED CARDS ─── */
.pred-leave {
    background: linear-gradient(135deg, rgba(251,113,133,0.12), rgba(159,18,57,0.15));
    border: 1px solid rgba(251,113,133,0.4);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-stay {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(6,78,59,0.15));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.pred-label { font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.3rem; }
.pred-verdict { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; }
.pred-conf { font-family: 'Space Mono', monospace; font-size: 1rem; }

/* ── STAT CHIPS ─── */
.chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0.8rem 0; }
.chip {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: var(--sky);
    font-weight: 500;
}
.chip.rose {
    background: rgba(251,113,133,0.08);
    border-color: rgba(251,113,133,0.25);
    color: var(--rose);
}
.chip.emerald {
    background: rgba(52,211,153,0.08);
    border-color: rgba(52,211,153,0.25);
    color: var(--emerald);
}

/* ── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background: var(--bg-mid) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-1) !important; }

/* ── WIDGETS ─── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-1) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(120deg, var(--indigo), #818cf8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(120deg, #4338ca, #6d28d9) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px);
}

/* ── LOGOUT BUTTON ─── */
.logout-btn > button {
    background: linear-gradient(120deg, rgba(251,113,133,0.2), rgba(159,18,57,0.3)) !important;
    border: 1px solid rgba(251,113,133,0.4) !important;
    color: var(--rose) !important;
    width: 100% !important;
}
.logout-btn > button:hover {
    background: linear-gradient(120deg, rgba(251,113,133,0.35), rgba(159,18,57,0.45)) !important;
    box-shadow: 0 4px 16px rgba(251,113,133,0.3) !important;
}

.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-2) !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--indigo) !important;
    color: #fff !important;
}

/* ── PROGRESS BAR ─── */
.prob-bar-wrap { background: var(--bg-card); border-radius: 12px; padding: 1.2rem; border: 1px solid var(--border); margin-top: 1.2rem; }
.prob-bar-label { font-size: 0.75rem; color: var(--text-2); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.6rem; }
.prob-bar-track { background: rgba(255,255,255,0.05); border-radius: 100px; height: 10px; overflow: hidden; margin-bottom: 0.4rem; }
.prob-bar-fill-stay { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #34d399, #059669); transition: width 1s ease; }
.prob-bar-fill-leave { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #f43f5e, #be123c); transition: width 1s ease; }
.prob-bar-val { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: var(--text-2); }

/* ── TIMELINE ─── */
.activity-feed { border-left: 2px solid var(--border); padding-left: 1.2rem; margin-top: 0.5rem; }
.activity-item { position: relative; margin-bottom: 1rem; }
.activity-item::before {
    content: '';
    position: absolute;
    left: -1.55rem; top: 0.3rem;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--sky);
    border: 2px solid var(--bg-deep);
}
.activity-time { font-size: 0.7rem; color: var(--text-3); font-family: 'Space Mono', monospace; }
.activity-text { font-size: 0.85rem; color: var(--text-2); }

/* ── ALERT BOX ─── */
.info-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text-2);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-hot); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def login(username: str, password: str) -> bool:
    u = username.strip().lower()
    if u in USERS and USERS[u]["password"] == hash_pw(password):
        st.session_state["authenticated"] = True
        st.session_state["username"]      = u
        st.session_state["user_info"]     = USERS[u]
        st.session_state["login_time"]    = time.strftime("%H:%M on %d %b %Y")
        return True
    return False

def logout():
    for key in ["authenticated","username","user_info","login_time","model","acc","cm","report","feature_cols"]:
        st.session_state.pop(key, None)


# ═════════════════════════════════════════════
# LOGIN PAGE
# ═════════════════════════════════════════════
if not st.session_state.get("authenticated"):
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown("""
        <div class="login-card">
            <div class="login-logo">🧠</div>
            <div class="login-title">AttritionIQ</div>
            <div class="login-sub">Employee Intelligence Platform · Sign in to continue</div>
            <div class="demo-hint">
                <strong>Demo accounts:</strong><br>
                admin / admin123 &nbsp;·&nbsp; analyst / analyst123 &nbsp;·&nbsp; demo / demo
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username…")
            password = st.text_input("Password", type="password", placeholder="Enter password…")
            submitted = st.form_submit_button("🔐  Sign In", use_container_width=True)

        if submitted:
            if login(username, password):
                st.success(f"Welcome back, {USERS[username.lower()]['name']}! 👋")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error("❌ Incorrect username or password. Please try again.")

    st.stop()


# ═════════════════════════════════════════════
# SIDEBAR (authenticated)
# ═════════════════════════════════════════════
user   = st.session_state["user_info"]
uname  = st.session_state["username"]

with st.sidebar:
    st.markdown(f"""
    <div class="user-pill">
        <div style="font-size:2rem;margin-bottom:0.3rem">{user['avatar']}</div>
        <div class="uname">{user['name']}</div>
        <div class="urole">{user['role']}</div>
        <div style="font-size:0.7rem;color:#475569;margin-top:0.3rem">
            Signed in at {st.session_state.get('login_time','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Settings")
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random State", 0, 999, 42, 1)
    show_raw = st.checkbox("Show DataFrame info", value=False)

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV (MFG10YearTerminationData)",
        type=["csv"],
    )
    # FIX 4: Auto-uncheck sample data when a file is uploaded
    use_sample = st.checkbox("Use synthetic sample data", value=(uploaded_file is None))

    st.markdown("---")
    st.markdown("### 📋 Session Activity")
    st.markdown("""
    <div class="activity-feed">
        <div class="activity-item">
            <div class="activity-time">Just now</div>
            <div class="activity-text">Session started</div>
        </div>
        <div class="activity-item">
            <div class="activity-time">—</div>
            <div class="activity-text">Awaiting model training</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("🚪  Sign Out", use_container_width=True):
        logout()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<br><small style='color:#1e293b'>AttritionIQ v2.0 · Built with Streamlit & scikit-learn</small>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════
# HERO BAR
# ═════════════════════════════════════════════
st.markdown(f"""
<div class="hero-bar">
    <div class="hero-left">
        <h1>🧠 AttritionIQ Dashboard</h1>
        <p>Predict employee attrition with Logistic Regression. Explore, analyse, train, and predict — all in one place.</p>
    </div>
    <div style="display:flex;flex-direction:column;gap:0.5rem;align-items:flex-end">
        <div class="hero-badge">✦ ML-Powered</div>
        <div class="hero-badge">👤 {user['name']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────
PSTYLE = {
    "axes.facecolor"  : "#111827",
    "figure.facecolor": "#111827",
    "axes.edgecolor"  : "#1e293b",
    "axes.labelcolor" : "#94a3b8",
    "xtick.color"     : "#64748b",
    "ytick.color"     : "#64748b",
    "text.color"      : "#e2e8f0",
    "grid.color"      : "#1e293b",
    "axes.titlecolor" : "#f1f5f9",
}
PAL = ["#6366f1", "#fb7185"]


# ─────────────────────────────────────────────
# GENERATE SAMPLE DATA
# ─────────────────────────────────────────────
def generate_sample_data(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    departments = ["IT", "HR", "Finance", "Operations", "Sales", "Engineering"]
    genders     = ["M", "F"]
    job_titles  = ["Analyst", "Manager", "Director", "Associate", "Executive"]
    statuses    = ["TERMINATED", "TERMINATED", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"]

    df = pd.DataFrame({
        "recorddate_key"     : pd.date_range("2014-01-01", periods=n, freq="10h").strftime("%Y%m%d").astype(int),
        "birthdate_key"      : rng.integers(19500101, 19991231, n),
        "orighiredate_key"   : rng.integers(19800101, 20100101, n),
        "terminationdate_key": rng.integers(20100101, 20200101, n),
        "age"                : rng.integers(22, 62, n),
        "length_of_service"  : rng.integers(1, 35, n),
        "city_name"          : rng.choice(["Vancouver", "Toronto", "Calgary", "Ottawa"], n),
        "department_name"    : rng.choice(departments, n),
        "job_title"          : rng.choice(job_titles, n),
        "store_name"         : rng.integers(1, 50, n),
        "gender_short"       : rng.choice(genders, n),
        "termreason_desc"    : rng.choice(["Resigned", "Layoff", "Retired", "N/A"], n),
        "termtype_desc"      : rng.choice(["Voluntary", "Involuntary", "N/A"], n),
        "STATUS"             : rng.choice(statuses, n),
        "BUSINESS_UNIT"      : rng.choice(["HEADOFFICE", "STORES"], n),
        "STATUS_YEAR"        : rng.integers(2010, 2020, n),
    })
    df["attrition"] = (df["STATUS"] != "ACTIVE").astype(int)
    return df


# ═════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📦 Dataset", "📊 Explore", "🤖 Train & Results", "🔮 Predict", "ℹ️ About"]
)

# ─────────────────────────────────────────────
# LOAD DATA (shared)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    with st.spinner("Reading CSV…"):
        df_raw = load_csv(uploaded_file)
elif use_sample:
    df_raw = generate_sample_data(800)
else:
    st.warning("⚠️ Upload a CSV or enable sample data in the sidebar.")
    st.stop()


# ═════════════════════════════════════════════
# TAB 1 — DATASET
# ═════════════════════════════════════════════
with tab1:
    src_label = "Uploaded file" if uploaded_file else "Synthetic sample"
    st.markdown(f"""
    <div class="stitle"><span class="stitle-num">1</span> Dataset Overview</div>
    <div class="chip-row">
        <span class="chip">📁 {src_label}</span>
        <span class="chip emerald">✓ {df_raw.shape[0]:,} rows</span>
        <span class="chip">📐 {df_raw.shape[1]} columns</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Preview (first 10 rows)", expanded=True):
        st.dataframe(df_raw.head(10), use_container_width=True)

    if show_raw:
        with st.expander("🗂 Column Info"):
            buf = pd.DataFrame({
                "Column"   : df_raw.columns,
                "Dtype"    : df_raw.dtypes.values,
                "Non-Null" : df_raw.notnull().sum().values,
                "Nulls"    : df_raw.isnull().sum().values,
                "Unique"   : df_raw.nunique().values,
            })
            st.dataframe(buf, use_container_width=True)

    st.markdown('<div class="stitle"><span class="stitle-num">2</span> Quick Stats</div>', unsafe_allow_html=True)
    num_df = df_raw.select_dtypes(include=np.number)
    st.dataframe(num_df.describe().round(2), use_container_width=True)


# ═════════════════════════════════════════════
# PREPROCESSING (shared between tabs)
# ═════════════════════════════════════════════

# FIX 1: Drop extra columns present in the real CSV to avoid leakage/errors
COLS_DROP = [
    "recorddate_key", "birthdate_key", "orighiredate_key",
    "terminationdate_key", "store_name", "city_name", "term_reason", "STATUS",
    "EmployeeID", "gender_full", "termreason_desc", "termtype_desc",
]
cols_drop_existing = [c for c in COLS_DROP if c in df_raw.columns]

df = df_raw.copy()

# FIX 2: Build target using latest record per employee + robust STATUS comparison
if "attrition" not in df.columns:
    if "STATUS" in df.columns:
        # Detect employee ID column
        id_col = None
        for candidate in ["EmployeeID", "employeeid", "employee_id", "emp_id"]:
            if candidate in df.columns:
                id_col = candidate
                break

        if id_col is not None:
            # Keep only the latest STATUS_YEAR row per employee to avoid duplicate rows
            sort_col = "STATUS_YEAR" if "STATUS_YEAR" in df.columns else df.columns[0]
            df = (
                df.sort_values(sort_col)
                  .groupby(id_col, sort=False)
                  .last()
                  .reset_index()
            )

        # Use .upper() for robust matching against both "ACTIVE" and "Active"
        active_mask = df["STATUS"].str.strip().str.upper() == "ACTIVE"
        df["attrition"] = (~active_mask).astype(int)
    else:
        st.error("No 'STATUS' or 'attrition' column found.")
        st.stop()

if len(df["attrition"].unique()) < 2:
    st.error("Target column has only one class. Check your dataset.")
    st.stop()

df.drop(columns=cols_drop_existing, errors="ignore", inplace=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if "attrition" in num_cols:
    num_cols.remove("attrition")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop(columns=["attrition"], errors="ignore")
y = df["attrition"]

n_left = int((y == 1).sum())
n_stay = int((y == 0).sum())
total  = n_left + n_stay

# Build a deduplicated version of df_raw for visualisations
# (same dedup logic so charts match the model's data)
if "EmployeeID" in df_raw.columns and "STATUS_YEAR" in df_raw.columns:
    df_viz = (
        df_raw.sort_values("STATUS_YEAR")
              .groupby("EmployeeID", sort=False)
              .last()
              .reset_index()
    )
else:
    df_viz = df_raw.copy()

# Attach attrition label to viz dataframe
if "attrition" not in df_viz.columns:
    if "STATUS" in df_viz.columns:
        df_viz["attrition"] = (
            df_viz["STATUS"].str.strip().str.upper() != "ACTIVE"
        ).astype(int)


# ═════════════════════════════════════════════
# TAB 2 — EXPLORE
# ═════════════════════════════════════════════
with tab2:
    st.markdown('<div class="stitle"><span class="stitle-num">3</span> Target Distribution</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Employees", f"{total:,}",              "—",                      "--indigo"),
        ("Stayed",          f"{n_stay:,}",             f"{n_stay/total*100:.1f}%","--emerald"),
        ("Left",            f"{n_left:,}",             f"{n_left/total*100:.1f}%","--rose"),
        ("Attrition Rate",  f"{n_left/total*100:.1f}%","of workforce",            "--amber"),
    ]
    for col, (lbl, val, sub, accent) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="mcard" style="--accent: var({accent})">
                <div class="ml">{lbl}</div>
                <div class="mv">{val}</div>
                <div class="ms">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle"><span class="stitle-num">4</span> Visual Analysis</div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)

    # Plot 1: Age distribution
    with pc1:
        st.markdown("**Age Distribution**")
        if "age" in df_viz.columns and "attrition" in df_viz.columns:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                for label, color, lname in zip([0, 1], PAL, ["Stayed", "Left"]):
                    ax.hist(df_viz[df_viz["attrition"] == label]["age"], bins=18,
                            alpha=0.75, color=color, label=lname, edgecolor="none")
                ax.set_xlabel("Age"); ax.set_ylabel("Count")
                ax.set_title("Age by Attrition", fontweight="bold")
                ax.legend(framealpha=0.15, fontsize=8)
                ax.grid(axis="y", alpha=0.2)
                fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.markdown('<div class="info-box">No <code>age</code> column found.</div>', unsafe_allow_html=True)

    # Plot 2: Department attrition rate
    with pc2:
        st.markdown("**Attrition by Department**")
        dept_col = next((c for c in ["department_name", "Department", "dept"] if c in df_viz.columns), None)
        attr_col = "attrition" if "attrition" in df_viz.columns else None
        if dept_col and attr_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                dept_attr = df_viz.groupby(dept_col)[attr_col].mean().sort_values(ascending=True)
                colors_bar = [PAL[1] if v > dept_attr.mean() else PAL[0] for v in dept_attr.values]
                bars = ax.barh(dept_attr.index, dept_attr.values * 100,
                               color=colors_bar, edgecolor="none", height=0.65)
                ax.set_xlabel("Attrition Rate (%)")
                ax.set_title("Rate by Department", fontweight="bold")
                ax.grid(axis="x", alpha=0.2)
                ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#64748b", fontsize=7.5)
                ax.axvline(dept_attr.mean() * 100, color="#fbbf24", linewidth=1,
                           linestyle="--", alpha=0.6, label="Avg")
                ax.legend(framealpha=0.1, fontsize=8)
                fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.markdown('<div class="info-box">No department column found in dataset.</div>', unsafe_allow_html=True)

    # Plot 3: Service length
    with pc3:
        st.markdown("**Years of Service**")
        svc_col = next((c for c in ["length_of_service", "tenure", "years_service"] if c in df_viz.columns), None)
        if svc_col and attr_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                for label, color, lname in zip([0, 1], PAL, ["Stayed", "Left"]):
                    ax.hist(df_viz[df_viz["attrition"] == label][svc_col], bins=18,
                            alpha=0.75, color=color, label=lname, edgecolor="none")
                ax.set_xlabel("Years of Service"); ax.set_ylabel("Count")
                ax.set_title("Tenure by Attrition", fontweight="bold")
                ax.legend(framealpha=0.15, fontsize=8)
                ax.grid(axis="y", alpha=0.2)
                fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.markdown('<div class="info-box">No service length column found.</div>', unsafe_allow_html=True)

    # Gender & Business Unit
    st.markdown('<div class="stitle"><span class="stitle-num">5</span> Gender & Business Unit</div>', unsafe_allow_html=True)
    gc1, gc2 = st.columns(2)

    gender_col = next((c for c in ["gender_short", "gender", "Gender"] if c in df_viz.columns), None)
    with gc1:
        if gender_col and attr_col:
            st.markdown("**Attrition by Gender**")
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                g_data = df_viz.groupby(gender_col)[attr_col].mean() * 100
                bars = ax.bar(g_data.index, g_data.values, color=PAL, edgecolor="none",
                              width=0.5, linewidth=0)
                ax.set_ylabel("Attrition Rate (%)"); ax.set_title("By Gender", fontweight="bold")
                ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#94a3b8", fontsize=9)
                ax.grid(axis="y", alpha=0.2); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    bu_col = next((c for c in ["BUSINESS_UNIT", "business_unit", "BU"] if c in df_viz.columns), None)
    with gc2:
        if bu_col and attr_col:
            st.markdown("**Attrition by Business Unit**")
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                bu_data = df_viz.groupby(bu_col)[attr_col].mean() * 100
                bars = ax.bar(bu_data.index, bu_data.values,
                              color=["#6366f1","#38bdf8"], edgecolor="none", width=0.5)
                ax.set_ylabel("Attrition Rate (%)"); ax.set_title("By Business Unit", fontweight="bold")
                ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#94a3b8", fontsize=9)
                ax.grid(axis="y", alpha=0.2); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    # Correlation heatmap
    with st.expander("🌡 Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(11, 6))
        with plt.rc_context(PSTYLE):
            fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f",
                        cmap="coolwarm", linewidths=0.5, linecolor="#1e293b",
                        cbar_kws={"shrink": 0.75}, annot_kws={"size": 7})
            ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
            fig.tight_layout()
        st.pyplot(fig); plt.close(fig)


# ═════════════════════════════════════════════
# TAB 3 — TRAIN & RESULTS
# ═════════════════════════════════════════════
with tab3:
    st.markdown('<div class="stitle"><span class="stitle-num">6</span> Model Training</div>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    ic1, ic2, ic3, ic4 = st.columns(4)
    for col, (lbl, val) in zip(
        [ic1, ic2, ic3, ic4],
        [("Total", f"{len(X):,}"), ("Train", f"{len(X_train):,}"),
         ("Test", f"{len(X_test):,}"), ("Features", str(X.shape[1]))]
    ):
        with col:
            st.markdown(f"""
            <div class="mcard">
                <div class="ml">{lbl}</div>
                <div class="mv" style="font-size:1.5rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀  Train Logistic Regression"):
        bar = st.progress(0, text="Initialising…")
        time.sleep(0.2); bar.progress(25, "Preparing data…")
        model = LogisticRegression(max_iter=1000, random_state=int(random_state))
        time.sleep(0.3); bar.progress(60, "Fitting model…")
        model.fit(X_train, y_train)
        time.sleep(0.2); bar.progress(85, "Evaluating…")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        st.session_state.update({
            "model": model, "acc": acc, "cm": cm, "report": report,
            "feature_cols": X.columns.tolist(), "fpr": fpr, "tpr": tpr,
            "roc_auc": roc_auc,
        })
        bar.progress(100, "Done!")
        time.sleep(0.3); bar.empty()
        st.success("✅ Model trained successfully!")

    if "model" in st.session_state:
        acc    = st.session_state["acc"]
        cm     = st.session_state["cm"]
        report = st.session_state["report"]
        fpr    = st.session_state["fpr"]
        tpr    = st.session_state["tpr"]
        roc_auc= st.session_state["roc_auc"]

        st.markdown('<div class="stitle"><span class="stitle-num">7</span> Performance Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        prec = report.get("1", report.get("weighted avg", {})).get("precision", 0)
        rec  = report.get("1", report.get("weighted avg", {})).get("recall", 0)
        f1   = report.get("1", report.get("weighted avg", {})).get("f1-score", 0)

        for col, (lbl, val, accent) in zip(
            [m1, m2, m3, m4],
            [("Accuracy",  f"{acc*100:.1f}%",  "--sky"),
             ("Precision", f"{prec*100:.1f}%", "--indigo"),
             ("Recall",    f"{rec*100:.1f}%",  "--amber"),
             ("F1 Score",  f"{f1*100:.1f}%",   "--emerald")]
        ):
            with col:
                st.markdown(f"""
                <div class="mcard" style="--accent:var({accent})">
                    <div class="ml">{lbl}</div>
                    <div class="mv">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(4.2, 3.5))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                            linewidths=0.5, linecolor="#1e293b",
                            xticklabels=["Stayed","Left"], yticklabels=["Stayed","Left"],
                            cbar=False, annot_kws={"size": 14, "weight": "bold"})
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix", fontweight="bold")
                fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with r2:
            st.markdown("**ROC Curve**")
            fig, ax = plt.subplots(figsize=(4.2, 3.5))
            with plt.rc_context(PSTYLE):
                fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
                ax.plot(fpr, tpr, color="#38bdf8", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
                ax.fill_between(fpr, tpr, alpha=0.1, color="#38bdf8")
                ax.plot([0,1],[0,1], "--", color="#475569", linewidth=1)
                ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve", fontweight="bold")
                ax.legend(framealpha=0.15, fontsize=9)
                ax.grid(alpha=0.15)
                fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        st.markdown("**Top Feature Importances (Coefficients)**")
        feature_cols = st.session_state["feature_cols"]
        model_obj    = st.session_state["model"]
        coef_df = pd.DataFrame({
            "Feature"    : feature_cols,
            "Coefficient": model_obj.coef_[0],
        }).sort_values("Coefficient", key=abs, ascending=False).head(12)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        with plt.rc_context(PSTYLE):
            fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            colors_feat = [PAL[1] if v > 0 else PAL[0] for v in coef_df["Coefficient"]]
            ax.barh(coef_df["Feature"], coef_df["Coefficient"],
                    color=colors_feat, edgecolor="none", height=0.65)
            ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Coefficient Value")
            ax.set_title("Feature Weights (Logistic Regression)", fontweight="bold")
            ax.grid(axis="x", alpha=0.15)
            p1 = mpatches.Patch(color=PAL[1], label="→ Increases attrition risk")
            p2 = mpatches.Patch(color=PAL[0], label="→ Decreases attrition risk")
            ax.legend(handles=[p1, p2], framealpha=0.1, fontsize=8)
            fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.markdown('<div class="info-box">👆 Click <strong>Train Logistic Regression</strong> above to see results.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# TAB 4 — PREDICT
# ═════════════════════════════════════════════
with tab4:
    st.markdown('<div class="stitle"><span class="stitle-num">8</span> Real-Time Prediction</div>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.markdown('<div class="info-box">⚠️ Train the model first in the <strong>Train & Results</strong> tab.</div>', unsafe_allow_html=True)
    else:
        st.markdown("Configure the employee profile below, then click **Predict**.")

        feature_cols = st.session_state["feature_cols"]
        input_vals = {}

        # FIX 3: gender_short uses "M"/"F" in the real dataset (not "Male"/"Female")
        PRETTY = {
            "age"              : ("Age", 18, 65, 35),
            "length_of_service": ("Years of Service", 0, 40, 5),
            "STATUS_YEAR"      : ("Status Year", 2010, 2024, 2018),
            "gender_short"     : ("Gender", ["M", "F"]),
            "department_name"  : ("Department", ["Accounting", "Customer Service", "Dairy",
                                                  "Executive", "IT", "Meats", "Produce",
                                                  "Processed Foods", "Bakery", "HR"]),
            "job_title"        : ("Job Title", ["Cashier", "Manager", "CEO", "Analyst",
                                                 "Director", "Clerk", "Supervisor"]),
            "BUSINESS_UNIT"    : ("Business Unit", ["HEADOFFICE", "STORES"]),
        }

        cols_ui = st.columns(3)
        for idx, feat in enumerate(feature_cols):
            with cols_ui[idx % 3]:
                if feat in PRETTY:
                    info = PRETTY[feat]
                    lbl = info[0]
                    if isinstance(info[1], list):
                        options = info[1]
                        # Only show options that exist in the encoder
                        if feat in le_dict:
                            known = list(le_dict[feat].classes_)
                            options = [o for o in options if o in known] or known
                        sel = st.selectbox(lbl, options, key=f"p_{feat}")
                        if feat in le_dict and sel in list(le_dict[feat].classes_):
                            input_vals[feat] = int(le_dict[feat].transform([sel])[0])
                        else:
                            input_vals[feat] = 0
                    else:
                        _, mn, mx, def_ = info
                        input_vals[feat] = st.slider(lbl, mn, mx, def_, key=f"p_{feat}")
                else:
                    col_min = int(df[feat].min()) if not np.isnan(df[feat].min()) else 0
                    col_max = int(df[feat].max()) if not np.isnan(df[feat].max()) else 100
                    col_def = int(df[feat].mean()) if not np.isnan(df[feat].mean()) else col_min
                    input_vals[feat] = st.slider(
                        feat.replace("_", " ").title(), col_min, col_max, col_def, key=f"p_{feat}"
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔮  Run Prediction", use_container_width=False):
            input_df    = pd.DataFrame([input_vals])[feature_cols]
            prediction  = st.session_state["model"].predict(input_df)[0]
            probability = st.session_state["model"].predict_proba(input_df)[0]

            res1, res2 = st.columns([1.2, 1])
            with res1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="pred-leave">
                        <div class="pred-icon">🚨</div>
                        <div class="pred-label" style="color:#fb7185">Attrition Risk</div>
                        <div class="pred-verdict" style="color:#fca5a5">Likely to LEAVE</div>
                        <div class="pred-conf" style="color:#fb7185">Confidence: {probability[1]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-stay">
                        <div class="pred-icon">✅</div>
                        <div class="pred-label" style="color:#34d399">Retention Signal</div>
                        <div class="pred-verdict" style="color:#6ee7b7">Likely to STAY</div>
                        <div class="pred-conf" style="color:#34d399">Confidence: {probability[0]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            with res2:
                st.markdown(f"""
                <div class="prob-bar-wrap">
                    <div class="prob-bar-label">Probability Breakdown</div>
                    <div style="margin-bottom:0.8rem">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                            <span style="font-size:0.8rem;color:#34d399">🟢 Stay</span>
                            <span class="prob-bar-val">{probability[0]*100:.1f}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill-stay" style="width:{probability[0]*100:.1f}%"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                            <span style="font-size:0.8rem;color:#fb7185">🔴 Leave</span>
                            <span class="prob-bar-val">{probability[1]*100:.1f}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill-leave" style="width:{probability[1]*100:.1f}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# TAB 5 — ABOUT
# ═════════════════════════════════════════════
with tab5:
    st.markdown('<div class="stitle"><span class="stitle-num">9</span> About This App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-size:0.92rem;line-height:1.8">
    <strong style="color:#38bdf8">AttritionIQ</strong> is an end-to-end employee attrition intelligence platform.<br><br>
    <strong>Workflow</strong><br>
    1. Upload a CSV dataset or use the built-in synthetic sample<br>
    2. Explore patterns via interactive charts<br>
    3. Train a Logistic Regression classifier<br>
    4. Predict attrition risk for individual employees<br><br>
    <strong>Model</strong> — Scikit-learn Logistic Regression (max_iter=1000, stratified split)<br>
    <strong>Target</strong> — Binary: 0 = Stayed (ACTIVE), 1 = Left (TERMINATED)<br>
    <strong>Metrics</strong> — Accuracy, Precision, Recall, F1, AUC-ROC<br><br>
    <strong>Expected CSV columns</strong> — STATUS, age, department_name, gender_short,
    length_of_service, BUSINESS_UNIT, job_title, STATUS_YEAR, EmployeeID (optional, used for deduplication)
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stitle"><span class="stitle-num">10</span> Demo Credentials</div>', unsafe_allow_html=True)
    cred_data = pd.DataFrame([
        {"Username": "admin",   "Password": "admin123",   "Role": "Administrator"},
        {"Username": "analyst", "Password": "analyst123", "Role": "HR Analyst"},
        {"Username": "demo",    "Password": "demo",       "Role": "Guest"},
    ])
    st.dataframe(cred_data, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#1e293b'>"
    "AttritionIQ · Employee Intelligence Platform · Built with Streamlit & Scikit-learn"
    "</small></center>",
    unsafe_allow_html=True,
)
