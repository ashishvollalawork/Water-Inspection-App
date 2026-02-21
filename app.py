import os
import io
import json
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
# ---- PDF (ReportLab) ----
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ------------- App Constants -------------
ARTIFACT_MODEL = "artifacts/drinkstatus_model.pkl"
ARTIFACT_LE    = "artifacts/drinkstatus_label_encoder.pkl"
BRAND_CSV      = "brand_master.csv"
LAST_JSON      = "last_inspection.json"

PAGE_TITLE = "💧 Water Inspection (ML + Offline)"
PAGE_ICON = "💧"

# ------------- Streamlit Page Setup -------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="collapsed")

# ------------- Custom Styling -------------
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"><style>* { font-family: 'Poppins', sans-serif; } :root{ --primary:#0066CC; --accent:#764BA2; --success:#10b981; --warning:#f59e0b; --danger:#ef4444; --info:#06b6d4; --muted:#6c7a89; --bg:#f7fbff; --surface:#ffffff; --radius:12px; --easing:cubic-bezier(.2,.9,.2,1); } html, body { background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); min-height: 100vh; } @keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} } @keyframes fadeInUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}} @keyframes slideInLeft{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}} @keyframes slideInRight{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}} @keyframes popIn{from{opacity:0;transform:scale(0.99)}to{opacity:1;transform:scale(1)}} @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}} @keyframes float{0%,100%{transform:translateY(0px)}50%{transform:translateY(-10px)}} @keyframes glow{0%{box-shadow:0 0 5px rgba(0,102,204,0.3)}50%{box-shadow:0 0 25px rgba(102,126,234,0.6)}100%{box-shadow:0 0 5px rgba(0,102,204,0.3)}} @keyframes shimmerFlow{0%{background-position:200% center}100%{background-position:-200% center}} @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}} @keyframes slideUp{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}} @keyframes scaleIn{from{opacity:0;transform:scale(0.9)}to{opacity:1;transform:scale(1)}} @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}} @media (prefers-reduced-motion: reduce){ * { animation:none !important; transition:none !important } } .stApp { background: transparent !important; } .stNumberInput > div > div > input, .stTextInput > div > input { border-radius:14px !important; border:2px solid rgba(102,126,234,0.3) !important; padding:14px 18px !important; background:linear-gradient(135deg,rgba(255,255,255,0.95) 0%,rgba(248,249,255,0.95) 100%) !important; animation:slideInLeft 500ms var(--easing) both; transition:all 300ms var(--easing) !important; font-size:1.05rem !important; font-weight:500 !important; backdrop-filter:blur(10px); } .stNumberInput > div > div > input:focus, .stTextInput > div > input:focus { border-color:#667eea !important; background:rgba(255,255,255,1) !important; box-shadow:0 0 0 4px rgba(102,126,234,0.15),0 12px 32px rgba(102,126,234,0.2) !important; transform:translateY(-3px) !important; } .stButton > button { border-radius:14px !important; padding:14px 28px !important; font-weight:800 !important; font-size:1rem !important; letter-spacing:0.5px !important; transition:all 280ms var(--easing) !important; animation:popIn 1200ms cubic-bezier(0.25,0.46,0.45,0.94) both; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important; border:none !important; color:#fff !important; box-shadow:0 8px 24px rgba(102,126,234,0.3),inset 0 1px 0 rgba(255,255,255,0.15) !important; position:relative; overflow:hidden; cursor:pointer; } .stButton > button::before { content:''; position:absolute; top:50%; left:50%; width:0; height:0; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,0.4) 0%,rgba(255,255,255,0) 70%); transform:translate(-50%,-50%); transition:width 700ms,height 700ms; } .stButton > button:hover { transform:translat2Y(-5px) !important; box-shadow:0 16px 40px rgba(102,126,234,0.4),inset 0 1px 0 rgba(255,255,255,0.2) !important; } .stButton > button:active { transform:translateY(-2px) !important; } .metric-card { border-radius:16px !important; box-shadow:0 12px 32px rgba(15,23,42,0.1); transition:all 350ms var(--easing); animation:fadeInUp 700ms var(--easing) both; background:linear-gradient(135deg,rgba(255,255,255,0.95) 0%,rgba(248,249,255,0.95) 100%); border:2px solid rgba(102,126,234,0.15); position:relative; overflow:hidden; backdrop-filter:blur(10px); } .metric-card::before { content:''; position:absolute; top:-50%; right:-50%; width:200%; height:200%; background:radial-gradient(circle,rgba(255,255,255,0.8) 0%,transparent 70%); opacity:0; transition:opacity 500ms ease; } .metric-card::after { content:''; position:absolute; top:0; left:-100%; width:100%; height:100%; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.5),transparent); transition:left 600ms ease; } .metric-card:hover { transform:translateY(-12px); box-shadow:0 24px 48px rgba(102,126,234,0.2),0 0 0 2px rgba(102,126,234,0.3); } .metric-card:hover::after { left:100%; } .metric-card:hover::before { opacity:1; } .stSelectbox > div > div { border-radius:14px !important; border:2px solid rgba(102,126,234,0.3) !important; transition:all 250ms var(--easing); background:linear-gradient(135deg,rgba(255,255,255,0.9) 0%,rgba(248,249,255,0.9) 100%) !important; } .stSelectbox > div > div:hover, .stSelectbox > div > div:focus { border-color:#667eea !important; box-shadow:0 0 0 4px rgba(102,126,234,0.15),0 8px 24px rgba(102,126,234,0.15) !important; } .info-card { background:linear-gradient(135deg,rgba(102,126,234,0.1) 0%,rgba(118,75,162,0.1) 100%); padding:1.5rem; border-radius:14px; border-left:5px solid #667eea; border:2px solid rgba(102,126,234,0.15); font-size:0.98rem; animation:fadeInUp 750ms var(--easing) both; backdrop-filter:blur(10px); font-weight:500; } .stDivider { margin:2.5rem 0 !important; opacity:0.3; } h2 { color:#0a1f3a !important; font-weight:700 !important; letter-spacing:-0.5px !important; font-size:2rem !important; text-shadow:0 1px 2px rgba(0,0,0,0.03); } h1 { color:#fff; font-weight:900; letter-spacing:-1px; } .header-banner { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important; border-radius:16px !important; padding:2.5rem !important; margin-bottom:0.8rem !important; animation:popIn 600ms var(--easing) both; box-shadow:0 16px 40px rgba(102,126,234,0.25),inset 0 1px 0 rgba(255,255,255,0.1); position:relative; overflow:hidden; } .header-banner::before { content:''; position:absolute; top:0; left:0; right:0; bottom:0; background:radial-gradient(circle at 20% 50%,rgba(255,255,255,0.1) 0%,transparent 50%),radial-gradient(circle at 80% 80%,rgba(255,255,255,0.05) 0%,transparent 50%); pointer-events:none; } .metric-number { font-size:2.8rem !important; font-weight:900 !important; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; animation:float 3s ease-in-out infinite; } .status-badge { display:inline-block; padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.9rem; letter-spacing:0.5px; } .badge-safe { background:linear-gradient(135deg,#10b981 0%,#059669 100%); color:white; } .badge-warning { background:linear-gradient(135deg,#f59e0b 0%,#d97706 100%); color:white; } .badge-danger { background:linear-gradient(135deg,#ef4444 0%,#dc2626 100%); color:white; } @keyframes pulse-glow { 0%,100% { box-shadow:0 0 10px rgba(102,126,234,0.3); } 50% { box-shadow:0 0 30px rgba(102,126,234,0.6); } } .feature-card { background:linear-gradient(135deg,rgba(255,255,255,0.95) 0%,rgba(248,249,255,0.95) 100%); border:2px solid rgba(102,126,234,0.15); border-radius:16px; padding:2rem; backdrop-filter:blur(10px); transition:all 350ms cubic-bezier(.2,.9,.2,1); animation:slideUp 700ms cubic-bezier(.2,.9,.2,1) both; position:relative; overflow:hidden; } .feature-card::before { content:''; position:absolute; top:-50%; right:-50%; width:200%; height:200%; background:radial-gradient(circle,rgba(102,126,234,0.1) 0%,transparent 70%); opacity:0; transition:opacity 500ms ease; } .feature-card:hover { transform:translateY(-8px); box-shadow:0 20px 40px rgba(102,126,234,0.15),0 0 0 2px rgba(102,126,234,0.2); border-color:rgba(102,126,234,0.3); } .feature-card:hover::before { opacity:1; } .feature-icon { font-size:3rem; margin-bottom:1rem; display:inline-block; animation:float 3s ease-in-out infinite; } .feature-title { font-size:1.3rem; font-weight:800; color:#0a1f3a; margin-bottom:0.75rem; letter-spacing:-0.5px; } .feature-desc { font-size:0.95rem; color:#555; line-height:1.6; font-weight:500; } .step-indicator { display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; font-weight:700; font-size:0.9rem; box-shadow:0 4px 15px rgba(102,126,234,0.3); animation:popIn 600ms cubic-bezier(.2,.9,.2,1) both; } .status-pill { display:inline-block; padding:8px 16px; border-radius:20px; font-weight:700; font-size:0.85rem; letter-spacing:0.5px; animation:slideUp 600ms cubic-bezier(.2,.9,.2,1) both; text-transform:uppercase; } .status-safe { background:linear-gradient(135deg,#10b981 0%,#059669 100%); color:white; box-shadow:0 4px 15px rgba(16,185,129,0.3); } .status-warning { background:linear-gradient(135deg,#f59e0b 0%,#d97706 100%); color:white; box-shadow:0 4px 15px rgba(245,158,11,0.3); } .status-danger { background:linear-gradient(135deg,#ef4444 0%,#dc2626 100%); color:white; box-shadow:0 4px 15px rgba(239,68,68,0.3); } .param-badge { background:linear-gradient(135deg,rgba(102,126,234,0.1) 0%,rgba(118,75,162,0.05) 100%); border:2px solid rgba(102,126,234,0.2); border-radius:12px; padding:1rem; margin:0.5rem 0; animation:slideUp 700ms cubic-bezier(.2,.9,.2,1) both; } .param-label { font-size:0.85rem; color:#666; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; } .param-value { font-size:1.5rem; font-weight:900; color:#667eea; margin-top:0.35rem; } .quality-indicator { width:100%; height:8px; border-radius:10px; background:linear-gradient(90deg,#ef4444 0%,#f59e0b 25%,#667eea 50%,#10b981 100%); margin:1rem 0; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1); } .quality-marker { height:100%; width:4px; background:#0a1f3a; border-radius:10px; animation:slideInLeft 800ms cubic-bezier(.2,.9,.2,1) both; } .comparison-panel { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1.5rem 0; } .panel-item { background:linear-gradient(135deg,rgba(255,255,255,0.95) 0%,rgba(248,249,255,0.95) 100%); border:2px solid rgba(102,126,234,0.15); border-radius:14px; padding:1.5rem; backdrop-filter:blur(10px); animation:slideUp 700ms cubic-bezier(.2,.9,.2,1) both; } .panel-title { font-size:0.95rem; font-weight:700; color:#0a1f3a; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.75rem; } .panel-value { font-size:1.8rem; font-weight:900; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; } .insight-box { background:linear-gradient(135deg,rgba(102,126,234,0.08) 0%,rgba(16,185,129,0.05) 100%); border-left:4px solid #667eea; border-radius:10px; padding:1.25rem; margin:1rem 0; animation:slideUp 800ms cubic-bezier(.2,.9,.2,1) both; } .insight-title { font-weight:700; color:#0a1f3a; margin-bottom:0.5rem; font-size:0.95rem; } .insight-text { color:#555; font-size:0.9rem; line-height:1.6; }</style>""", unsafe_allow_html=True)

# ------------- Helpers -------------
@st.cache_resource
def load_model_and_encoder():
    if not (os.path.exists(ARTIFACT_MODEL) and os.path.exists(ARTIFACT_LE)):
        return None, None
    try:
        model = joblib.load(ARTIFACT_MODEL)
        le = joblib.load(ARTIFACT_LE)
        return model, le
    except Exception:
        return None, None

@st.cache_data
def load_brand_master(path=BRAND_CSV):
    if not os.path.exists(path):
        # bootstrap a sensible default
        df = pd.DataFrame({
            "SwitchBoards": ["Legrand","GM","Schneider","Anchor","Havells"],
            "Faucets": ["Jaquar","Kohler","Cera","Hindware","Parryware"],
            "WashBasins": ["Cera","Hindware","Kohler","Jaquar","Parryware"],
            "WC": ["Hindware","Parryware","Cera","Kohler","Jaquar"]
        })
        df.to_csv(path, index=False)
        return {c: sorted(df[c].dropna().astype(str).unique()) for c in df.columns}

    df = pd.read_csv(path)
    expected = ["SwitchBoards","Faucets","WashBasins","WC"]
    # Fill any missing expected columns
    for col in expected:
        if col not in df.columns:
            df[col] = []
    brands = {c: sorted(list(df[c].dropna().astype(str).unique())) for c in expected}
    return brands

def load_last():
    if not os.path.exists(LAST_JSON):
        return None
    try:
        with open(LAST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # empty file or {}
        return data if data else None
    except Exception:
        return None  # corrupted or unreadable

def save_last(payload: dict):
    with open(LAST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def ensure_last_json_ui():
    if not os.path.exists(LAST_JSON):
        st.info("`last_inspection.json` not found. You can create it now (optional).")
        if st.button("Create last_inspection.json"):
            save_last({})
            st.success("Created `last_inspection.json`. It will store your last saved inspection.")
            st.experimental_rerun()

def compute_ph_status(ph: float) -> str:
    # Simple, app-local rule bands (adjust if you have different policy)
    if ph < 6.5:
        return "Low"
    if ph > 8.5:
        return "High"
    return "Normal"

def compute_water_quality(tds: float) -> str:
    # Simple bands (app-local; adjust as needed)
    if tds <= 150:
        return "Good"
    elif tds <= 300:
        return "Fair"
    elif tds <= 500:
        return "Poor"
    else:
        return "Very Poor"

def compute_risk_and_reco(drink_label: str, tds: float, ph: float):
    # Coarse risk logic using both ML output and ranges
    if drink_label.lower() == "safe" and (0 <= tds <= 500) and (6.5 <= ph <= 8.5):
        return "Low", "Water is acceptable for drinking."
    # Higher risk thresholds
    high_risk = (tds > 800) or (ph < 6.0) or (ph > 9.0)
    risk = "High" if high_risk else "Medium"
    reasons = []
    if tds > 500: reasons.append("high TDS")
    if ph < 6.5 or ph > 8.5: reasons.append("pH out of range")
    reason_txt = "; ".join(reasons) if reasons else "quality concerns"
    reco = f"Not suitable for direct drinking due to {reason_txt}. Consider treatment or alternate source."
    return risk, reco

def predict_drinking_status(model, le, tds: float, ph: float):
    """
    Uses trained artifacts if present; otherwise falls back to a simple rule.
    Returns (label, proba_safe or None).
    """
    if (model is None) or (le is None):
        # Fallback rule if model not available
        safe = (0 <= tds <= 500) and (6.5 <= ph <= 8.5)
        return ("Safe" if safe else "Not Safe"), None

    x = np.array([[tds, ph]], dtype=float)
    y_pred = model.predict(x)[0]
    label = le.inverse_transform([y_pred])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba_all = model.predict_proba(x)
            # Map to 'Safe' probability using label encoder
            if isinstance(proba_all, list):
                proba_all = proba_all[0]  # some pipelines wrap it
            if len(le.classes_) == proba_all.shape[1] and "Safe" in le.classes_:
                safe_idx = list(le.classes_).index("Safe")
                proba = float(proba_all[:, safe_idx])
            elif proba_all.shape[1] == 2:
                proba = float(proba_all[:, 1])
        except Exception:
            proba = None

    return label, proba

def make_pdf_bytes(payload: dict) -> bytes:
    """
    Build an A4 PDF with the required order:
      1) Water Analysis (TDS, pH, results, recommendation)
      2) Fixture Brands
    No inspector name or timestamp included (per requirement).
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left = 18 * mm
    top = height - 20 * mm
    line = top

    def draw_h1(text):
        nonlocal line
        c.setFont("Helvetica-Bold", 16)
        c.drawString(left, line, text)
        line -= 10 * mm

    def draw_h2(text):
        nonlocal line
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, line, text)
        line -= 6 * mm

    def draw_kv(k, v):
        nonlocal line
        c.setFont("Helvetica", 11)
        c.drawString(left, line, f"{k}: {v}")
        line -= 6 * mm

    # Title
    draw_h1("Water Inspection Report")

    # Section 1: Water Analysis
    draw_h2("1) Water Analysis")
    draw_kv("TDS (mg/L)", payload.get("TDS"))
    draw_kv("pH", payload.get("PH"))
    draw_kv("Water Quality", payload.get("WaterQuality"))
    draw_kv("pH Status", payload.get("PHStatus"))
    draw_kv("Drinking Status", payload.get("DrinkingStatus"))
    draw_kv("Risk Level", payload.get("RiskLevel"))

    # Recommendation (wrap basic)
    c.setFont("Helvetica", 11)
    rec_text = f"Recommendation: {payload.get('Recommendation', '')}"
    # Simple wrap at ~95 chars
    max_chars = 95
    y = line
    for i in range(0, len(rec_text), max_chars):
        c.drawString(left, y, rec_text[i:i+max_chars])
        y -= 6 * mm
    line = y - 2 * mm

    # Section 2: Fixture Brands
    draw_h2("2) Fixture Brands")
    draw_kv("SwitchBoards", payload.get("SwitchBoardBrand", "-"))
    draw_kv("Faucets", payload.get("FaucetBrand", "-"))
    draw_kv("WashBasins", payload.get("WashBasinBrand", "-"))
    draw_kv("WC", payload.get("WCBrand", "-"))

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ------------- UI -------------
# Header with gradient effect
st.markdown("""
<div class='header-banner'>
    <div style='position: relative; z-index: 1;'>
        <h1 style='margin: 0; font-size: 3rem; text-align: center;'>💧 Water Inspection</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.75rem 0 0 0; font-size: 1.15rem; text-align: center; font-weight: 500;'>Professional Water Quality Assessment with AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("Follows your specified procedure: **Water first** → **Brands** → **A4 PDF Report**. Last inspection auto-loads.", unsafe_allow_html=True)

# Feature cards section
st.markdown("<div style='margin-top: 2.5rem; margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True)

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class='feature-card' data-scroll style='animation-delay: 100ms;'>
        <div class='feature-icon'>🧪</div>
        <div class='feature-title'>ML-Powered Analysis</div>
        <div class='feature-desc'>Advanced machine learning algorithms analyze water parameters with high accuracy for reliable results.</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class='feature-card' data-scroll style='animation-delay: 200ms;'>
        <div class='feature-icon'>📊</div>
        <div class='feature-title'>Instant Results</div>
        <div class='feature-desc'>Get immediate insights about water quality, safety status, and actionable recommendations in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class='feature-card' data-scroll style='animation-delay: 300ms;'>
        <div class='feature-icon'>📋</div>
        <div class='feature-title'>PDF Reports</div>
        <div class='feature-desc'>Generate professional A4 PDF reports with all water analysis data and fixture brand selections.</div>
    </div>
    """, unsafe_allow_html=True)
components.html(r"""
<script>
  (function(){
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if(entry.isIntersecting){
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -100px 0px' });
    
    setTimeout(() => {
      document.querySelectorAll('[data-scroll]').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(50px)';
        el.style.transition = 'opacity 1200ms cubic-bezier(.2,.9,.2,1), transform 600ms cubic-bezier(.2,.9,.2,1)';
        observer.observe(el);
      });
    }, 100);
  })();
</script>
""", height=0)

# Load artifacts and brands
model, le = load_model_and_encoder()
brands = load_brand_master()

# Session state for gating
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "loaded_last" not in st.session_state:
    st.session_state.loaded_last = False

# Load last inspection (once per session)
if not st.session_state.loaded_last:
    last = load_last()
    st.session_state.loaded_last = True
else:
    last = None

# --------- WATER SECTION (must be first) ---------
st.markdown("<h2 style='animation:fadeInUp 480ms cubic-bezier(.2,.9,.2,1) both; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top: 0.5rem;'>🌊 Water Testing & Analysis</h2>", unsafe_allow_html=True)

st.markdown("<div class='info-card'><span class='step-indicator'>1</span> <strong style='margin-left: 0.75rem;'>Enter water parameters (TDS & pH) and analyze to proceed</strong></div>", unsafe_allow_html=True)

# Defaults (either from last saved or sensible defaults)
default_tds = float(last.get("TDS")) if last else 250.0
default_ph  = float(last.get("PH")) if last else 7.2

c1, c2, c3 = st.columns([1, 1, 0.5])
with c1:
    tds = st.number_input("📊 TDS (mg/L) *", min_value=0.0, max_value=5000.0, value=default_tds, step=10.0, format="%.2f", help="Total Dissolved Solids in mg/L")
with c2:
    ph = st.number_input("🔬 pH Level *", min_value=0.0, max_value=14.0, value=default_ph, step=0.1, format="%.2f", help="pH value (0-14)")
with c3:
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ Analyze", use_container_width=True)

if analyze_btn:
    if ph < 0 or ph > 14:
        st.error("pH must be between 0 and 14.")
        st.stop()
    # Predict DrinkingStatus (ML or fallback)
    label, proba = predict_drinking_status(model, le, tds, ph)
    ph_status = compute_ph_status(ph)
    water_quality = compute_water_quality(tds)
    risk, reco = compute_risk_and_reco(label, tds, ph)

    st.session_state.analyzed = True
    st.session_state.analysis = {
        "TDS": round(float(tds), 2),
        "PH": round(float(ph), 2),
        "WaterQuality": water_quality,
        "PHStatus": ph_status,
        "DrinkingStatus": label,
        "RiskLevel": risk,
        "Recommendation": reco,
        "ProbaSafe": (None if proba is None else round(float(proba), 4)),
    }

# Show results (if already analyzed, including when loaded from last)
if st.session_state.analyzed:
    A = st.session_state.analysis
    
    # Color coding for status
    drink_emoji = "🟢" if A['DrinkingStatus'].lower() == "safe" else "🔴"
    risk_emoji = "✅" if A['RiskLevel'] == "Low" else "⚠️" if A['RiskLevel'] == "Medium" else "❌"
    quality_emoji = "⭐" if A['WaterQuality'] == "Good" else "👍" if A['WaterQuality'] == "Fair" else "👎"
    
    # Display analysis results in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' data-scroll style='background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(5,150,105,0.05) 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #10b981; text-align: center; animation:fadeInUp 700ms cubic-bezier(.2,.9,.2,1) 80ms both; border:2px solid rgba(16,185,129,0.25); backdrop-filter:blur(10px)'>
            <div style='font-size: 3.2rem; margin-bottom:0.75rem; animation:float 3s ease-in-out infinite'>{drink_emoji}</div>
            <div style='font-size: 0.88rem; color: #666; font-weight:600; text-transform: uppercase; letter-spacing:1px;'>Drinking Status</div>
            <div style='font-size: 1.6rem; font-weight: 900; color: #10b981; margin-top:0.75rem;'>{A['DrinkingStatus']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' data-scroll style='background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.05) 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #ef4444; text-align: center; animation:fadeInUp 700ms cubic-bezier(.2,.9,.2,1) 160ms both; border:2px solid rgba(239,68,68,0.25); backdrop-filter:blur(10px)'>
            <div style='font-size: 3.2rem; margin-bottom:0.75rem; animation:float 3s ease-in-out infinite 0.2s'>{risk_emoji}</div>
            <div style='font-size: 0.88rem; color: #666; font-weight:600; text-transform: uppercase; letter-spacing:1px;'>Risk Level</div>
            <div style='font-size: 1.6rem; font-weight: 900; color: #ef4444; margin-top:0.75rem;'>{A['RiskLevel']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card' data-scroll style='background: linear-gradient(135deg, rgba(6,182,212,0.1) 0%, rgba(8,145,178,0.05) 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #06b6d4; text-align: center; animation:fadeInUp 700ms cubic-bezier(.2,.9,.2,1) 240ms both; border:2px solid rgba(6,182,212,0.25); backdrop-filter:blur(10px)'>
            <div style='font-size: 3.2rem; margin-bottom:0.75rem; animation:float 3s ease-in-out infinite 0.4s'>{quality_emoji}</div>
            <div style='font-size: 0.88rem; color: #666; font-weight:600; text-transform: uppercase; letter-spacing:1px;'>Water Quality</div>
            <div style='font-size: 1.6rem; font-weight: 900; color: #06b6d4; margin-top:0.75rem;'>{A['WaterQuality']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card' data-scroll style='background: linear-gradient(135deg, rgba(168,85,247,0.1) 0%, rgba(147,51,234,0.05) 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #a855f7; text-align: center; animation:fadeInUp 700ms cubic-bezier(.2,.9,.2,1) 320ms both; border:2px solid rgba(168,85,247,0.25); backdrop-filter:blur(10px)'>
            <div style='font-size: 3.2rem; margin-bottom:0.75rem; animation:float 3s ease-in-out infinite 0.6s'>⚗️</div>
            <div style='font-size: 0.88rem; color: #666; font-weight:600; text-transform: uppercase; letter-spacing:1px;'>pH Status</div>
            <div style='font-size: 1.6rem; font-weight: 900; color: #a855f7; margin-top:0.75rem;'>{A['PHStatus']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div data-scroll style='animation: slideUp 800ms cubic-bezier(.2,.9,.2,1) 200ms both'><strong>📈 TDS Reading:</strong> <span style='color:#667eea; font-weight:900; font-size:1.2rem'>{A['TDS']} mg/L</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div data-scroll style='animation: slideUp 800ms cubic-bezier(.2,.9,.2,1) 300ms both; margin-top:1rem;'><strong>⚗️ pH Reading:</strong> <span style='color:#667eea; font-weight:900; font-size:1.2rem'>{A['PH']}</span></div>", unsafe_allow_html=True)
    with col2:
        if A["ProbaSafe"] is not None:
            progress = A['ProbaSafe']
            st.markdown(f"<div data-scroll style='animation: slideUp 800ms cubic-bezier(.2,.9,.2,1) 400ms both'><strong>🎯 Safety Confidence:</strong> <span style='color:#10b981; font-weight:900; font-size:1.2rem'>{A['ProbaSafe']*100:.1f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top: 0.5rem; animation: slideUp 800ms cubic-bezier(.2,.9,.2,1) 500ms both' data-scroll>", unsafe_allow_html=True)
            st.progress(min(progress, 1.0))
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendation box
    st.markdown(f"""
    <div data-scroll style='background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.1) 100%); 
                border-left: 6px solid #667eea; 
                padding: 2.25rem; 
                border-radius: 16px; 
                margin: 2rem 0; 
                border: 2px solid rgba(102,126,234,0.2);
                backdrop-filter: blur(10px);
                animation: fadeInUp 800ms cubic-bezier(.2,.9,.2,1) 400ms both;'>
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
            <div style='font-size: 2rem;'>💡</div>
            <strong style='font-size: 1.2rem; color:#0a1f3a; text-transform: uppercase; letter-spacing: 0.5px;'>Recommendation</strong>
        </div>
        <p style='margin: 0; font-size: 1rem; color:#333; line-height:1.8; font-weight: 500;'>{A['Recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quality insights
    st.markdown("<div style='margin-top: 2rem; margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0a1f3a; font-weight: 800; margin-bottom: 1rem;'>📊 Quality Analysis Breakdown</h3>", unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(f"""
        <div class='insight-box'>
            <div class='insight-title'>💧 TDS Assessment</div>
            <div class='insight-text'>
                Your water has {A['TDS']} mg/L dissolved solids.
                {"✅ Within safe limits (0-500 mg/L)" if A['TDS'] <= 500 else "⚠️ Exceeds recommended limit (500+ mg/L)"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f"""
        <div class='insight-box'>
            <div class='insight-title'>⚗️ pH Assessment</div>
            <div class='insight-text'>
                Your water pH is {A['PH']}.
                {"✅ Neutral & safe (6.5-8.5)" if 6.5 <= A['PH'] <= 8.5 else "⚠️ Outside safe range (6.5-8.5)"}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --------- BRANDS (must follow water) ---------
st.markdown("<h2 style='animation:fadeInUp 480ms cubic-bezier(.2,.9,.2,1) 320ms both; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🏢 Fixture Brands Selection</h2>", unsafe_allow_html=True)

if not st.session_state.analyzed:
    st.markdown("<div class='info-card'><span class='step-indicator'>2</span> <strong style='margin-left: 0.75rem;'>Complete water analysis above first, then select fixture brands</strong></div>", unsafe_allow_html=True)
    sb = fa = wb = wc = "(pending)"
else:
    st.markdown("<div class='info-card'><span class='step-indicator'>2</span> <strong style='margin-left: 0.75rem;'>Select brands for fixtures (or choose 'Other' to enter manually)</strong></div>", unsafe_allow_html=True)
    if brands:
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        
        # SwitchBoards
        with c1:
            sb = st.selectbox("🔌 SwitchBoards", options=["(none)"] + list(brands["SwitchBoards"]) + ["Other"], index=0)
            if sb == "Other":
                sb = st.text_input("Enter SwitchBoard brand", key="sb_other", placeholder="e.g., Legrand, GM...")
        
        # Faucets
        with c2:
            fa = st.selectbox("🚰 Faucets", options=["(none)"] + list(brands["Faucets"]) + ["Other"], index=0)
            if fa == "Other":
                fa = st.text_input("Enter Faucet brand", key="fa_other", placeholder="e.g., Jaquar, Kohler...")
        
        # WashBasins
        with c3:
            wb = st.selectbox("🚿 WashBasins", options=["(none)"] + list(brands["WashBasins"]) + ["Other"], index=0)
            if wb == "Other":
                wb = st.text_input("Enter WashBasin brand", key="wb_other", placeholder="e.g., Cera, Hindware...")
        
        # WC
        with c4:
            wc = st.selectbox("🚽 WC (Toilet)", options=["(none)"] + list(brands["WC"]) + ["Other"], index=0)
            if wc == "Other":
                wc = st.text_input("Enter WC brand", key="wc_other", placeholder="e.g., Parryware...")
    else:
        st.warning("⚠️ `brand_master.csv` not found. You can still type brands manually below.")
        sb = st.text_input("🔌 SwitchBoards", "")
        fa = st.text_input("🚰 Faucets", "")
        wb = st.text_input("🚿 WashBasins", "")
        wc = st.text_input("🚽 WC (Toilet)", "")

# --------- ACTIONS ---------
st.markdown("---")
st.markdown("<h2 style='animation:fadeInUp 480ms cubic-bezier(.2,.9,.2,1) 400ms both; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>📋 Actions & Report Generation</h2>", unsafe_allow_html=True)

st.markdown("<div class='info-card'><span class='step-indicator'>3</span> <strong style='margin-left: 0.75rem;'>Save your inspection or generate a professional PDF report</strong></div>", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)

with colA:
    if st.button("💾 Save Inspection", use_container_width=True):
        if not st.session_state.analyzed:
            st.error("Analyze water first.")
        else:
            payload = {
                **st.session_state.analysis,
                "SwitchBoardBrand": sb,
                "FaucetBrand": fa,
                "WashBasinBrand": wb,
                "WCBrand": wc,
            }
            save_last(payload)
            st.success("✅ Saved! Next time the app opens, this inspection loads automatically.")

with colB:
    if st.button("🧾 Generate A4 PDF", use_container_width=True):
        if not st.session_state.analyzed:
            st.error("Analyze water first.")
        else:
            payload = {
                **st.session_state.analysis,
                "SwitchBoardBrand": sb or "-",
                "FaucetBrand": fa or "-",
                "WashBasinBrand": wb or "-",
                "WCBrand": wc or "-",
            }
            pdf_bytes = make_pdf_bytes(payload)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="Water_Inspection_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

with colC:
    if st.button("🔁 Reset Form", use_container_width=True):
        for k in ["analyzed", "analysis"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

# --------- Footer ---------
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(16,185,129,0.08) 100%); padding: 2.5rem 2rem; border-radius: 16px; text-align: center; border:2px solid rgba(102,126,234,0.2); animation:slideUp 1000ms cubic-bezier(.2,.9,.2,1) both; position: relative; overflow: hidden;'>
  <div style='position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #10b981 100%);'></div>
  <p style='margin: 0 0 1rem 0; color: #0a1f3a; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.5px;'>✨ Built for Excellence</p>
  <p style='margin: 0 0 0.75rem 0; color: #555; font-weight: 500; font-size: 0.95rem; line-height: 1.7;'>A modern water quality analysis platform engineered with <strong style=\"color: #667eea;\">precision, performance & accessibility</strong> in mind</p>
  <p style='margin: 0.5rem 0 0 0; color: #999; font-weight: 400; font-size: 0.85rem; letter-spacing: 0.3px;'>🔬 ML-Powered • 📊 Real-Time Analysis • 📄 PDF Reports</p>
  <p style='margin: 1rem 0 0 0; padding-top: 1rem; border-top: 1px solid rgba(102,126,234,0.2); color: #999; font-weight: 400; font-size: 0.8rem;'>Crafted with passion for clean water & clean code</p>
</div>
""", unsafe_allow_html=True)