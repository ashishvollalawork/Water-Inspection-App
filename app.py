import os
import io
import json
import time
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

# ---- PDF (ReportLab) ----
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader  # to draw images from bytes

# ------------- App Constants -------------
ARTIFACT_MODEL = "artifacts/drinkstatus_model.pkl"
ARTIFACT_LE = "artifacts/drinkstatus_label_encoder.pkl"
BRAND_CSV = "brand_master.csv"
LAST_JSON = "last_inspection.json"

PAGE_TITLE = "💧 Water Inspection (ML + Offline)"
PAGE_ICON = "💧"

# ------------- Streamlit Page Setup -------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------- Custom Styling -------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Poppins', sans-serif; }
:root{
    --primary:#0b63c7; --accent:#ff9800; --success:#19a55a; --warning:#f59e0b;
    --danger:#d92d20; --info:#06b6d4; --muted:#5f6b7a; --bg:#f1f3f5; --surface:#ffffff;
  --radius:12px; --easing:cubic-bezier(.2,.9,.2,1);
}
html, body { background: var(--bg); min-height: 100vh; }
@keyframes fadeInUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{from{opacity:0;transform:scale(0.99)}to{opacity:1;transform:scale(1)}}
@keyframes slideInLeft{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
@keyframes slideInRight{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
@keyframes float{0%,100%{transform:translateY(0px)}50%{transform:translateY(-10px)}}

.stApp { background: transparent !important; }

/* Force readable text contrast on light background */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
.stSubheader,
.stCaption,
.stMarkdown,
.stAlert {
    color:#1b2a41 !important;
}

/* File uploader readability */
[data-testid="stFileUploaderDropzone"] {
    background:#f6f9ff !important;
    border:2px dashed #9fb6d8 !important;
    border-radius:12px !important;
}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderFileName"] {
    color:#1b2a41 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background:#0b63c7 !important;
    color:#ffffff !important;
    border:1px solid #0b63c7 !important;
    border-radius:10px !important;
    font-weight:700 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background:#0953a7 !important;
    border-color:#0953a7 !important;
}

.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4,
.block-container p,
.block-container label,
.block-container span {
    color:#1b2a41;
}

.stNumberInput > div > div > input,
.stTextInput > div > input {
    border-radius:12px !important; border:2px solid #c7d3e3 !important;
    padding:14px 18px !important; background:#ffffff !important;
    animation:slideInLeft 450ms var(--easing) both; transition:all 220ms var(--easing) !important;
  font-size:1.05rem !important; font-weight:500 !important; backdrop-filter:blur(10px);
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > input:focus {
    border-color:var(--primary) !important; background:#fff !important;
    box-shadow:0 0 0 3px rgba(11,99,199,0.12) !important;
    transform:translateY(-1px) !important;
}

.stButton > button {
    border-radius:12px !important; padding:13px 24px !important; font-weight:800 !important;
    font-size:1rem !important; letter-spacing:0.2px !important; transition:all 220ms var(--easing) !important;
    animation:popIn 700ms cubic-bezier(0.25,0.46,0.45,0.94) both;
    background:linear-gradient(135deg,#2478da 0%,#0b63c7 100%) !important;
  border:none !important; color:#fff !important; box-shadow:0 8px 24px rgba(102,126,234,0.3),inset 0 1px 0 rgba(255,255,255,0.15) !important;
  position:relative; overflow:hidden; cursor:pointer;
}
.stButton > button:hover { transform:translateY(-2px) !important; }

/* Keep download button colors consistent on hover/focus */
.stDownloadButton > button {
    border-radius:12px !important;
    background:linear-gradient(135deg,#2478da 0%,#0b63c7 100%) !important;
    color:#ffffff !important;
    border:none !important;
    font-weight:800 !important;
}
.stDownloadButton > button:hover,
.stDownloadButton > button:focus,
.stDownloadButton > button:active {
    background:linear-gradient(135deg,#1e6fcd 0%,#0953a7 100%) !important;
    color:#ffffff !important;
    border:none !important;
    box-shadow:0 0 0 3px rgba(11,99,199,0.18) !important;
}

.metric-card {
    border-radius:14px !important; box-shadow:none;
    transition:all 200ms var(--easing); animation:fadeInUp 500ms var(--easing) both;
    background:#ffffff;
    border:2px solid #d5deea; position:relative; overflow:hidden;
    color:#1b2a41 !important;
}
.metric-number {
    font-size:2.4rem !important; font-weight:900 !important;
    color:#2f58b8 !important;
    animation:none;
}

.stSelectbox > div > div {
    border-radius:12px !important; border:2px solid #c7d3e3 !important;
  transition:all 250ms var(--easing);
    background:#ffffff !important;
}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] *,
.stSelectbox [role="combobox"],
.stSelectbox [role="combobox"] * {
    color:#1b2a41 !important;
}
.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus { border-color:var(--primary) !important; }

.section-title {
    color:#0b63c7 !important;
    font-size:1.35rem;
    font-weight:800;
    margin:0.4rem 0 0.8rem;
    border-bottom:3px solid #0b63c7;
    padding-bottom:0.35rem;
}

.section-help {
    color:#5f6b7a !important;
    font-size:0.96rem;
    margin-top:-0.45rem;
    margin-bottom:0.75rem;
}

.upload-help {
    background:#eef5ff;
    border:1px solid #c9dbf6;
    color:#1b2a41;
    border-radius:10px;
    padding:0.6rem 0.75rem;
    font-size:0.93rem;
    font-weight:600;
    margin:0.35rem 0 0.55rem;
}

.header-banner {
    background:linear-gradient(135deg,#0b63c7 0%,#1253a8 100%) !important;
    border-radius:14px !important; padding:2.2rem !important; margin-bottom:0.8rem !important;
  animation:popIn 600ms var(--easing) both; box-shadow:0 16px 40px rgba(102,126,234,0.25),inset 0 1px 0 rgba(255,255,255,0.1);
  position:relative; overflow:hidden;
}

</style>
""", unsafe_allow_html=True)

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
        return data if data else None
    except Exception:
        return None  # corrupted or unreadable

def save_last(payload: dict):
    with open(LAST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def ensure_last_json_ui():
    if not os.path.exists(LAST_JSON):
        st.info("`last_inspection.json` not found. You can create it now (optional).")
        if st.button("Create last_inspection.json", type="secondary"):
            save_last({})
            st.success("Created `last_inspection.json`. It will store your last saved inspection.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

def compute_ph_status(ph: float) -> str:
    # App-local bands mirroring common guidance; adjust if you have different policy
    if ph < 6.5:
        return "Low"
    if ph > 8.5:
        return "High"
    return "Normal"

def compute_water_quality(tds: float) -> str:
    """
    Quality label tuned to match the example interface:
    0-150 = Good, 151-300 = Fair, 301-500 = Poor, 501-900 = Moderate Quality, >900 = Very Poor
    (Adjust ranges as per your policy/standards)
    """
    if tds <= 150:
        return "Good"
    elif tds <= 300:
        return "Fair"
    elif tds <= 500:
        return "Poor"
    elif tds <= 900:
        return "Moderate Quality"  # aligns with sample report showing 700 mg/L → "Moderate Quality"
    else:
        return "Very Poor"

def compute_risk_and_reco(drink_label: str, tds: float, ph: float):
    # Coarse risk logic using both ML output and ranges
    if drink_label.lower() == "safe" and (0 <= tds <= 500) and (6.5 <= ph <= 8.5):
        return "Low", "Water is acceptable for drinking."
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
            if isinstance(proba_all, list):
                proba_all = proba_all[0]
            if len(le.classes_) == proba_all.shape[1] and "Safe" in le.classes_:
                safe_idx = list(le.classes_).index("Safe")
                proba = float(proba_all[:, safe_idx])
            elif proba_all.shape[1] == 2:
                proba = float(proba_all[:, 1])
        except Exception:
            proba = None

    return label, proba

def wrap_text(canvas_obj, text, left, y, max_chars=95, line_gap=6*mm):
    for i in range(0, len(text), max_chars):
        canvas_obj.drawString(left, y, text[i:i+max_chars])
        y -= line_gap
    return y

def make_pdf_bytes(payload: dict) -> bytes:
    """
    Build an A4 PDF with the required order:
      1) Water Analysis (TDS, pH, results, recommendation)
      2) Fixture Brands
    No inspector name or timestamp included (per requirement).
    Meter images are optional (TDSImageBytes, PHImageBytes).
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
    draw_h1("Water Quality Report")

    # Section 1: Water Analysis
    draw_h2("1) Water Analysis")
    draw_kv("TDS (mg/L)", payload.get("TDS"))
    draw_kv("pH", payload.get("PH"))
    draw_kv("Water Quality", payload.get("WaterQuality"))
    draw_kv("pH Status", payload.get("PHStatus"))
    draw_kv("Drinking Status", payload.get("DrinkingStatus"))
    draw_kv("Risk Level", payload.get("RiskLevel"))

    # Recommendation (basic wrap)
    c.setFont("Helvetica", 11)
    rec_text = f"Recommendation: {payload.get('Recommendation', '')}"
    y = wrap_text(c, rec_text, left, line, max_chars=95)
    line = y - 2 * mm

    # Meter images (optional)
    tds_img_bytes = payload.get("TDSImageBytes")
    ph_img_bytes = payload.get("PHImageBytes")
    img_w = 70 * mm
    img_h = 45 * mm
    gap_x = 12 * mm

    if tds_img_bytes or ph_img_bytes:
        draw_h2("Meter Readings")
        img_y = line - img_h
        x1 = left
        x2 = left + img_w + gap_x

        if tds_img_bytes:
            try:
                c.drawImage(ImageReader(io.BytesIO(tds_img_bytes)), x1, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 10)
                c.drawString(x1, img_y - 4 * mm, "TDS Meter Reading")
            except Exception:
                pass

        if ph_img_bytes:
            try:
                c.drawImage(ImageReader(io.BytesIO(ph_img_bytes)), x2, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 10)
                c.drawString(x2, img_y - 4 * mm, "pH Meter Reading")
            except Exception:
                pass

        line = (img_y - 8 * mm)

    # BIS note (to mirror interface content)
    c.setFont("Helvetica-Oblique", 10)
    bis_note = "Results based on BIS IS 10500:2012 Indian Standards for Drinking Water"
    c.drawString(left, line, bis_note)
    line -= 10 * mm

    # Section 2: Fixture Brands
    draw_h2("2) Selected Product Brands")
    draw_kv("SwitchBoards", payload.get("SwitchBoardBrand", "-"))
    draw_kv("Faucets", payload.get("FaucetBrand", "-"))
    draw_kv("WashBasins", payload.get("WashBasinBrand", "-"))
    draw_kv("WC", payload.get("WCBrand", "-"))

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def get_safe_index(options, value):
    if value in options:
        return options.index(value)
    return 0

def _extract_numeric_candidates(text: str):
    if not text:
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]

def read_meter_value_from_image(uploaded_file, meter_type: str):
    """Read numeric meter value from uploaded image using OCR."""
    if uploaded_file is None:
        return None, "No image uploaded."

    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageFilter
    except Exception:
        return None, "OCR not available. Install pytesseract and Tesseract OCR."

    try:
        raw = uploaded_file.getvalue()
        if not raw:
            return None, "Empty image file."

        base = Image.open(io.BytesIO(raw)).convert("L")
        boosted = ImageOps.autocontrast(base)
        thresh = boosted.point(lambda p: 255 if p > 145 else 0)
        sharpened = boosted.filter(ImageFilter.SHARPEN)
        candidates = []
        whitelist_cfg = "-c tessedit_char_whitelist=0123456789."
        configs = [f"--psm 6 {whitelist_cfg}", f"--psm 7 {whitelist_cfg}"]

        for img in [base, boosted, thresh, sharpened]:
            for cfg in configs:
                text = pytesseract.image_to_string(img, config=cfg)
                candidates.extend(_extract_numeric_candidates(text))

        if meter_type == "tds":
            valid = [v for v in candidates if 0 <= v <= 3000]
            if not valid:
                return None, "Could not detect TDS value from image."
            value = float(round(valid[0], 1))
            return value, f"Detected TDS: {value} mg/L"

        valid = [v for v in candidates if 0 <= v <= 14]
        if not valid:
            # Common OCR issue: reads pH 7.1 as 71
            scaled = [v / 10.0 for v in candidates if 0 <= v <= 140]
            valid = [v for v in scaled if 0 <= v <= 14]
        if not valid:
            return None, "Could not detect pH value from image."
        value = float(round(valid[0], 1))
        return value, f"Detected pH: {value}"

    except Exception:
        return None, "OCR failed. Ensure image is clear and Tesseract is installed."

# ------------- UI -------------

# Header banner
st.markdown("""
<div class='header-banner'>
  <div style='position: relative; z-index: 1; text-align: center;'>
    <h1 style='margin:0; font-size:3rem; color:#fff;'>💧 Water Inspection</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.75rem 0 0; font-size: 1.15rem; font-weight: 500;'>
      Professional Water Quality Assessment with AI
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# BIS reference (as in sample interface)
st.caption("✓ Results based on **BIS IS 10500:2012** Indian Standards for Drinking Water.")  # mirrors sample
# (This mirrors the text shown in your PDF UI)  # cite: the displayed interface in the uploaded PDF

# Ensure last-inspection file exists (optional)
ensure_last_json_ui()

# Load resources
model, le = load_model_and_encoder()
brands = load_brand_master()
last = load_last()

# Apply deferred OCR updates before widgets are created.
if st.session_state.pop("pending_ocr_apply", False):
    if "ocr_tds_value" in st.session_state:
        st.session_state["tds_input"] = float(st.session_state.pop("ocr_tds_value"))
    if "ocr_ph_value" in st.session_state:
        st.session_state["ph_input"] = float(st.session_state.pop("ocr_ph_value"))
    st.session_state.pop("analysis_result", None)

# Apply deferred reset before widgets are instantiated.
if st.session_state.pop("pending_form_reset", False):
    st.session_state["tds_input"] = 0.0
    st.session_state["ph_input"] = 7.0
    st.session_state["brand_switch"] = "-"
    st.session_state["brand_faucet"] = "-"
    st.session_state["brand_wash"] = "-"
    st.session_state["brand_wc"] = "-"
    st.session_state.pop("tds_img", None)
    st.session_state.pop("ph_img", None)
    st.session_state.pop("analysis_result", None)

# Initialize widget state once so Reset can reliably restore defaults.
if "tds_input" not in st.session_state:
    st.session_state["tds_input"] = float(last.get("TDS", 0.0)) if last else 0.0
if "ph_input" not in st.session_state:
    st.session_state["ph_input"] = float(last.get("PH", 7.0)) if last else 7.0

switch_options = ["-"] + brands.get("SwitchBoards", [])
faucet_options = ["-"] + brands.get("Faucets", [])
wash_options = ["-"] + brands.get("WashBasins", [])
wc_options = ["-"] + brands.get("WC", [])

if "brand_switch" not in st.session_state:
    st.session_state["brand_switch"] = last.get("SwitchBoardBrand", "-") if last else "-"
if st.session_state["brand_switch"] not in switch_options:
    st.session_state["brand_switch"] = "-"

if "brand_faucet" not in st.session_state:
    st.session_state["brand_faucet"] = last.get("FaucetBrand", "-") if last else "-"
if st.session_state["brand_faucet"] not in faucet_options:
    st.session_state["brand_faucet"] = "-"

if "brand_wash" not in st.session_state:
    st.session_state["brand_wash"] = last.get("WashBasinBrand", "-") if last else "-"
if st.session_state["brand_wash"] not in wash_options:
    st.session_state["brand_wash"] = "-"

if "brand_wc" not in st.session_state:
    st.session_state["brand_wc"] = last.get("WCBrand", "-") if last else "-"
if st.session_state["brand_wc"] not in wc_options:
    st.session_state["brand_wc"] = "-"

# ---------- Inputs: Water Parameters & Images ----------
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("<div class='section-title'>🧪 Water Test Input</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-help'>Step 1: Enter TDS and pH values. Add meter photos if available.</div>", unsafe_allow_html=True)
    tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=3000.0, step=1.0, key="tds_input")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, format="%.1f", key="ph_input")

    st.markdown("**📷 Meter Readings (Optional)**")
    st.markdown(
        "<div class='upload-help'>Upload clear close-up photos of the meter display. Accepted formats: PNG, JPG, JPEG, WEBP.</div>",
        unsafe_allow_html=True,
    )
    tds_img = st.file_uploader("Upload TDS meter photo (PNG/JPG/JPEG/WEBP)", type=["png","jpg","jpeg","webp"], key="tds_img")
    st.caption("TDS image: meter display with numeric TDS value visible.")
    ph_img = st.file_uploader("Upload pH meter photo (PNG/JPG/JPEG/WEBP)", type=["png","jpg","jpeg","webp"], key="ph_img")
    st.caption("pH image: meter display with pH number clearly visible.")

    if st.button("📸 Read values from images", use_container_width=True):
        tds_val, tds_msg = read_meter_value_from_image(tds_img, "tds")
        ph_val, ph_msg = read_meter_value_from_image(ph_img, "ph")
        st.session_state["ocr_feedback"] = f"TDS: {tds_msg} | pH: {ph_msg}"

        if tds_val is not None:
            st.session_state["ocr_tds_value"] = tds_val
        if ph_val is not None:
            st.session_state["ocr_ph_value"] = ph_val

        if (tds_val is not None) or (ph_val is not None):
            st.session_state["pending_ocr_apply"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    if "ocr_feedback" in st.session_state:
        st.info(st.session_state["ocr_feedback"])

    # Preview images
    prev_cols = st.columns(2)
    if tds_img:
        with prev_cols[0]:
            st.image(tds_img, caption="TDS Meter Reading", use_container_width=True)
    if ph_img:
        with prev_cols[1]:
            st.image(ph_img, caption="pH Meter Reading", use_container_width=True)

    if not tds_img and not ph_img:
        st.caption("Tip: Add meter photos to include them in your PDF report.")

    analyze_clicked = st.button("🔎 Analyze Water", type="primary", use_container_width=True)

    if analyze_clicked:
        drink_label, proba_safe = predict_drinking_status(model, le, tds, ph)
        ph_status = compute_ph_status(ph)
        water_quality = compute_water_quality(tds)
        risk_level, recommendation = compute_risk_and_reco(drink_label, tds, ph)
        st.session_state["analysis_result"] = {
            "tds": float(tds),
            "ph": float(ph),
            "drink_label": drink_label,
            "proba_safe": proba_safe,
            "ph_status": ph_status,
            "water_quality": water_quality,
            "risk_level": risk_level,
            "recommendation": recommendation,
        }

# Live values are used for save/PDF defaults; report view is shown only after Analyze click.
live_drink_label, live_proba_safe = predict_drinking_status(model, le, tds, ph)
live_ph_status = compute_ph_status(ph)
live_water_quality = compute_water_quality(tds)
live_risk_level, live_recommendation = compute_risk_and_reco(live_drink_label, tds, ph)

analysis_result = st.session_state.get("analysis_result")
if analysis_result is None:
    current_tds = float(tds)
    current_ph = float(ph)
    current_drink_label = live_drink_label
    current_proba_safe = live_proba_safe
    current_ph_status = live_ph_status
    current_water_quality = live_water_quality
    current_risk_level = live_risk_level
    current_recommendation = live_recommendation
else:
    current_tds = float(analysis_result.get("tds", tds))
    current_ph = float(analysis_result.get("ph", ph))
    current_drink_label = analysis_result["drink_label"]
    current_proba_safe = analysis_result["proba_safe"]
    current_ph_status = analysis_result["ph_status"]
    current_water_quality = analysis_result["water_quality"]
    current_risk_level = analysis_result["risk_level"]
    current_recommendation = analysis_result["recommendation"]

with right_col:
    st.markdown("<div class='section-title'>📊 Water Quality Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-help'>Step 2: Review drinking safety, risk level, and recommendation.</div>", unsafe_allow_html=True)

    if analysis_result is None:
        st.info("Click Analyze Water to view water quality report.")
    else:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("<div class='metric-card' style='padding:1.2rem'>"
                        "<div style='font-weight:700;color:#0a1f3a'>TDS Level</div>"
                        f"<div class='metric-number'>{int(current_tds)} mg/L</div>"
                        f"<div style='margin-top:6px;font-weight:700;'>{current_water_quality}</div>"
                        "</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-card' style='padding:1.2rem'>"
                        "<div style='font-weight:700;color:#0a1f3a'>pH Level</div>"
                        f"<div class='metric-number'>{current_ph:.1f}</div>"
                        f"<div style='margin-top:6px;font-weight:700;'>{current_ph_status}</div>"
                        "</div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-card' style='padding:1.2rem;margin-top:0.8rem;color:#1b2a41;'>"
                    f"<div style='font-weight:700;'><b>Drinking Status:</b> {current_drink_label}"
                    f"{f' (Safe probability: {current_proba_safe*100:.1f}%)' if current_proba_safe is not None else ''}</div>"
                    f"<div style='margin-top:6px;font-weight:700;'><b>Risk Level:</b> {current_risk_level}</div>"
                    f"<div style='margin-top:6px;'><b>Recommendation:</b> {current_recommendation}</div>"
                    "</div>", unsafe_allow_html=True)

st.divider()

# ---------- Brands Section ----------
st.markdown("<div class='section-title'>🏪 Selected Product Brands</div>", unsafe_allow_html=True)
st.markdown("<div class='section-help'>Step 3: Select brand details to include in the report.</div>", unsafe_allow_html=True)
b_cols = st.columns(4)
with b_cols[0]:
    brand_switch = st.selectbox(
        "Electrical Switch",
        options=switch_options,
        key="brand_switch"
    )
with b_cols[1]:
    brand_faucet = st.selectbox(
        "Faucet",
        options=faucet_options,
        key="brand_faucet"
    )
with b_cols[2]:
    brand_wash = st.selectbox(
        "Wash Basin",
        options=wash_options,
        key="brand_wash"
    )
with b_cols[3]:
    brand_wc = st.selectbox(
        "Commode (WC)",
        options=wc_options,
        key="brand_wc"
    )

# Persist / recall
persist_cols = st.columns(3)
with persist_cols[0]:
    if st.button("💾 Save as Last Inspection", use_container_width=True):
        payload = {
            "TDS": float(tds), "PH": float(ph),
            "WaterQuality": current_water_quality,
            "PHStatus": current_ph_status,
            "DrinkingStatus": current_drink_label,
            "RiskLevel": current_risk_level,
            "Recommendation": current_recommendation,
            "SwitchBoardBrand": brand_switch if brand_switch != "-" else "",
            "FaucetBrand": brand_faucet if brand_faucet != "-" else "",
            "WashBasinBrand": brand_wash if brand_wash != "-" else "",
            "WCBrand": brand_wc if brand_wc != "-" else ""
        }
        save_last(payload)
        st.success("Inspection saved successfully.")

with persist_cols[1]:
    if st.button("📂 Load Last Inspection", use_container_width=True):
        loaded = load_last()
        if loaded:
            st.info("Loaded last inspection. Please re-select brands if list changed.")
            # Rerun to reflect values (we rely on defaults pulling from last)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.warning("No previous inspection found.")

with persist_cols[2]:
    if st.button("🧹 Reset Form", type="secondary", use_container_width=True):
        st.session_state["pending_form_reset"] = True
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

st.divider()

# ---------- PDF Generation ----------
st.markdown("<div class='section-title'>📄 Generate A4 PDF Report</div>", unsafe_allow_html=True)
st.markdown("<div class='section-help'>Step 4: Download the ready A4 report.</div>", unsafe_allow_html=True)

pdf_payload = {
    "TDS": float(current_tds),
    "PH": float(current_ph),
    "WaterQuality": current_water_quality,
    "PHStatus": current_ph_status,
    "DrinkingStatus": current_drink_label,
    "RiskLevel": current_risk_level,
    "Recommendation": current_recommendation,
    "SwitchBoardBrand": brand_switch if brand_switch != "-" else "-",
    "FaucetBrand": brand_faucet if brand_faucet != "-" else "-",
    "WashBasinBrand": brand_wash if brand_wash != "-" else "-",
    "WCBrand": brand_wc if brand_wc != "-" else "-"
}

# Read images as bytes if provided
tds_img_bytes = tds_img.getvalue() if tds_img else None
ph_img_bytes = ph_img.getvalue() if ph_img else None
pdf_payload["TDSImageBytes"] = tds_img_bytes
pdf_payload["PHImageBytes"] = ph_img_bytes

pdf_bytes = make_pdf_bytes(pdf_payload)
st.download_button(
    label="⬇️ Download Water Inspection Report (A4)",
    data=pdf_bytes,
    file_name="water_inspection_report.pdf",
    mime="application/pdf",
    use_container_width=True
)

# Footer info card
st.markdown(
    "<div class='metric-card' style='padding:1rem; margin-top:1.5rem;'>"
    "This report is for informational purposes. Consult water quality experts for professional advice."
    "</div>",
    unsafe_allow_html=True,
)