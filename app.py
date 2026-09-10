import re
import csv
import io
from collections import Counter

import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# =========================================================
# CONFIGURATION — MODEL PIPELINE UNCHANGED
# =========================================================

MODEL_ID = "AbdelrahmanAkl/arabic-sentiment-compass-arabert"
MAX_LENGTH = 128

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

ARABIC_LABELS = {
    "Negative": "سلبي",
    "Neutral": "محايد",
    "Positive": "إيجابي",
}

LABELS_ORDER = [
    "Negative",
    "Neutral",
    "Positive",
]

SENTIMENT_ICONS = {
    "Negative": "−",
    "Neutral": "•",
    "Positive": "+",
}


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Arabic Sentiment Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# ORIGINAL PREPROCESSING — DO NOT CHANGE
# =========================================================

ARABIC_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)

URL_RE = re.compile(r"https?://*\S*+|www\\.*\S*+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")


def preprocess_arabic_tweet(text):
    text = str(text)
    text = URL_RE.sub(" رابط ", text)
    text = MENTION_RE.sub(" مستخدم ", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# MODEL LOADING — REAL MODEL ONLY
# =========================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
    )

    model.eval()

    return tokenizer, model


# =========================================================
# OFFICIAL PREDICTION PIPELINE — DO NOT CHANGE
# =========================================================

def predict_sentiment(text, tokenizer, model):

    processed_text = preprocess_arabic_tweet(text)

    encoded = tokenizer(
        processed_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.inference_mode():

        logits = model(**encoded).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    scores = probabilities[0]

    predicted_id = int(scores.argmax())

    predicted_label = ID2LABEL[predicted_id]

    confidence = float(
        scores[predicted_id]
    )

    probabilities_dict = {
        ID2LABEL[i]: float(scores[i])
        for i in range(len(ID2LABEL))
    }

    return (
        processed_text,
        predicted_label,
        confidence,
        probabilities_dict,
    )


# =========================================================
# SESSION STATE
# =========================================================

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "csv_results" not in st.session_state:
    st.session_state.csv_results = None

if "csv_columns" not in st.session_state:
    st.session_state.csv_columns = []

if "single_text" not in st.session_state:
    st.session_state.single_text = ""


# =========================================================
# CALLBACKS
# =========================================================

def clear_single_input():
    """Clear the single-text widget and its latest prediction safely."""
    st.session_state["single_text"] = ""
    st.session_state["single_result"] = None



# =========================================================
# PREMIUM UI — COMPLETE REDESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

:root{
    --ink:#17203a;
    --muted:#718096;
    --line:#e7eaf2;
    --surface:#ffffff;
    --bg:#f7f8fc;
    --purple:#6857f5;
    --blue:#3c82f6;
    --cyan:#19b7d8;
    --green:#23b58a;
    --coral:#ef6b78;
}

html, body, [class*="css"]{
    font-family:"IBM Plex Sans Arabic","Cairo",sans-serif;
}

.stApp{
    background:
      radial-gradient(circle at 4% 3%, rgba(104,87,245,.10), transparent 22%),
      radial-gradient(circle at 96% 8%, rgba(25,183,216,.10), transparent 24%),
      linear-gradient(180deg,#fbfcff 0%,#f7f8fc 100%);
    color:var(--ink);
}

.block-container{
    max-width:1220px;
    padding:28px 34px 60px;
}

/* hide Streamlit chrome */
#MainMenu, footer, header {visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stDecoration"]{display:none;}

/* ---------- top navigation ---------- */
.navbar{
    height:64px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 4px 0 10px;
    margin-bottom:24px;
}
.brand{
    display:flex;
    align-items:center;
    gap:12px;
}
.brand-mark{
    width:42px;height:42px;border-radius:13px;
    display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,var(--purple),var(--blue));
    color:white;font-size:21px;font-weight:800;
    box-shadow:0 10px 25px rgba(104,87,245,.22);
}
.brand-name{font-size:16px;font-weight:800;letter-spacing:-.2px;}
.brand-sub{font-size:11px;color:var(--muted);margin-top:1px;}
.status-pill{
    display:flex;align-items:center;gap:7px;
    padding:8px 13px;border:1px solid var(--line);
    background:rgba(255,255,255,.78);border-radius:999px;
    color:#566174;font-size:11px;font-weight:600;
    box-shadow:0 4px 15px rgba(24,32,58,.04);
}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);}

/* ---------- hero ---------- */
.hero{
    position:relative;overflow:hidden;
    border:1px solid #e6e8f2;
    border-radius:30px;
    padding:52px 56px 44px;
    background:
      radial-gradient(circle at 8% 10%,rgba(104,87,245,.12),transparent 30%),
      radial-gradient(circle at 96% 92%,rgba(25,183,216,.13),transparent 28%),
      rgba(255,255,255,.86);
    box-shadow:0 18px 55px rgba(36,42,72,.07);
    margin-bottom:28px;
}
.hero:after{
    content:"";
    position:absolute;right:-90px;top:-100px;
    width:250px;height:250px;border-radius:50%;
    border:1px solid rgba(104,87,245,.10);
    box-shadow:0 0 0 35px rgba(104,87,245,.035),0 0 0 70px rgba(104,87,245,.025);
}
.hero-kicker{
    display:inline-flex;align-items:center;gap:8px;
    color:#6253dc;font-size:10px;font-weight:800;
    letter-spacing:1.6px;text-transform:uppercase;
    margin-bottom:14px;
}
.hero-kicker span{
    width:6px;height:6px;border-radius:50%;
    background:var(--cyan);
}
.hero h1{
    font-family:"Cairo",sans-serif;
    font-size:clamp(34px,5vw,58px);
    line-height:1.18;
    margin:0;
    color:#151d38;
    letter-spacing:-1.8px;
    font-weight:800;
}
.hero p{
    max-width:730px;
    margin:17px 0 24px;
    color:#6d7890;
    font-size:14px;
    line-height:2;
}
.hero-meta{display:flex;flex-wrap:wrap;gap:9px;}
.meta{
    border:1px solid #e5e8f1;background:#fff;
    padding:8px 12px;border-radius:11px;
    font-size:10px;color:#657087;
    box-shadow:0 4px 12px rgba(30,40,70,.035);
}
.meta b{color:#27314a;margin-left:4px;}

/* ---------- tabs ---------- */
.stTabs{
    margin-top:6px;
}
.stTabs [data-baseweb="tab-list"]{
    gap:8px;
    border-bottom:1px solid #e3e6ef;
    background:transparent;
}
.stTabs [data-baseweb="tab"]{
    height:48px;
    border-radius:12px 12px 0 0;
    color:#7b8497;
    font-weight:700;
    font-size:12px;
    padding:0 18px;
}
.stTabs [aria-selected="true"]{
    color:#4f43d6 !important;
    background:#fff;
    box-shadow:inset 0 -2px 0 var(--purple);
}

/* ---------- section headers ---------- */
.section-head{
    display:flex;align-items:flex-end;justify-content:space-between;
    gap:20px;margin:30px 0 17px;
}
.eyebrow{
    font-size:9px;font-weight:800;letter-spacing:1.8px;
    color:#7769e8;text-transform:uppercase;margin-bottom:5px;
}
.section-title{
    font-family:"Cairo",sans-serif;
    font-size:23px;font-weight:800;color:#18213c;
}
.section-note{font-size:11px;color:#8992a4;}

/* ---------- cards ---------- */
.card{
    background:rgba(255,255,255,.94);
    border:1px solid #e5e8f0;
    border-radius:22px;
    box-shadow:0 12px 34px rgba(35,43,73,.055);
    padding:22px;
}
.card-label{
    font-size:9px;letter-spacing:1.2px;font-weight:800;
    color:#8992a5;text-transform:uppercase;margin-bottom:8px;
}
.card-title{font-size:16px;font-weight:800;color:#202a44;}
.card-sub{font-size:11px;color:#8790a3;margin-top:4px;}

/* ---------- input ---------- */
.input-shell{
    border:1px solid #e1e5ef;border-radius:20px;
    background:#fff;padding:6px;
    box-shadow:0 12px 32px rgba(40,48,80,.055);
}
textarea{
    font-family:"IBM Plex Sans Arabic","Cairo",sans-serif !important;
    font-size:16px !important;
    color:#1d2741 !important;
}
[data-testid="stTextArea"] textarea{
    border:0 !important;
    box-shadow:none !important;
    background:#fff !important;
    border-radius:15px !important;
    min-height:190px !important;
    padding:17px !important;
}
[data-testid="stTextArea"] textarea:focus{
    border:0 !important;
    box-shadow:0 0 0 2px rgba(104,87,245,.10) !important;
}

/* ---------- buttons ---------- */
.stButton > button{
    border-radius:13px !important;
    min-height:46px !important;
    font-family:"IBM Plex Sans Arabic","Cairo",sans-serif !important;
    font-weight:700 !important;
    font-size:12px !important;
    border:1px solid #e1e5ee !important;
    background:#fff !important;
    color:#364058 !important;
    box-shadow:0 5px 15px rgba(30,40,70,.04) !important;
    transition:.2s ease !important;
}
.stButton > button:hover{
    border-color:#bfc5ff !important;
    transform:translateY(-1px);
}
.primary-btn .stButton > button{
    background:linear-gradient(135deg,#6958f5,#497ff3) !important;
    color:#fff !important;
    border:0 !important;
    box-shadow:0 10px 24px rgba(94,83,235,.23) !important;
}

/* ---------- result card ---------- */
.result-card{
    position:relative;overflow:hidden;
    min-height:386px;
    display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    text-align:center;
    border-radius:26px;
    border:1px solid #e3e7f0;
    background:
      radial-gradient(circle at 50% 20%,rgba(104,87,245,.08),transparent 34%),
      #fff;
    box-shadow:0 16px 45px rgba(35,43,73,.065);
}
.result-card:before{
    content:"";
    position:absolute;left:0;right:0;top:0;height:4px;
    background:linear-gradient(90deg,var(--purple),var(--blue),var(--cyan));
}
.result-caption{font-size:9px;letter-spacing:1.4px;color:#8b94a7;font-weight:800;}
.result-label{
    font-family:"Cairo",sans-serif;font-size:30px;
    font-weight:800;color:#18213c;margin-top:6px;
}
.result-en{font-size:11px;color:#8a93a6;}
.confidence{
    margin-top:15px;padding:8px 15px;border-radius:999px;
    background:#f2efff;color:#5d50d8;border:1px solid #e3ddff;
    font-size:11px;font-weight:800;
}

/* ---------- result visual ---------- */
.result-icon{
    width:82px;height:82px;border-radius:24px;
    display:flex;align-items:center;justify-content:center;
    margin:10px 0 18px;
    background:linear-gradient(135deg,#f0eeff,#eaf9fc);
    border:1px solid #e0e3f4;
    box-shadow:0 10px 25px rgba(48,58,96,.08);
    font-size:30px;font-weight:800;color:#6456e8;
}
/* ---------- probability ---------- */
.prob-card{
    background:#fff;border:1px solid #e5e8f0;border-radius:18px;
    padding:17px 18px;box-shadow:0 8px 24px rgba(35,43,73,.045);
    height:100%;
}
.prob-top{display:flex;justify-content:space-between;align-items:center;}
.prob-name{font-size:12px;font-weight:800;color:#27314a;}
.prob-en{font-size:9px;color:#8d95a7;}
.prob-value{font-size:19px;font-weight:800;color:#17203a;}
.bar{height:7px;border-radius:99px;background:#edf0f5;margin-top:13px;overflow:hidden;}
.bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#6b5bf4,#25b9d2);}

/* ---------- pipeline ---------- */
.pipeline{
    display:flex;align-items:stretch;gap:8px;
    overflow-x:auto;padding:5px 1px 10px;
}
.step{
    min-width:150px;flex:1;
    background:#fff;border:1px solid #e5e8f0;border-radius:17px;
    padding:16px;text-align:center;
}
.step-num{
    width:30px;height:30px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    margin:0 auto 9px;
    background:#f1efff;color:#5f52db;font-size:11px;font-weight:800;
}
.step-title{font-size:12px;font-weight:800;color:#27314a;}
.step-desc{font-size:9px;color:#8a93a5;margin-top:4px;line-height:1.7;}
.arrow{display:flex;align-items:center;color:#b2b8c8;font-size:17px;}

/* ---------- info/stat ---------- */
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.info-box{
    background:#fff;border:1px solid #e5e8f0;border-radius:17px;padding:17px;
}
.info-k{font-size:9px;color:#8992a4;text-transform:uppercase;letter-spacing:1px;}
.info-v{font-size:15px;font-weight:800;color:#202a44;margin-top:6px;}
.info-s{font-size:9px;color:#8a93a5;margin-top:3px;}
.callout{
    border:1px solid #e4e0ff;background:#f8f7ff;
    border-radius:17px;padding:16px 18px;color:#59637a;
    font-size:11px;line-height:1.9;
}

/* ---------- upload ---------- */
[data-testid="stFileUploader"]{
    border:1px dashed #cfd4e2 !important;
    border-radius:17px !important;
    background:#fafbfe !important;
    padding:8px !important;
}

/* ---------- dataframe ---------- */
[data-testid="stDataFrame"]{
    border:1px solid #e4e7ef;
    border-radius:16px;
    overflow:hidden;
}

/* ---------- footer ---------- */
.footer{
    margin-top:40px;padding-top:22px;border-top:1px solid #e4e7ef;
    display:flex;justify-content:space-between;gap:15px;
    color:#929aaa;font-size:10px;
}
.footer strong{color:#566075;}

/* ---------- responsive ---------- */
@media(max-width:850px){
    .block-container{padding:18px 15px 40px;}
    .hero{padding:34px 24px;}
    .hero h1{font-size:34px;}
    .info-grid{grid-template-columns:1fr;}
    .navbar{margin-bottom:14px;}
}
</style>
""", unsafe_allow_html=True)


# ---------- top nav ----------
st.markdown("""
<div class="navbar">
  <div class="brand">
    <div class="brand-mark">🧭</div>
    <div>
      <div class="brand-name">Arabic Sentiment Compass</div>
      <div class="brand-sub">Arabic NLP • AraBERT • Real Inference</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span> Real model online</div>
</div>
""", unsafe_allow_html=True)


# ---------- hero ----------
st.markdown("""
<div class="hero" dir="rtl">
  <div class="hero-kicker"><span></span> AI SENTIMENT INTELLIGENCE</div>
  <h1>بوصلة المشاعر العربية</h1>
  <p>
    حلّل المشاعر في النصوص العربية باستخدام نموذج <b>AraBERT</b> حقيقي،
    مع عرض النتيجة ودرجة الثقة وتوزيع الاحتمالات بشكل واضح وسريع.
  </p>
  <div class="hero-meta" dir="ltr">
    <div class="meta"><b>AraBERT</b> Model</div>
    <div class="meta"><b>3</b> Classes</div>
    <div class="meta"><b>128</b> Max Length</div>
    <div class="meta"><b>Real</b> Inference</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------- tabs ----------
tab_text, tab_csv, tab_about = st.tabs(
    ["✦ تحليل النص", "▦ تحليل CSV", "◈ عن النظام"]
)


# =========================================================
# TAB 1 — SINGLE TEXT
# =========================================================
with tab_text:
    st.markdown("""
    <div class="section-head" dir="rtl">
      <div>
        <div class="eyebrow">SINGLE TEXT ANALYSIS</div>
        <div class="section-title">اكتشف المشاعر في أي نص عربي</div>
      </div>
      <div class="section-note">أدخل النص ← حلّل ← استكشف القرار</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.03, .97], gap="large")

    with left:
        st.markdown("""
        <div class="card" dir="rtl">
          <div class="card-label">Arabic Text</div>
          <div class="card-title">النص المراد تحليله</div>
          <div class="card-sub">اكتب جملة أو تغريدة أو رأيًا باللغة العربية.</div>
        </div>
        """, unsafe_allow_html=True)

        user_text = st.text_area(
            "Arabic text",
            placeholder="مثال: الخدمة كانت ممتازة والتجربة أفضل مما توقعت ✨",
            height=190,
            label_visibility="collapsed",
            key="single_text",
        )

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        analyze_button = st.button(
            "✦  تحليل المشاعر",
            use_container_width=True,
            key="analyze_single",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Clear is intentionally separated from the primary action.
        clear_button = st.button(
            "مسح النص",
            use_container_width=True,
            key="clear_single",
            on_click=clear_single_input,
        )

        if analyze_button:
            if not user_text.strip():
                st.warning("اكتب نصًا عربيًا أولًا ثم اضغط تحليل المشاعر.")
            else:
                try:
                    with st.spinner("جاري تحليل النص باستخدام AraBERT..."):
                        tokenizer, model = load_model()
                        processed_text, predicted_label, confidence, probabilities_dict = predict_sentiment(
                            user_text, tokenizer, model
                        )
                    st.session_state.single_result = {
                        "processed_text": processed_text,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "probabilities_dict": probabilities_dict,
                    }
                except Exception as exc:
                    st.error("حدث خطأ أثناء تشغيل النموذج.")
                    st.exception(exc)

        result = st.session_state.get("single_result")

    with right:
        if not result:
            st.markdown("""
            <div class="result-card" dir="rtl">
              <div class="result-caption">AI SENTIMENT RESULT</div>
              <div class="result-icon">✦</div>
              <div class="result-label" style="font-size:21px;">جاهز للتحليل</div>
              <div class="result-en">Your sentiment result will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            label = result["predicted_label"]
            ar_label = ARABIC_LABELS[label]
            confidence = result["confidence"] * 100

            st.markdown(f"""
            <div class="result-card" dir="rtl">
              <div class="result-caption">AI SENTIMENT RESULT</div>
              <div class="result-icon">✦</div>
              <div class="result-label">{ar_label}</div>
              <div class="result-en">Predicted Sentiment · {label}</div>
              <div class="confidence">درجة الثقة {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    if result:
        st.markdown("""
        <div class="section-head" dir="rtl">
          <div>
            <div class="eyebrow">PROBABILITY DISTRIBUTION</div>
            <div class="section-title">توزيع احتمالات المشاعر</div>
          </div>
          <div class="section-note">Softmax scores from the real model</div>
        </div>
        """, unsafe_allow_html=True)

        pcols = st.columns(3, gap="medium")
        for col, label in zip(pcols, LABELS_ORDER):
            value = result["probabilities_dict"][label] * 100
            ar = ARABIC_LABELS[label]
            icon = SENTIMENT_ICONS[label]
            with col:
                st.markdown(f"""
                <div class="prob-card" dir="rtl">
                  <div class="prob-top">
                    <div>
                      <div class="prob-name">{ar} <span class="prob-en">{label}</span></div>
                    </div>
                    <div style="font-size:17px;color:#7a83f4;">{icon}</div>
                  </div>
                  <div class="prob-value" style="margin-top:12px;">{value:.2f}%</div>
                  <div class="bar"><div class="bar-fill" style="width:{max(0,min(100,value)):.2f}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-head" dir="rtl">
          <div>
            <div class="eyebrow">HOW THE AI DECIDED</div>
            <div class="section-title">كيف وصل النموذج إلى النتيجة؟</div>
          </div>
        </div>
        <div class="pipeline" dir="ltr">
          <div class="step"><div class="step-num">01</div><div class="step-title">User Input</div><div class="step-desc">النص العربي الذي أدخلته</div></div>
          <div class="arrow">→</div>
          <div class="step"><div class="step-num">02</div><div class="step-title">Preprocessing</div><div class="step-desc">تنظيف وتطبيع النص</div></div>
          <div class="arrow">→</div>
          <div class="step"><div class="step-num">03</div><div class="step-title">Tokenizer</div><div class="step-desc">تحويل النص إلى tokens</div></div>
          <div class="arrow">→</div>
          <div class="step"><div class="step-num">04</div><div class="step-title">AraBERT</div><div class="step-desc">حساب logits للنموذج الحقيقي</div></div>
          <div class="arrow">→</div>
          <div class="step"><div class="step-num">05</div><div class="step-title">Prediction</div><div class="step-desc">Softmax ثم أعلى احتمال</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("عرض النص بعد الـ preprocessing"):
            st.code(result["processed_text"], language="text")


# =========================================================
# TAB 2 — CSV
# =========================================================
with tab_csv:
    st.markdown("""
    <div class="section-head" dir="rtl">
      <div>
        <div class="eyebrow">BATCH ANALYSIS</div>
        <div class="section-title">حلّل آلاف النصوص دفعة واحدة</div>
      </div>
      <div class="section-note">CSV → AraBERT → Results</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout" dir="rtl">
      ارفع ملف CSV يحتوي على عمود نصي، اختر العمود المطلوب،
      ثم شغّل النموذج للحصول على المشاعر والاحتمالات وملف نتائج قابل للتحميل.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([.85, 1.15], gap="large")

    with left:
        st.markdown('<div class="card" dir="rtl"><div class="card-label">CSV DATASET</div><div class="card-title">رفع ملف البيانات</div><div class="card-sub">UTF-8 أو UTF-8-SIG</div></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "رفع ملف CSV",
            type=["csv"],
            key="csv_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                try:
                    csv_text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    csv_text = raw_bytes.decode("utf-8")

                reader = csv.DictReader(io.StringIO(csv_text))
                rows = list(reader)
                fieldnames = reader.fieldnames or []

                if not rows or not fieldnames:
                    st.warning("الملف فارغ أو لا يحتوي على أعمدة.")
                else:
                    text_column = st.selectbox(
                        "اختر عمود النص",
                        fieldnames,
                        key="csv_text_column",
                    )
                    st.caption(f"عدد السجلات: {len(rows):,}")

                    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                    analyze_csv_button = st.button(
                        "✦  بدء التحليل",
                        use_container_width=True,
                        key="analyze_csv",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if analyze_csv_button:
                        try:
                            with st.spinner("جاري تحليل الملف باستخدام النموذج..."):
                                tokenizer, model = load_model()
                                results = []
                                progress = st.progress(0.0)
                                total = len(rows)

                                for index, row in enumerate(rows, start=1):
                                    text_value = str(row.get(text_column, "") or "")
                                    if text_value.strip():
                                        processed_text, predicted_label, confidence, probabilities_dict = predict_sentiment(
                                            text_value, tokenizer, model
                                        )
                                    else:
                                        processed_text = ""
                                        predicted_label = "Neutral"
                                        confidence = 0.0
                                        probabilities_dict = {
                                            "Negative": 0.0,
                                            "Neutral": 0.0,
                                            "Positive": 0.0,
                                        }

                                    new_row = dict(row)
                                    new_row["processed_text"] = processed_text
                                    new_row["sentiment"] = predicted_label
                                    new_row["sentiment_ar"] = ARABIC_LABELS[predicted_label]
                                    new_row["confidence"] = round(confidence, 6)
                                    new_row["negative_probability"] = round(probabilities_dict["Negative"], 6)
                                    new_row["neutral_probability"] = round(probabilities_dict["Neutral"], 6)
                                    new_row["positive_probability"] = round(probabilities_dict["Positive"], 6)
                                    results.append(new_row)
                                    progress.progress(index / total)

                            st.session_state.csv_results = results
                            st.session_state.csv_columns = list(results[0].keys()) if results else []
                            st.success("تم تحليل الملف بنجاح.")
                        except Exception as exc:
                            st.error("حدث خطأ أثناء تحليل ملف CSV.")
                            st.exception(exc)
            except Exception as exc:
                st.error("تعذر قراءة ملف CSV.")
                st.exception(exc)

    with right:
        csv_results = st.session_state.get("csv_results")

        if not csv_results:
            st.markdown("""
            <div class="result-card" style="min-height:330px;" dir="rtl">
              <div style="font-size:42px;margin-bottom:10px;">▦</div>
              <div class="result-label" style="font-size:22px;">بانتظار ملف البيانات</div>
              <div class="result-en">Upload a CSV to see batch sentiment analytics</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            counts = Counter(row.get("sentiment", "Neutral") for row in csv_results)
            total_rows = len(csv_results)
            stats = [
                ("إجمالي السجلات", f"{total_rows:,}", "Total"),
                ("إيجابي", f"{counts.get('Positive',0)/total_rows*100:.1f}%", f"{counts.get('Positive',0):,} سجل"),
                ("محايد", f"{counts.get('Neutral',0)/total_rows*100:.1f}%", f"{counts.get('Neutral',0):,} سجل"),
                ("سلبي", f"{counts.get('Negative',0)/total_rows*100:.1f}%", f"{counts.get('Negative',0):,} سجل"),
            ]
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            # Render through columns so cards remain responsive in Streamlit.
            cols = st.columns(4, gap="small")
            for col, (k, v, s) in zip(cols, stats):
                with col:
                    st.markdown(f'<div class="info-box" dir="rtl"><div class="info-k">{k}</div><div class="info-v">{v}</div><div class="info-s">{s}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-head" dir="rtl"><div><div class="eyebrow">PREVIEW</div><div class="section-title">معاينة النتائج</div></div></div>', unsafe_allow_html=True)
            st.dataframe(csv_results[:50], use_container_width=True, hide_index=True)

            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=st.session_state.get("csv_columns", []),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(csv_results)

            st.download_button(
                "↓  تنزيل النتائج كاملة CSV",
                data=output.getvalue().encode("utf-8-sig"),
                file_name="arabic_sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_results",
            )


# =========================================================
# TAB 3 — ABOUT
# =========================================================
with tab_about:
    st.markdown("""
    <div class="section-head" dir="rtl">
      <div>
        <div class="eyebrow">SYSTEM INTELLIGENCE</div>
        <div class="section-title">عن Arabic Sentiment Compass</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" dir="rtl">
      <div class="card-label">PROJECT OVERVIEW</div>
      <div class="card-title">نظام تحليل مشاعر عربي مبني على AraBERT</div>
      <div class="card-sub" style="line-height:2;">
        تطبيق NLP عملي يقدّم inference حقيقي على نصوص عربية، ويعرض التصنيف،
        درجة الثقة، وتوزيع الاحتمالات مع دعم التحليل الفردي وتحليل ملفات CSV.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    about_items = [
        ("MODEL", "AraBERT", "aubmindlab/bert-base-arabertv02-twitter"),
        ("TASK", "3-Class", "Negative • Neutral • Positive"),
        ("MAX LENGTH", "128", "Tokenizer truncation limit"),
    ]
    for col, (k, v, s) in zip(cols, about_items):
        with col:
            st.markdown(f'<div class="info-box" dir="rtl"><div class="info-k">{k}</div><div class="info-v">{v}</div><div class="info-s">{s}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" dir="rtl">
      <div>
        <div class="eyebrow">INFERENCE PIPELINE</div>
        <div class="section-title">مسار التنبؤ</div>
      </div>
    </div>
    <div class="pipeline" dir="ltr">
      <div class="step"><div class="step-num">01</div><div class="step-title">Validation</div><div class="step-desc">تحقق من النص المدخل</div></div>
      <div class="arrow">→</div>
      <div class="step"><div class="step-num">02</div><div class="step-title">Preprocessing</div><div class="step-desc">التنظيف والتطبيع الرسمي</div></div>
      <div class="arrow">→</div>
      <div class="step"><div class="step-num">03</div><div class="step-title">Tokenizer</div><div class="step-desc">AraBERT tokenizer</div></div>
      <div class="arrow">→</div>
      <div class="step"><div class="step-num">04</div><div class="step-title">Model</div><div class="step-desc">Sequence classification</div></div>
      <div class="arrow">→</div>
      <div class="step"><div class="step-num">05</div><div class="step-title">Softmax</div><div class="step-desc">Probability distribution</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" dir="rtl">
      <div>
        <div class="eyebrow">SUPPORTED OUTPUT</div>
        <div class="section-title">ماذا يعرض النظام؟</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4, gap="small")
    outputs = [
        ("Sentiment", "التصنيف النهائي"),
        ("Confidence", "درجة الثقة"),
        ("Probabilities", "احتمالات الفئات الثلاث"),
        ("CSV Export", "نتائج قابلة للتحميل"),
    ]
    for col, (k, v) in zip(cols, outputs):
        with col:
            st.markdown(f'<div class="info-box" dir="rtl"><div class="info-k">{k}</div><div class="info-v">{v}</div></div>', unsafe_allow_html=True)


# ---------- footer ----------
st.markdown("""
<div class="footer" dir="rtl">
  <div><strong>Arabic Sentiment Compass</strong> · Arabic NLP Portfolio Project</div>
  <div>Real AraBERT Inference · Streamlit</div>
</div>
""", unsafe_allow_html=True)
