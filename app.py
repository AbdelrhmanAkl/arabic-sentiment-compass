
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
# CONFIGURATION — KEEP MODEL PIPELINE UNCHANGED
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

LABELS_ORDER = ["Negative", "Neutral", "Positive"]

SENTIMENT_ICONS = {
    "Negative": "🔴",
    "Neutral": "⚪",
    "Positive": "🟢",
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

AR_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")


def preprocess_arabic_tweet(text):
    text = str(text)
    text = URL_RE.sub(" رابط ", text)
    text = MENTION_RE.sub(" مستخدم ", text)
    text = AR_DIACRITICS.sub("", text)
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
        probabilities = torch.softmax(logits, dim=-1)

    scores = probabilities[0]

    predicted_id = int(scores.argmax())
    predicted_label = ID2LABEL[predicted_id]
    confidence = float(scores[predicted_id])

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
# CSS — THREE-TAB PROFESSIONAL UI
# =========================================================

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f7f5f0;
    --navy: #0b1526;
    --navy-2: #111d31;
    --gold: #c69a2b;
    --gold-soft: #e3c875;
    --border: #ded8ca;
    --text: #172033;
    --muted: #687080;
    --white: #ffffff;
}

html, body, [class*="css"] {
    font-family: "Cairo", sans-serif !important;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
}

/* HERO */
.compass-hero {
    direction: rtl;
    position: relative;
    overflow: hidden;
    min-height: 210px;
    padding: 1.65rem 2.25rem;
    border-radius: 18px;
    background:
        radial-gradient(circle at 75% 30%, rgba(198,154,43,.18), transparent 27%),
        linear-gradient(115deg, #091325 0%, #0d182a 55%, #26303a 100%);
    color: white;
    box-shadow: 0 12px 30px rgba(11,21,38,.12);
}

.compass-hero::after {
    content: "🧭";
    position: absolute;
    left: 28px;
    bottom: -25px;
    font-size: 8rem;
    opacity: .07;
}

.hero-badge {
    display: inline-block;
    border: 1px solid rgba(198,154,43,.75);
    color: #e8c85d;
    border-radius: 999px;
    padding: .22rem .75rem;
    font-size: .72rem;
    margin-bottom: .75rem;
}

.compass-hero h1 {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.1rem);
    line-height: 1.15;
    font-weight: 800;
}

.compass-hero p {
    max-width: 760px;
    margin: .7rem 0 0;
    color: #d6dce5;
    line-height: 1.9;
    font-size: .86rem;
}

/* TABS */
button[data-baseweb="tab"] {
    font-family: "Cairo", sans-serif !important;
    color: #495261 !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #111b2c !important;
}

div[data-baseweb="tab-highlight"] {
    background: var(--gold) !important;
}

/* PANELS */
.panel-title {
    direction: rtl;
    color: var(--navy);
    font-weight: 800;
    font-size: 1.1rem;
    margin: .55rem 0 .25rem;
}

.panel-subtitle {
    margin-top: 0 !important;
    direction: rtl;
    color: var(--muted);
    font-size: .78rem;
    margin-bottom: .8rem;
}

.surface {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
}

.gold-title {
    color: var(--gold);
    font-weight: 800;
}

textarea {
    direction: rtl !important;
    text-align: right !important;
    font-family: "Cairo", sans-serif !important;
}

/* Remove any accidental empty single-line Streamlit text input.
   The actual analysis field is the textarea below it. */
div[data-testid="stTextInput"] {
    display: none !important;
}



/* Single-text analysis workspace */
div[data-testid="stTextArea"] {
    background: #ffffff !important;
    border: 1px solid #ddd7ca !important;
    border-radius: 14px !important;
    padding: .85rem !important;
    box-shadow: 0 8px 24px rgba(11, 21, 38, .045) !important;
}

div[data-testid="stTextArea"] textarea {
    background: #fffdf9 !important;
    border: 1px solid #d9d1c0 !important;
    border-radius: 10px !important;
    min-height: 150px !important;
    box-shadow: none !important;
}

div[data-testid="stTextArea"] textarea {
    background: #fffdf9 !important;
    border: 1px solid #d9d1c0 !important;
    border-radius: 11px !important;
    min-height: 150px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(198,154,43,.10) !important;
}

div.stButton > button {
    font-family: "Cairo", sans-serif !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 2.45rem !important;
}

.primary-btn button {
    background: var(--navy) !important;
    color: white !important;
    border: 1px solid var(--navy) !important;
}

.secondary-btn button {
    background: white !important;
    color: var(--navy) !important;
    border: 1px solid #d6d0c2 !important;
}

/* RESULT */
.result-card {
    direction: rtl;
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
}

.result-question {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto .65rem;
    border: 1px solid #ded7c8;
    border-radius: 11px;
    color: var(--gold);
    font-size: 1.25rem;
}

.result-label {
    color: var(--navy);
    font-size: 1.35rem;
    font-weight: 800;
}

.result-en {
    color: var(--muted);
    font-size: .78rem;
}

.confidence {
    display: inline-block;
    margin-top: .65rem;
    padding: .38rem .8rem;
    border-radius: 8px;
    background: #fbf7e9;
    border: 1px solid #eadfb9;
    color: #765a08;
    font-size: .82rem;
    font-weight: 700;
}

.prob-card {
    direction: rtl;
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .9rem;
    text-align: center;
}

.prob-icon {
    font-size: 1.15rem;
}

.prob-title {
    color: var(--navy);
    font-weight: 800;
    font-size: .9rem;
}

.prob-en {
    color: var(--muted);
    font-size: .7rem;
}

.prob-value {
    color: var(--gold);
    font-size: 1.15rem;
    font-weight: 800;
    margin-top: .3rem;
}

/* CSV */
.csv-note {
    direction: rtl;
    background: #fffdf8;
    border: 1px solid #e6dcc0;
    border-right: 4px solid var(--gold);
    border-radius: 9px;
    padding: .7rem .9rem;
    color: #5d5748;
    font-size: .76rem;
    line-height: 1.8;
}

/* ABOUT */
.about-card {
    direction: rtl;
    background: white;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 1rem;
    min-height: 105px;
}

.about-card h4 {
    color: var(--navy);
    margin: 0 0 .35rem;
    font-size: .9rem;
}

.about-card p {
    color: #646b77;
    margin: 0;
    line-height: 1.8;
    font-size: .72rem;
}

.footer {
    direction: rtl;
    text-align: center;
    color: #7b7d82;
    font-size: .72rem;
    padding-top: 1.4rem;
    margin-top: 2rem;
    border-top: 1px solid #e6e0d5;
}

@media (max-width: 768px) {
    .compass-hero {
        padding: 1.25rem;
        min-height: 190px;
    }

    .compass-hero h1 {
        font-size: 2rem;
    }
}

/* =========================================================
   PREMIUM V2 VISUAL SYSTEM
   ========================================================= */

:root {
    --bg: #f4f2ed;
    --navy: #091425;
    --navy-2: #111d2e;
    --gold: #c49a32;
    --gold-light: #ead58d;
    --ink: #172033;
    --muted: #697180;
    --border: #ddd7ca;
    --card: #ffffff;
    --shadow: 0 14px 38px rgba(11, 21, 38, .08);
}

.stApp {
    background:
        radial-gradient(circle at 8% 6%, rgba(196,154,50,.045), transparent 22%),
        linear-gradient(180deg, #f7f5f0 0%, #f1efe9 100%);
}

.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.compass-hero {
    min-height: 235px;
    padding: 2rem 2.5rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at 16% 62%, rgba(196,154,50,.15), transparent 20%),
        radial-gradient(circle at 86% 15%, rgba(255,255,255,.08), transparent 28%),
        linear-gradient(120deg, #071224 0%, #0d192a 52%, #28313a 100%);
    box-shadow:
        0 20px 45px rgba(8, 18, 35, .16),
        inset 0 1px 0 rgba(255,255,255,.06);
}

.compass-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent 0 72%, rgba(196,154,50,.06) 72% 72.2%, transparent 72.2%),
        linear-gradient(0deg, transparent 0 78%, rgba(255,255,255,.035) 78% 78.2%, transparent 78.2%);
    pointer-events: none;
}

.compass-hero::after {
    content: "🧭";
    left: 38px;
    bottom: -22px;
    font-size: 8.5rem;
    filter: grayscale(1);
    opacity: .11;
}

.hero-badge {
    position: relative;
    z-index: 2;
    border-color: rgba(232,200,93,.72);
    background: rgba(196,154,50,.08);
    padding: .28rem .82rem;
    font-size: .7rem;
    letter-spacing: .02em;
}

.compass-hero h1 {
    position: relative;
    z-index: 2;
    font-size: clamp(2.2rem, 5vw, 3.65rem);
    letter-spacing: -.035em;
    text-shadow: 0 4px 20px rgba(0,0,0,.16);
}

.compass-hero p {
    position: relative;
    z-index: 2;
    max-width: 850px;
    font-size: .9rem;
    line-height: 2;
}

.hero-meta {
    position: relative;
    z-index: 2;
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    margin-top: 1rem;
    direction: rtl;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .28rem .65rem;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 999px;
    background: rgba(255,255,255,.055);
    color: #dce3ed;
    font-size: .67rem;
}

.hero-chip strong {
    color: #e5c55d;
}

div[data-baseweb="tab-list"] {
    gap: .3rem !important;
    border-bottom: 1px solid #ddd7ca !important;
    padding: .35rem .15rem 0 !important;
}

button[data-baseweb="tab"] {
    min-height: 2.65rem !important;
    padding: .45rem 1rem !important;
    border-radius: 9px 9px 0 0 !important;
    transition: .18s ease !important;
}

button[data-baseweb="tab"]:hover {
    background: rgba(196,154,50,.06) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #fff !important;
    color: var(--navy) !important;
    box-shadow: 0 -1px 0 #ddd7ca, 0 1px 0 #fff !important;
}

div[data-baseweb="tab-highlight"] {
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
    background: linear-gradient(90deg, #b88620, #e4c968) !important;
}

.panel-title {
    font-size: 1.18rem;
    letter-spacing: -.01em;
}

.panel-subtitle {
    margin-bottom: 1rem;
}

.surface {
    border: 1px solid rgba(221,215,202,.9);
    box-shadow: var(--shadow);
    padding: 1.2rem;
}

div[data-testid="stTextArea"] textarea {
    min-height: 165px !important;
    background: #fffefa !important;
    border-color: #d6cdbb !important;
    box-shadow: inset 0 1px 2px rgba(20,30,45,.03);
}

div.stButton > button {
    min-height: 2.65rem !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-1px);
}

.primary-btn button {
    background: linear-gradient(135deg, #0a1628, #17263a) !important;
    box-shadow: 0 7px 16px rgba(9,20,37,.14);
}

.secondary-btn button {
    background: #fff !important;
}

.result-card {
    min-height: 190px;
    border-radius: 18px;
    border-color: #ddd5c4;
    box-shadow: var(--shadow);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 4px;
    background: linear-gradient(90deg, #b98d28, #e6cf78, #b98d28);
}

.result-question {
    width: 50px;
    height: 50px;
    border-radius: 14px;
    background: #fbf8f0;
    box-shadow: 0 5px 15px rgba(20,30,45,.06);
}

.result-label {
    font-size: 1.65rem;
    margin-top: .15rem;
}

.confidence {
    padding: .48rem 1rem;
    border-radius: 999px;
    font-size: .8rem;
    background: linear-gradient(180deg, #fffaf0, #f8f0d8);
}

.prob-card {
    min-height: 155px;
    padding: 1rem;
    border-radius: 16px;
    box-shadow: 0 9px 25px rgba(11,21,38,.055);
    transition: transform .18s ease, box-shadow .18s ease;
}

.prob-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 30px rgba(11,21,38,.09);
}

.prob-value {
    font-size: 1.35rem;
}

div[data-testid="stProgress"] > div {
    background: #e8e6df !important;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #b88620, #e2c75d) !important;
}

.csv-note {
    box-shadow: 0 5px 16px rgba(11,21,38,.035);
}

div[data-testid="stFileUploader"] {
    border-radius: 14px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #ddd7ca !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 9px 24px rgba(11,21,38,.05);
}

div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataFrame"] [role="columnheader"] {
    font-family: "Cairo", sans-serif !important;
}

.about-card {
    min-height: 125px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(11,21,38,.045);
    transition: transform .16s ease;
}

.about-card:hover {
    transform: translateY(-2px);
}

.footer {
    margin-top: 2.5rem;
    padding-top: 1.5rem;
}

.footer strong {
    color: var(--navy);
}

@media (max-width: 768px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }

    .compass-hero {
        min-height: 220px;
        padding: 1.4rem;
        border-radius: 18px;
    }

    .compass-hero h1 {
        font-size: 2rem;
    }

    .hero-meta {
        gap: .3rem;
    }
}

</style>
""")


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="compass-hero">
    <div class="hero-badge">UCAS · مشروع تخرج</div>
    <h1>بوصلة المشاعر العربية</h1>
    <p>
        منصة تفاعلية لتحليل المشاعر العربية باستخدام نموذج AraBERT الحقيقي،
        مع دعم تحليل النصوص المفردة ومجموعات البيانات بصيغة CSV.
    </p>
    <div class="hero-meta">
        <span class="hero-chip">🤖 <strong>Model</strong>&nbsp; AraBERT</span>
        <span class="hero-chip">🎯 <strong>Classes</strong>&nbsp; 3</span>
        <span class="hero-chip">📏 <strong>Max Length</strong>&nbsp; 128</span>
        <span class="hero-chip">⚡ <strong>Inference</strong>&nbsp; Real Model</span>
    </div>
</div>
""")


# =========================================================
# THREE MAIN TABS
# =========================================================

tab_text, tab_csv, tab_about = st.tabs(
    ["📝 تحليل نص مفرد", "📊 تحليل ملف CSV", "ℹ️ نبذة عن النظام"]
)


# =========================================================
# TAB 1 — SINGLE TEXT
# =========================================================

with tab_text:
    st.markdown(
        '<div class="panel-title">تحليل نص عربي</div>'
        '<div class="panel-subtitle">اكتب النص الذي تريد تحليله ثم ابدأ التحليل.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown(
            '<div class="panel-title" style="font-size:.95rem;">اكتب النص المراد تحليله</div>',
            unsafe_allow_html=True,
        )

        user_text = st.text_area(
            "النص العربي",
            placeholder="مثال: الخدمة ممتازة والتجربة رائعة جدًا",
            height=165,
            label_visibility="collapsed",
            key="single_text",
        )

        c1, c2 = st.columns([1, 1], gap="small")

        with c1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            analyze_button = st.button(
                "تحليل المشاعر",
                use_container_width=True,
                key="analyze_single",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            clear_button = st.button(
                "تفريغ الحقول",
                use_container_width=True,
                key="clear_single",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if clear_button:
            st.session_state.single_result = None
            st.rerun()

    with right:
        result = st.session_state.get("single_result")

        if result is None:
            st.html("""
            <div class="result-card" style="min-height:190px; display:flex; flex-direction:column; justify-content:center;">
                <div class="result-question">؟</div>
                <div class="result-label" style="font-size:1.05rem;">بانتظار النص للتحليل</div>
                <div class="result-en">أدخل النص ثم اضغط تحليل المشاعر لعرض النتيجة.</div>
            </div>
            """)
        else:
            label = result["predicted_label"]
            confidence = result["confidence"]

            st.html(f"""
            <div class="result-card">
                <div class="result-question">{SENTIMENT_ICONS[label]}</div>
                <div class="result-label">{ARABIC_LABELS[label]}</div>
                <div class="result-en">Predicted sentiment · {label}</div>
                <div class="confidence">درجة الثقة · {confidence * 100:.2f}%</div>
            </div>
            """)

    if analyze_button:
        if not user_text.strip():
            st.warning("من فضلك اكتب نصًا أولًا.")
        else:
            try:
                with st.spinner("جاري تحليل النص باستخدام AraBERT..."):
                    tokenizer, model = load_model()
                    (
                        processed_text,
                        predicted_label,
                        confidence,
                        probabilities_dict,
                    ) = predict_sentiment(user_text, tokenizer, model)

                st.session_state.single_result = {
                    "processed_text": processed_text,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "probabilities_dict": probabilities_dict,
                }
                st.rerun()

            except Exception as exc:
                st.error("حدث خطأ أثناء تحليل النص.")
                st.exception(exc)

    result = st.session_state.get("single_result")

    if result is not None:
        st.markdown(
            '<div class="panel-title" style="margin-top:1.2rem;">احتمالات جميع المشاعر</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3, gap="medium")

        for col, label in zip(cols, LABELS_ORDER):
            probability = result["probabilities_dict"][label]

            with col:
                st.html(f"""
                <div class="prob-card">
                    <div class="prob-icon">{SENTIMENT_ICONS[label]}</div>
                    <div class="prob-title">{ARABIC_LABELS[label]}</div>
                    <div class="prob-en">{label}</div>
                    <div class="prob-value">{probability * 100:.2f}%</div>
                </div>
                """)
                st.progress(probability)


# =========================================================
# TAB 2 — CSV
# =========================================================

with tab_csv:
    st.markdown(
        '<div class="panel-title">تحليل مجموعة نصوص دفعة واحدة</div>'
        '<div class="panel-subtitle">ارفع ملف CSV يحتوي على عمود للنصوص ثم حلله باستخدام النموذج الحقيقي.</div>',
        unsafe_allow_html=True,
    )

    st.html("""
    <div class="csv-note">
        يجب أن يحتوي الملف على عمود نصي واحد على الأقل.
        بعد رفع الملف اختر العمود المراد تحليله، ثم ابدأ تحليل الملف.
    </div>
    """)

    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "رفع ملف CSV",
            type=["csv"],
            key="csv_uploader",
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

                    st.caption(f"عدد الصفوف: {len(rows)}")

                    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                    analyze_csv_button = st.button(
                        "بدء تحليل الملف",
                        use_container_width=True,
                        key="analyze_csv",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if analyze_csv_button:
                        try:
                            with st.spinner("جاري تحليل الملف باستخدام AraBERT..."):
                                tokenizer, model = load_model()

                                results = []

                                progress = st.progress(0.0)
                                total = len(rows)

                                for index, row in enumerate(rows, start=1):
                                    text = str(row.get(text_column, "") or "")

                                    if text.strip():
                                        (
                                            processed_text,
                                            predicted_label,
                                            confidence,
                                            probabilities_dict,
                                        ) = predict_sentiment(
                                            text,
                                            tokenizer,
                                            model,
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
                                    new_row["negative_probability"] = round(
                                        probabilities_dict["Negative"], 6
                                    )
                                    new_row["neutral_probability"] = round(
                                        probabilities_dict["Neutral"], 6
                                    )
                                    new_row["positive_probability"] = round(
                                        probabilities_dict["Positive"], 6
                                    )

                                    results.append(new_row)
                                    progress.progress(index / total)

                                st.session_state.csv_results = results
                                st.session_state.csv_columns = (
                                    list(results[0].keys()) if results else []
                                )

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
            st.html("""
            <div class="result-card" style="min-height:210px; display:flex; flex-direction:column; justify-content:center;">
                <div class="result-question">?</div>
                <div class="result-label" style="font-size:1.05rem;">نتائج تحليل الملف</div>
                <div class="result-en">ستظهر النتائج هنا بعد تحليل ملف CSV.</div>
            </div>
            """)
        else:
            columns = st.session_state.get("csv_columns", [])

            st.markdown(
                '<div class="panel-title" style="font-size:.95rem;">نتائج التحليل</div>',
                unsafe_allow_html=True,
            )

            preview_rows = csv_results[:50]

            # Native Streamlit table for reliable RTL-compatible data rendering.
            st.dataframe(
                preview_rows,
                use_container_width=True,
                hide_index=True,
            )

            counts = Counter(
                row.get("sentiment", "Neutral")
                for row in csv_results
            )

            summary_cols = st.columns(3, gap="small")
            for summary_col, label in zip(summary_cols, LABELS_ORDER):
                with summary_col:
                    st.metric(
                        ARABIC_LABELS[label],
                        counts.get(label, 0),
                    )

            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(csv_results)

            st.download_button(
                "تنزيل النتائج كاملة CSV",
                data=output.getvalue().encode("utf-8-sig"),
                file_name="arabic_sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_results",
            )


# =========================================================
# TAB 3 — ABOUT SYSTEM
# =========================================================

with tab_about:
    st.markdown(
        '<div class="panel-title">نبذة عن النظام</div>'
        '<div class="panel-subtitle">معلومات مختصرة عن النموذج والمشروع وطريقة التحليل.</div>',
        unsafe_allow_html=True,
    )

    row1 = st.columns(3, gap="medium")

    with row1[0]:
        st.html("""
        <div class="about-card">
            <h4>النموذج</h4>
            <p>
                AraBERT02-Twitter / AraBERT مخصص للنصوص العربية القصيرة
                ومهام تحليل المشاعر.
            </p>
        </div>
        """)

    with row1[1]:
        st.html("""
        <div class="about-card">
            <h4>عدد الفئات</h4>
            <p>
                ثلاث فئات رسمية: سلبي، محايد، إيجابي.
                ويُعرض ترتيب الاحتمالات دائمًا بنفس الترتيب.
            </p>
        </div>
        """)

    with row1[2]:
        st.html("""
        <div class="about-card">
            <h4>المعالجة</h4>
            <p>
                يتم تطبيق preprocessing الرسمي على النص قبل إرساله
                إلى Tokenizer والموديل.
            </p>
        </div>
        """)

    row2 = st.columns(3, gap="medium")

    with row2[0]:
        st.html("""
        <div class="about-card">
            <h4>MAX_LENGTH</h4>
            <p>
                الحد الأقصى لطول الإدخال المستخدم في الـTokenizer هو 128 token.
            </p>
        </div>
        """)

    with row2[1]:
        st.html("""
        <div class="about-card">
            <h4>Confidence</h4>
            <p>
                قيمة الـConfidence هي أعلى Softmax Score للناتج المتوقع،
                وليست calibrated probability.
            </p>
        </div>
        """)

    with row2[2]:
        st.html("""
        <div class="about-card">
            <h4>Model Hub</h4>
            <p>
                يتم تحميل الـTokenizer والموديل النهائي من مستودع
                Hugging Face الخاص بالمشروع.
            </p>
        </div>
        """)

    st.html("""
    <div class="csv-note" style="margin-top:1rem;">
        <span class="gold-title">نطاق المشروع:</span>
        Arabic Sentiment Compass هو تطبيق تفاعلي لتحليل المشاعر العربية
        باستخدام نموذج AraBERT الحقيقي، مع واجهة لتحليل نص مفرد وتحليل ملفات CSV.
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    <strong>Arabic Sentiment Compass 🧭</strong><br>
    Arabic Sentiment Analysis · Powered by AraBERT
    <br>
    <span style="font-size:.65rem;">Real model inference · 3-class Arabic sentiment classification</span>
</div>
""")
