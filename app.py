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
# PREMIUM UI
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');


/* =========================================================
   GLOBAL
   ========================================================= */

:root {
    --bg: #f4f6fa;
    --surface: #ffffff;
    --surface-2: #f8f9fc;

    --navy: #0b1730;
    --navy-soft: #1b2942;

    --purple: #6557e8;
    --purple-light: #8378f5;

    --cyan: #22b8c7;

    --green: #169b6b;
    --red: #d95757;
    --gold: #b78a28;

    --text: #182235;
    --muted: #718096;

    --border: #e5e9f0;

    --shadow-sm:
        0 6px 20px rgba(15, 23, 42, 0.045);

    --shadow:
        0 16px 40px rgba(15, 23, 42, 0.065);
}


html,
body,
[class*="css"] {

    font-family:
        "IBM Plex Sans Arabic",
        "Cairo",
        sans-serif !important;
}


.stApp {

    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(101, 87, 232, 0.055),
            transparent 26%
        ),

        radial-gradient(
            circle at 100% 10%,
            rgba(34, 184, 199, 0.045),
            transparent 25%
        ),

        linear-gradient(
            180deg,
            #f8fafc 0%,
            #f3f5f8 100%
        );

    color: var(--text);
}


header,
footer,
#MainMenu {

    visibility: hidden;
}


.block-container {

    max-width: 1280px;

    padding-top: 1rem;
    padding-bottom: 3rem;

}


/* =========================================================
   TOP BAR
   ========================================================= */

.topbar {

    direction: rtl;

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 1rem;

}


.brand {

    display: flex;

    align-items: center;

    gap: 0.7rem;

}


.brand-icon {

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 13px;

    color: white;

    font-size: 1.2rem;

    background:
        linear-gradient(
            135deg,
            var(--purple),
            var(--purple-light)
        );

    box-shadow:
        0 8px 20px
        rgba(101, 87, 232, 0.22);

}


.brand-title {

    color: var(--navy);

    font-size: 0.94rem;

    font-weight: 800;

    line-height: 1.2;

}


.brand-subtitle {

    color: var(--muted);

    font-size: 0.59rem;

    margin-top: 0.12rem;

}


.status {

    display: flex;

    align-items: center;

    gap: 0.45rem;

    direction: ltr;

    padding:
        0.4rem 0.72rem;

    border:
        1px solid var(--border);

    border-radius: 999px;

    background: rgba(255,255,255,0.85);

    color: #596579;

    font-size: 0.62rem;

    font-weight: 700;

}


.status-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #19aa78;

    box-shadow:
        0 0 0 4px
        rgba(25,170,120,0.10);

}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    direction: rtl;

    padding:
        2.6rem 3rem;

    border-radius: 26px;

    background:

        radial-gradient(
            circle at 85% 15%,
            rgba(34,184,199,0.11),
            transparent 26%
        ),

        radial-gradient(
            circle at 10% 85%,
            rgba(101,87,232,0.25),
            transparent 28%
        ),

        linear-gradient(
            120deg,
            #091426 0%,
            #0d1d34 55%,
            #172943 100%
        );

    box-shadow:
        0 22px 55px
        rgba(8,20,38,0.14);

}


.hero::after {

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    left: -150px;
    bottom: -190px;

    border:
        1px solid
        rgba(255,255,255,0.055);

    border-radius: 50%;

    box-shadow:
        0 0 0 35px rgba(255,255,255,0.018),
        0 0 0 70px rgba(255,255,255,0.012);

}


.hero-content {

    position: relative;

    z-index: 2;

    max-width: 850px;

}


.hero-badge {

    display: inline-flex;

    align-items: center;

    gap: 0.45rem;

    padding:
        0.35rem 0.7rem;

    margin-bottom: 1rem;

    border:
        1px solid
        rgba(255,255,255,0.12);

    border-radius: 999px;

    background:
        rgba(255,255,255,0.045);

    color: #cbd5e3;

    font-size: 0.61rem;

    font-weight: 600;

}


.hero-badge strong {

    color: #9a91ff;

}


.hero h1 {

    margin: 0;

    color: white;

    font-size:
        clamp(2.2rem, 5vw, 3.7rem);

    line-height: 1.25;

    font-weight: 800;

    letter-spacing: -0.035em;

}


.hero-description {

    max-width: 760px;

    margin-top: 0.9rem;

    color: #b7c3d4;

    font-size: 0.82rem;

    line-height: 2;

}


.hero-meta {

    display: flex;

    flex-wrap: wrap;

    gap: 0.5rem;

    margin-top: 1.4rem;

}


.meta-item {

    display: flex;

    align-items: center;

    gap: 0.4rem;

    direction: ltr;

    padding:
        0.43rem 0.7rem;

    border:
        1px solid
        rgba(255,255,255,0.095);

    border-radius: 9px;

    background:
        rgba(255,255,255,0.035);

    color: #c7d1df;

    font-size: 0.6rem;

}


.meta-item strong {

    color: white;

}


/* =========================================================
   TABS
   ========================================================= */

div[data-baseweb="tab-list"] {

    display: flex;

    gap: 0.3rem !important;

    margin-top: 1.4rem;

    padding: 0.3rem !important;

    border:
        1px solid var(--border) !important;

    border-radius: 13px !important;

    background:
        rgba(255,255,255,0.78) !important;

}


button[data-baseweb="tab"] {

    min-height: 2.6rem !important;

    padding:
        0.45rem 1.1rem !important;

    border-radius: 9px !important;

    color: #667085 !important;

    font-family:
        "IBM Plex Sans Arabic",
        "Cairo",
        sans-serif !important;

    font-size: 0.72rem !important;

    font-weight: 700 !important;

}


button[data-baseweb="tab"]:hover {

    background:
        rgba(101,87,232,0.055) !important;

}


button[data-baseweb="tab"][aria-selected="true"] {

    color: white !important;

    background:
        linear-gradient(
            135deg,
            var(--purple),
            var(--purple-light)
        ) !important;

    box-shadow:
        0 6px 15px
        rgba(101,87,232,0.18) !important;

}


div[data-baseweb="tab-highlight"] {

    display: none !important;

}


/* =========================================================
   SECTION
   ========================================================= */

.section {

    direction: rtl;

    margin:
        1.8rem 0 1rem;

}


.section-eyebrow {

    color: var(--purple);

    font-size: 0.59rem;

    font-weight: 800;

    letter-spacing: 0.07em;

}


.section-title {

    margin-top: 0.15rem;

    color: var(--navy);

    font-size: 1.22rem;

    font-weight: 800;

}


.section-description {

    margin-top: 0.22rem;

    color: var(--muted);

    font-size: 0.68rem;

    line-height: 1.8;

}


/* =========================================================
   CARDS
   ========================================================= */

.card {

    direction: rtl;

    padding: 1.2rem;

    border:
        1px solid var(--border);

    border-radius: 18px;

    background: var(--surface);

    box-shadow: var(--shadow-sm);

}


.card-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 0.8rem;

}


.card-title {

    color: var(--navy);

    font-size: 0.85rem;

    font-weight: 800;

}


.card-caption {

    color: #8b95a5;

    font-size: 0.58rem;

}


/* =========================================================
   TEXT INPUT
   ========================================================= */

div[data-testid="stTextArea"] {

    margin-top: 0.2rem;

}


div[data-testid="stTextArea"] textarea {

    direction: rtl !important;

    text-align: right !important;

    font-family:
        "IBM Plex Sans Arabic",
        "Cairo",
        sans-serif !important;

    min-height: 175px !important;

    padding: 0.95rem !important;

    border:
        1px solid #dfe4ec !important;

    border-radius: 13px !important;

    background: #fbfcfe !important;

    color: #172033 !important;

    font-size: 0.8rem !important;

    line-height: 2 !important;

}


div[data-testid="stTextArea"] textarea:focus {

    border-color:
        var(--purple) !important;

    box-shadow:
        0 0 0 3px
        rgba(101,87,232,0.09) !important;

}


/* =========================================================
   BUTTONS
   ========================================================= */

div.stButton > button {

    width: 100%;

    min-height: 2.75rem;

    border-radius: 10px !important;

    font-family:
        "IBM Plex Sans Arabic",
        "Cairo",
        sans-serif !important;

    font-size: 0.7rem !important;

    font-weight: 700 !important;

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease !important;

}


div.stButton > button:hover {

    transform:
        translateY(-1px);

}


.primary button {

    color: white !important;

    border:
        1px solid var(--purple) !important;

    background:
        linear-gradient(
            135deg,
            var(--purple),
            var(--purple-light)
        ) !important;

    box-shadow:
        0 8px 18px
        rgba(101,87,232,0.16) !important;

}


.secondary button {

    color: #4d596c !important;

    border:
        1px solid #dfe4ec !important;

    background:
        white !important;

}


/* =========================================================
   EMPTY STATE
   ========================================================= */

.empty {

    min-height: 275px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    direction: rtl;

    padding: 1.5rem;

    border:
        1px dashed #d7dee8;

    border-radius: 18px;

    background:
        linear-gradient(
            180deg,
            #ffffff,
            #fafbfd
        );

}


.empty-icon {

    width: 54px;
    height: 54px;

    display: flex;

    align-items: center;
    justify-content: center;

    margin-bottom: 0.8rem;

    border-radius: 15px;

    color: var(--purple);

    background:
        rgba(101,87,232,0.065);

    border:
        1px solid
        rgba(101,87,232,0.11);

    font-size: 1.25rem;

}


.empty-title {

    color: var(--navy);

    font-size: 0.92rem;

    font-weight: 800;

}


.empty-text {

    max-width: 280px;

    margin-top: 0.35rem;

    color: var(--muted);

    font-size: 0.65rem;

    line-height: 1.9;

}


/* =========================================================
   RESULT
   ========================================================= */

.result {

    position: relative;

    overflow: hidden;

    min-height: 275px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    direction: rtl;

    padding: 1.5rem;

    border:
        1px solid var(--border);

    border-radius: 18px;

    background:
        linear-gradient(
            180deg,
            #ffffff,
            #fafbfe
        );

    box-shadow: var(--shadow-sm);

}


.result::before {

    content: "";

    position: absolute;

    top: 0;
    left: 0;
    right: 0;

    height: 3px;

    background:
        linear-gradient(
            90deg,
            var(--purple),
            var(--cyan)
        );

}


.result-orb {

    width: 70px;
    height: 70px;

    display: flex;

    align-items: center;
    justify-content: center;

    margin-bottom: 0.75rem;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(101,87,232,0.13),
            rgba(101,87,232,0.025)
        );

    border:
        1px solid
        rgba(101,87,232,0.13);

}


.result-symbol {

    font-size: 2rem;

    font-weight: 700;

    color: var(--purple);

}


.result-label {

    color: var(--navy);

    font-size: 1.5rem;

    font-weight: 800;

}


.result-sub {

    margin-top: 0.12rem;

    color: var(--muted);

    font-size: 0.6rem;

}


.confidence {

    margin-top: 0.75rem;

    padding:
        0.4rem 0.75rem;

    border-radius: 999px;

    color: #584aa8;

    background:
        rgba(101,87,232,0.065);

    border:
        1px solid
        rgba(101,87,232,0.10);

    font-size: 0.66rem;

    font-weight: 700;

}


/* =========================================================
   PROBABILITIES
   ========================================================= */

.prob-title {

    direction: rtl;

    margin:
        1.25rem 0 0.65rem;

    color: var(--navy);

    font-size: 0.82rem;

    font-weight: 800;

}


 .prob {

        direction: rtl !important;
        text-align: right !important;

        position: relative;
        overflow: hidden;

        padding: 1.15rem 1.15rem 1rem;

        min-height: 138px;

        border: 1px solid #e4e8f0;
        border-radius: 18px;

        background:
            linear-gradient(145deg, #ffffff 0%, #fbfcff 100%);

        box-shadow:
            0 8px 24px rgba(15, 23, 42, 0.045);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease;

    }


    .prob:hover {

        transform: translateY(-2px);

        border-color: #d9deea;

        box-shadow:
            0 14px 30px rgba(15, 23, 42, 0.08);

    }


    .prob::before {

        content: "";

        position: absolute;

        top: 0;
        right: 0;

        width: 100%;
        height: 3px;

        background:
            linear-gradient(
                90deg,
                rgba(101,87,232,0.20),
                rgba(34,184,199,0.65)
            );

    }


.prob-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

}


.prob-name {

    color: var(--navy);

    font-size: 0.76rem;

    font-weight: 800;

}


.prob-en {

    margin-top: 0.05rem;

    color: #929aaa;

    font-size: 0.54rem;

}


.prob-icon {

    width: 31px;
    height: 31px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 9px;

    background: #f4f6fa;

    color: #667085;

    font-weight: 700;

}


.prob-value {

    margin-top: 0.65rem;

    color: var(--navy);

    font-size: 1.18rem;

    font-weight: 800;

}


.prob-track {

    height: 5px;

    overflow: hidden;

    margin-top: 0.5rem;

    border-radius: 999px;

    background: #edf0f4;

}


.prob-fill {

    height: 100%;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            var(--purple),
            var(--cyan)
        );

}


/* =========================================================
   PROCESSED TEXT
   ========================================================= */

.processed {

    direction: rtl;

    margin-top: 0.9rem;

    padding:
        0.85rem 0.95rem;

    border:
        1px solid var(--border);

    border-radius: 13px;

    background: #fafbfc;

}


.processed-label {

    color: var(--navy);

    font-size: 0.63rem;

    font-weight: 800;

}


.processed-value {

    margin-top: 0.3rem;

    color: #687386;

    font-size: 0.64rem;

    line-height: 1.9;

}


/* =========================================================
   CSV
   ========================================================= */

.info {

    direction: rtl;

    margin-bottom: 1rem;

    padding:
        0.8rem 0.95rem;

    border:
        1px solid #e2e7ee;

    border-right:
        3px solid var(--purple);

    border-radius: 12px;

    background: white;

    color: #687386;

    font-size: 0.64rem;

    line-height: 1.9;

}


div[data-testid="stFileUploader"] {

    padding: 0.35rem;

    border:
        1px dashed #ccd5e1;

    border-radius: 13px;

    background: #fafbfc;

}


/* =========================================================
   STAT CARDS
   ========================================================= */

.stat {

    direction: rtl;

    padding: 0.9rem;

    border:
        1px solid var(--border);

    border-radius: 14px;

    background: white;

    box-shadow: var(--shadow-sm);

}


.stat-label {

    color: var(--muted);

    font-size: 0.58rem;

}


.stat-value {

    margin-top: 0.2rem;

    color: var(--navy);

    font-size: 1.18rem;

    font-weight: 800;

}


.stat-sub {

    margin-top: 0.08rem;

    color: #98a1af;

    font-size: 0.52rem;

}


/* =========================================================
   DATAFRAME
   ========================================================= */

div[data-testid="stDataFrame"] {

    overflow: hidden !important;

    border:
        1px solid var(--border) !important;

    border-radius: 13px !important;

    box-shadow: var(--shadow-sm);

}


div[data-testid="stDataFrame"] * {

    font-family:
        "IBM Plex Sans Arabic",
        "Cairo",
        sans-serif !important;

}


/* =========================================================
   MODEL INTELLIGENCE
   ========================================================= */

.intel {

    direction: rtl;

    min-height: 130px;

    padding: 1rem;

    border:
        1px solid var(--border);

    border-radius: 15px;

    background:
        linear-gradient(
            180deg,
            #ffffff,
            #fafbfe
        );

    box-shadow: var(--shadow-sm);

}


.intel-label {

    color: var(--muted);

    font-size: 0.57rem;

}


.intel-value {

    margin-top: 0.35rem;

    color: var(--navy);

    font-size: 0.92rem;

    font-weight: 800;

}


.intel-accent {

    color: var(--purple);

}


/* =========================================================
   PIPELINE
   ========================================================= */

.pipeline {

    position: relative;

    direction: rtl;

    min-height: 118px;

    padding: 0.9rem;

    border:
        1px solid var(--border);

    border-radius: 15px;

    background: white;

    box-shadow: var(--shadow-sm);

}


.pipeline-number {

    color: var(--purple);

    font-size: 0.56rem;

    font-weight: 800;

}


.pipeline-title {

    margin-top: 0.4rem;

    color: var(--navy);

    font-size: 0.72rem;

    font-weight: 800;

}


.pipeline-sub {

    margin-top: 0.25rem;

    color: var(--muted);

    font-size: 0.55rem;

    line-height: 1.6;

}


/* =========================================================
   ABOUT
   ========================================================= */

/* =========================================================
   ABOUT — POLISHED RTL CARDS
   ========================================================= */

.about {

    direction: rtl !important;
    text-align: right !important;

    min-height: 178px;
    padding: 1.25rem;

    border: 1px solid var(--border);
    border-radius: 18px;

    background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);

    box-shadow: var(--shadow-sm);

    display: flex;
    flex-direction: column;
    align-items: flex-start;

    overflow: hidden;

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        border-color 0.18s ease;

}

.about:hover {

    transform: translateY(-2px);
    border-color: #d8dcef;

    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);

}

.about-icon {

    width: 40px;
    height: 40px;

    flex: 0 0 auto;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-bottom: 0.85rem;
    border-radius: 11px;

    color: var(--purple);

    background: linear-gradient(
        135deg,
        rgba(101,87,232,0.10),
        rgba(34,184,199,0.07)
    );

    border: 1px solid rgba(101,87,232,0.10);

    font-size: 0.95rem;
    font-weight: 800;

}

.about-title {

    width: 100%;

    color: var(--navy);

    font-size: 0.82rem;
    line-height: 1.45;
    font-weight: 800;

    unicode-bidi: plaintext;

}

.about-text {

    width: 100%;

    margin-top: 0.5rem;

    color: #687386;

    font-size: 0.68rem;
    line-height: 2.05;
    font-weight: 500;

    direction: rtl !important;
    text-align: right !important;

    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: normal;

    unicode-bidi: plaintext;

}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    direction: rtl;

    margin-top: 2.8rem;

    padding:
        1.4rem 0 0.5rem;

    border-top:
        1px solid var(--border);

    text-align: center;

}


.footer-title {

    color: var(--navy);

    font-size: 0.72rem;

    font-weight: 800;

}


.footer-text {

    margin-top: 0.25rem;

    color: #8c96a5;

    font-size: 0.56rem;

}


.footer-tech {

    margin-top: 0.25rem;

    color: #a2a9b4;

    font-size: 0.5rem;

}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {

        padding-left: 0.7rem;
        padding-right: 0.7rem;

    }

    .hero {

        padding: 1.6rem;

        border-radius: 20px;

    }

    .hero h1 {

        font-size: 2.15rem;

    }

    .hero-description {

        font-size: 0.72rem;

    }

    .hero-meta {

        gap: 0.35rem;

    }

    .meta-item {

        font-size: 0.53rem;

    }

    .brand-title {

        font-size: 0.78rem;

    }

    .brand-subtitle {

        font-size: 0.52rem;

    }

    .status {

        font-size: 0.52rem;

    }

    .result,
    .empty {

        min-height: 235px;

    }

    .about {

        min-height: 155px;
        padding: 1rem;

    }

    .prob {

        min-height: 126px;

        padding: 1rem;

    }

    .prob-value {

        font-size: 1.18rem;

    }

    .about-text {

        font-size: 0.64rem;
        line-height: 1.9;

    }

    .about-icon {

        width: 36px;
        height: 36px;
        margin-bottom: 0.65rem;

    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# TOP BAR
# =========================================================

st.markdown(
    """
<div class="topbar">
    <div class="brand">
        <div class="brand-icon">
            🧭
        </div>
        <div>
            <div class="brand-title">
                Arabic Sentiment Compass
            </div>
            <div class="brand-subtitle">
                Arabic Sentiment Intelligence
            </div>
        </div>
    </div>
    <div class="status">
        <span class="status-dot"></span>
        MODEL READY
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">
            <strong>✦</strong>
            Arabic NLP
            <span>·</span>
            Sentiment Intelligence
        </div>
        <h1>
            بوصلة المشاعر العربية
        </h1>
        <div class="hero-description">
            منصة متخصصة لتحليل المشاعر في النصوص العربية
            باستخدام نموذج BERT حقيقي، مع دعم التحليل الفوري
            وتحليل ملفات CSV على نطاق واسع.
        </div>
        <div class="hero-meta">
            <div class="meta-item">
                <strong>Model</strong>
                AraBERT
            </div>
            <div class="meta-item">
                <strong>Classes</strong>
                3
            </div>
            <div class="meta-item">
                <strong>Max Length</strong>
                128
            </div>
            <div class="meta-item">
                <strong>Inference</strong>
                Real Model
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MAIN TABS
# =========================================================

tab_text, tab_csv, tab_about = st.tabs(
    [
        "✦ تحليل النص",
        "▦ تحليل CSV",
        "◈ ذكاء النموذج",
    ]
)


# =========================================================
# TAB 1 — SINGLE TEXT
# =========================================================

with tab_text:

    st.markdown(
        """
<div class="section">
    <div class="section-eyebrow">
        SENTIMENT ANALYSIS
    </div>
    <div class="section-title">
        حلّل المشاعر في أي نص عربي
    </div>
    <div class="section-description">
        أدخل نصًا عربيًا لتحصل على التصنيف المتوقع
        ودرجة الثقة وتوزيع احتمالات الفئات الثلاث.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.03, 0.97],
        gap="large",
    )


    # =====================================================
    # INPUT
    # =====================================================

    with left:

        st.markdown(
            """
<div class="card">
    <div class="card-head">
        <div class="card-title">
            النص المراد تحليله
        </div>
        <div class="card-caption">
            ARABIC TEXT
        </div>
    </div>
""",
            unsafe_allow_html=True,
        )

        user_text = st.text_area(
            "Arabic text",
            placeholder=(
                "مثال: الخدمة ممتازة والتجربة كانت رائعة جدًا، "
                "وسأكرر التعامل معهم بالتأكيد."
            ),
            height=175,
            label_visibility="collapsed",
            key="single_text",
        )

        c1, c2 = st.columns(
            [1.2, 1],
            gap="small",
        )

        with c1:

            st.markdown(
                '<div class="primary">',
                unsafe_allow_html=True,
            )

            analyze_button = st.button(
                "✦ تحليل المشاعر",
                use_container_width=True,
                key="analyze_single",
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        with c2:

            st.markdown(
                '<div class="secondary">',
                unsafe_allow_html=True,
            )

            clear_button = st.button(
                "مسح",
                use_container_width=True,
                key="clear_single",
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
</div>
""",
            unsafe_allow_html=True,
        )


    # =====================================================
    # RESULT
    # =====================================================

    with right:

        result = st.session_state.get(
            "single_result"
        )

        if result is None:

            st.markdown(
                """
<div class="empty">
    <div class="empty-icon">
        ✦
    </div>
    <div class="empty-title">
        بانتظار النص
    </div>
    <div class="empty-text">
        اكتب النص في الجهة المقابلة واضغط
        "تحليل المشاعر" لعرض النتيجة.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        else:

            label = result["predicted_label"]

            confidence = result["confidence"]

            st.markdown(
                f"""
<div class="result">
    <div class="result-orb">
        <div class="result-symbol">
            {SENTIMENT_ICONS[label]}
        </div>
    </div>
    <div class="result-label">
        {ARABIC_LABELS[label]}
    </div>
    <div class="result-sub">
        Predicted Sentiment · {label}
    </div>
    <div class="confidence">
        درجة الثقة · {confidence * 100:.2f}%
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


    # =====================================================
    # ACTIONS
    # =====================================================

    if clear_button:

        st.session_state.single_result = None

        st.session_state.single_text = ""

        st.rerun()


    if analyze_button:

        if not user_text.strip():

            st.warning(
                "من فضلك اكتب نصًا أولًا."
            )

        else:

            try:

                with st.spinner(
                    "جاري تحليل النص باستخدام النموذج..."
                ):

                    tokenizer, model = load_model()

                    (
                        processed_text,
                        predicted_label,
                        confidence,
                        probabilities_dict,
                    ) = predict_sentiment(
                        user_text,
                        tokenizer,
                        model,
                    )

                st.session_state.single_result = {

                    "processed_text":
                        processed_text,

                    "predicted_label":
                        predicted_label,

                    "confidence":
                        confidence,

                    "probabilities_dict":
                        probabilities_dict,

                }

                st.rerun()

            except Exception as exc:

                st.error(
                    "حدث خطأ أثناء تحليل النص."
                )

                st.exception(exc)


    # =====================================================
    # PROBABILITIES
    # =====================================================

    result = st.session_state.get(
        "single_result"
    )

    if result is not None:

        st.markdown(
            """
<div class="prob-title">
    <span class="prob-title-main">توزيع احتمالات المشاعر</span>
    <span class="prob-title-sub">Model confidence distribution</span>
</div>
""",
            unsafe_allow_html=True,
        )

        cols = st.columns(
            3,
            gap="medium",
        )

        for col, label in zip(
            cols,
            LABELS_ORDER,
        ):

            probability = (
                result["probabilities_dict"][label]
            )

            percentage = probability * 100

            with col:

                st.markdown(
                    f"""
<div class="prob">
    <div class="prob-head">
        <div>
            <div class="prob-name">
                {ARABIC_LABELS[label]}
            </div>
            <div class="prob-en">
                {label}
            </div>
        </div>
        <div class="prob-icon">
            {SENTIMENT_ICONS[label]}
        </div>
    </div>
    <div class="prob-value">
        {percentage:.2f}%
    </div>
    <div class="prob-track">
        <div
            class="prob-fill"
            style="width:{percentage:.2f}%"
        ></div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )


        # =================================================
        # PROCESSED TEXT
        # =================================================

        st.markdown(
            f"""
<div class="processed">
    <div class="processed-label">
        النص بعد المعالجة المسبقة
    </div>
    <div class="processed-value">
        {result["processed_text"]}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


# =========================================================
# TAB 2 — CSV
# =========================================================

with tab_csv:

    st.markdown(
        """
<div class="section">
    <div class="section-eyebrow">
        BATCH ANALYSIS
    </div>
    <div class="section-title">
        تحليل مجموعة نصوص دفعة واحدة
    </div>
    <div class="section-description">
        ارفع ملف CSV، اختر عمود النص، ثم شغّل النموذج
        على جميع السجلات واحصل على ملف نتائج جاهز للتحميل.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        """
<div class="info">
    <strong>
        طريقة الاستخدام:
    </strong>
    ارفع ملف CSV يحتوي على عمود نصي واحد على الأقل،
    ثم اختر العمود المطلوب واضغط على
    "بدء تحليل الملف".
</div>
""",
        unsafe_allow_html=True,
    )


    left, right = st.columns(
        [0.9, 1.1],
        gap="large",
    )


    # =====================================================
    # CSV INPUT
    # =====================================================

    with left:

        st.markdown(
            """
<div class="card">
    <div class="card-head">
        <div class="card-title">
            رفع مجموعة البيانات
        </div>
        <div class="card-caption">
            CSV DATASET
        </div>
    </div>
""",
            unsafe_allow_html=True,
        )


        uploaded_file = st.file_uploader(
            "رفع ملف CSV",
            type=["csv"],
            key="csv_uploader",
            label_visibility="collapsed",
        )


        analyze_csv_button = False


        if uploaded_file is not None:

            try:

                raw_bytes = uploaded_file.getvalue()


                try:

                    csv_text = raw_bytes.decode(
                        "utf-8-sig"
                    )

                except UnicodeDecodeError:

                    csv_text = raw_bytes.decode(
                        "utf-8"
                    )


                reader = csv.DictReader(
                    io.StringIO(csv_text)
                )

                rows = list(reader)

                fieldnames = (
                    reader.fieldnames or []
                )


                if not rows or not fieldnames:

                    st.warning(
                        "الملف فارغ أو لا يحتوي على أعمدة."
                    )

                else:

                    text_column = st.selectbox(
                        "اختر عمود النص",
                        fieldnames,
                        key="csv_text_column",
                    )


                    st.caption(
                        f"عدد السجلات: {len(rows):,}"
                    )


                    st.markdown(
                        '<div class="primary">',
                        unsafe_allow_html=True,
                    )


                    analyze_csv_button = st.button(
                        "✦ بدء تحليل الملف",
                        use_container_width=True,
                        key="analyze_csv",
                    )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


                    if analyze_csv_button:

                        try:

                            with st.spinner(
                                "جاري تحليل الملف باستخدام النموذج..."
                            ):

                                tokenizer, model = (
                                    load_model()
                                )


                                results = []

                                progress = st.progress(
                                    0.0
                                )

                                total = len(rows)


                                for index, row in enumerate(
                                    rows,
                                    start=1,
                                ):

                                    text = str(
                                        row.get(
                                            text_column,
                                            "",
                                        )
                                        or ""
                                    )


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


                                    new_row[
                                        "processed_text"
                                    ] = processed_text


                                    new_row[
                                        "sentiment"
                                    ] = predicted_label


                                    new_row[
                                        "sentiment_ar"
                                    ] = ARABIC_LABELS[
                                        predicted_label
                                    ]


                                    new_row[
                                        "confidence"
                                    ] = round(
                                        confidence,
                                        6,
                                    )


                                    new_row[
                                        "negative_probability"
                                    ] = round(
                                        probabilities_dict[
                                            "Negative"
                                        ],
                                        6,
                                    )


                                    new_row[
                                        "neutral_probability"
                                    ] = round(
                                        probabilities_dict[
                                            "Neutral"
                                        ],
                                        6,
                                    )


                                    new_row[
                                        "positive_probability"
                                    ] = round(
                                        probabilities_dict[
                                            "Positive"
                                        ],
                                        6,
                                    )


                                    results.append(
                                        new_row
                                    )


                                    progress.progress(
                                        index / total
                                    )


                            st.session_state.csv_results = (
                                results
                            )


                            st.session_state.csv_columns = (
                                list(
                                    results[0].keys()
                                )
                                if results
                                else []
                            )


                            st.success(
                                "تم تحليل الملف بنجاح."
                            )


                        except Exception as exc:

                            st.error(
                                "حدث خطأ أثناء تحليل ملف CSV."
                            )

                            st.exception(exc)


            except Exception as exc:

                st.error(
                    "تعذر قراءة ملف CSV."
                )

                st.exception(exc)


        st.markdown(
            """
</div>
""",
            unsafe_allow_html=True,
        )


    # =====================================================
    # CSV RESULTS
    # =====================================================

    with right:

        csv_results = st.session_state.get(
            "csv_results"
        )


        if not csv_results:

            st.markdown(
                """
<div class="empty">
    <div class="empty-icon">
        ▦
    </div>
    <div class="empty-title">
        نتائج تحليل الملف
    </div>
    <div class="empty-text">
        بعد رفع ملف CSV وتشغيل التحليل،
        ستظهر الإحصائيات والنتائج هنا.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


        else:

            counts = Counter(
                row.get(
                    "sentiment",
                    "Neutral",
                )
                for row in csv_results
            )


            total_rows = len(
                csv_results
            )


            positive_count = counts.get(
                "Positive",
                0,
            )

            neutral_count = counts.get(
                "Neutral",
                0,
            )

            negative_count = counts.get(
                "Negative",
                0,
            )


            positive_pct = (
                positive_count / total_rows * 100
                if total_rows
                else 0
            )


            neutral_pct = (
                neutral_count / total_rows * 100
                if total_rows
                else 0
            )


            negative_pct = (
                negative_count / total_rows * 100
                if total_rows
                else 0
            )


            st.markdown(
                """
<div class="section" style="margin-top:0;">
    <div class="section-eyebrow">
        RESULTS
    </div>
    <div class="section-title">
        ملخص التحليل
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


            stat_cols = st.columns(
                4,
                gap="small",
            )


            stats = [

                (
                    "إجمالي السجلات",
                    f"{total_rows:,}",
                    "Total",
                ),

                (
                    "إيجابي",
                    f"{positive_pct:.1f}%",
                    f"{positive_count:,} سجل",
                ),

                (
                    "محايد",
                    f"{neutral_pct:.1f}%",
                    f"{neutral_count:,} سجل",
                ),

                (
                    "سلبي",
                    f"{negative_pct:.1f}%",
                    f"{negative_count:,} سجل",
                ),

            ]


            for col, stat in zip(
                stat_cols,
                stats,
            ):

                title, value, sub = stat

                with col:

                    st.markdown(
                        f"""
<div class="stat">
    <div class="stat-label">
        {title}
    </div>
    <div class="stat-value">
        {value}
    </div>
    <div class="stat-sub">
        {sub}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )


            st.markdown(
                """
<div class="prob-title">
    معاينة النتائج
</div>
""",
                unsafe_allow_html=True,
            )


            preview_rows = csv_results[:50]


            st.dataframe(
                preview_rows,
                use_container_width=True,
                hide_index=True,
            )


            output = io.StringIO()


            writer = csv.DictWriter(
                output,
                fieldnames=st.session_state.get(
                    "csv_columns",
                    [],
                ),
                extrasaction="ignore",
            )


            writer.writeheader()

            writer.writerows(
                csv_results
            )


            st.download_button(
                "↓ تنزيل النتائج كاملة CSV",
                data=output.getvalue().encode(
                    "utf-8-sig"
                ),
                file_name=(
                    "arabic_sentiment_results.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_csv_results",
            )


# =========================================================
# TAB 3 — MODEL INTELLIGENCE
# =========================================================

with tab_about:

    st.markdown(
        """
<div class="section">
    <div class="section-eyebrow">
        MODEL INTELLIGENCE
    </div>
    <div class="section-title">
        نظرة على النموذج والنظام
    </div>
    <div class="section-description">
        معلومات مختصرة عن النموذج ومسار معالجة النص
        وآلية استخراج التنبؤ.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


    # =====================================================
    # MODEL STATS
    # =====================================================

    model_cols = st.columns(
        4,
        gap="medium",
    )


    model_stats = [

        (
            "النموذج",
            "AraBERT",
            "Arabic BERT",
        ),

        (
            "Task",
            "Sentiment",
            "3-class classification",
        ),

        (
            "Max Length",
            "128",
            "Tokenizer sequence length",
        ),

        (
            "Inference",
            "Real",
            "Hugging Face model",
        ),

    ]


    for col, stat in zip(
        model_cols,
        model_stats,
    ):

        title, value, subtitle = stat

        with col:

            st.markdown(
                f"""
<div class="intel">
    <div class="intel-label">
        {title}
    </div>
    <div class="intel-value intel-accent">
        {value}
    </div>
    <div
        class="intel-label"
        style="margin-top:.4rem;"
    >
        {subtitle}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


    # =====================================================
    # PIPELINE
    # =====================================================

    st.markdown(
        """
<div class="prob-title">
    مسار التنبؤ
</div>
""",
        unsafe_allow_html=True,
    )


    pipeline_cols = st.columns(
        5,
        gap="small",
    )


    pipeline_steps = [

        (
            "01",
            "User Input",
            "Arabic text",
        ),

        (
            "02",
            "Preprocessing",
            "Text normalization",
        ),

        (
            "03",
            "Tokenizer",
            "AraBERT tokenizer",
        ),

        (
            "04",
            "Model",
            "Neural inference",
        ),

        (
            "05",
            "Prediction",
            "3-class sentiment",
        ),

    ]


    for col, step in zip(
        pipeline_cols,
        pipeline_steps,
    ):

        number, title, subtitle = step

        with col:

            st.markdown(
                f"""
<div class="pipeline">
    <div class="pipeline-number">
        STEP {number}
    </div>
    <div class="pipeline-title">
        {title}
    </div>
    <div class="pipeline-sub">
        {subtitle}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


    # =====================================================
    # SYSTEM DETAILS
    # =====================================================

    st.markdown(
        """
<div class="prob-title">
    تفاصيل النظام
</div>
""",
        unsafe_allow_html=True,
    )


    row1 = st.columns(
        3,
        gap="medium",
    )



    about_items = [
        (
            "◉",
            "التصنيفات",
            "النظام يصنف النصوص العربية إلى ثلاث فئات: سلبي، محايد، وإيجابي.",
        ),
        (
            "✦",
            "المعالجة المسبقة",
            "يتم تطبيق المعالجة المسبقة على النص قبل إرساله إلى الـTokenizer والنموذج.",
        ),
        (
            "◆",
            "Real Inference",
            "التطبيق يستخدم النموذج الحقيقي من Hugging Face لإجراء التنبؤات.",
        ),
        (
            "↗",
            "Confidence",
            "درجة الثقة المعروضة هي أعلى Softmax Score للناتج المتوقع.",
        ),
        (
            "▦",
            "Batch Analysis",
            "يدعم التطبيق رفع CSV وتحليل مجموعة من النصوص ثم تنزيل النتائج في ملف جديد.",
        ),
        (
            "∞",
            "Arabic NLP",
            "المشروع يركز على تطبيقات تحليل المشاعر للنصوص العربية القصيرة.",
        ),
    ]

    for start_index in (0, 3):
        row = st.columns(3, gap="medium")

        for col, item in zip(row, about_items[start_index:start_index + 3]):
            icon, title, description = item

            with col:
                st.markdown(
                    f'''
<div class="about">
    <div class="about-icon">{icon}</div>
    <div class="about-title" dir="auto">{title}</div>
    <div class="about-text" dir="rtl">{description}</div>
</div>
''',
                    unsafe_allow_html=True,
                )



# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
    <div class="footer-title">
        🧭 Arabic Sentiment Compass
    </div>
    <div class="footer-text">
        Arabic Sentiment Intelligence
    </div>
    <div class="footer-tech">
        Real Model Inference · 3-Class Arabic Sentiment Classification
    </div>
</div>
""",
    unsafe_allow_html=True,
)