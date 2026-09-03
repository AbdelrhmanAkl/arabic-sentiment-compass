🧭 Arabic Sentiment Compass — بوصلة المشاعر العربية

Arabic Sentiment Analysis powered by a Twitter-adapted AraBERT model

Arabic Sentiment Compass is a professional Arabic sentiment-analysis application for classifying Arabic social-media text into Negative, Neutral, or Positive sentiment using a fine-tuned AraBERTv02-Twitter transformer.

The project combines an academic NLP pipeline with a polished Streamlit interface supporting single-text analysis and CSV batch analysis.

## 🚀 Live Demo

[🧭 Try Arabic Sentiment Compass](https://arabic-sentiment-compass-randsalem.streamlit.app/)

✨ Project Overview

Arabic social-media text is challenging because it contains spelling variation, dialectal vocabulary, elongation, emojis, mentions, URLs, and informal writing.

This project investigates a reproducible three-class sentiment-classification pipeline for noisy Arabic Twitter text, with an Egyptian-focused evaluation component and a broader multi-source Arabic Twitter experiment using ASTD and ArSAS.

The final practical system uses:

aubmindlab/bert-base-arabertv02-twitter

🎯 Objectives

Build a reproducible Arabic sentiment-analysis pipeline.

Classify Arabic tweets into three sentiment classes.

Preserve sentiment-bearing information during preprocessing.

Compare classical and transformer-based approaches.

Evaluate the proposed Twitter-adapted AraBERT system.

Provide a professional graduation-project and portfolio application.

Support real-time text inference and CSV batch inference.

🧠 Sentiment Classes

| ID | English | Arabic |
|---:|---|---|
| `0` | Negative | سلبي |
| `1` | Neutral | محايد |
| `2` | Positive | إيجابي |

🤖 Model

Base checkpoint

aubmindlab/bert-base-arabertv02-twitter

Final model

randsalem/arabic-sentiment-compass-arabert

The application loads the tokenizer and sequence-classification model from the Hugging Face Model Hub.

Architecture: AutoModelForSequenceClassification
Classes: 3
Maximum sequence length: 128 tokens
Primary metric: Macro F1

🔄 Prediction Pipeline

User Input
    ↓
Input Validation
    ↓
Arabic Preprocessing
    ↓
Tokenizer
    ↓
AraBERT Forward Pass
    ↓
Logits
    ↓
Softmax
    ↓
Class Probabilities
    ↓
Argmax
    ↓
Predicted Sentiment

The displayed confidence is the maximum Softmax score for the predicted class. It is not a calibrated probability.

🧹 Official Preprocessing

The project uses:

preprocess_arabic_tweet

The pipeline replaces URLs and mentions, removes Arabic diacritics and tatweel, normalizes Alef variants and ى, reduces excessive character repetition, and normalizes whitespace.

The preprocessing logic is kept consistent between the trained system and the application.

📊 Experimental Methodology

Stage I — ASTD-Centered Development

Stage I investigates preprocessing, class weighting, emoji handling, character normalization, training duration, AraSarcasm augmentation, random oversampling, and encoder selection.

Selected light-preprocessed AraBERT:

Accuracy: 73.80%
Macro F1: 71.49%

Strongest recorded ASTD-only encoder, MARBERT:

Accuracy: 75.60%
Macro F1: 73.34%

MARBERT was not used as the final Stage II model because Stage II was designed as a controlled comparison within the AraBERT family.

Stage II — Final Combined Experiment

The final filtered ASTD–ArSAS corpus contains:

13,915 unique examples

Class distribution:

Negative: 5,880
Neutral:  5,062
Positive: 2,973

Split:

Train:      11,132
Validation:  1,391
Test:        1,392

ArSAS examples were retained for the three target sentiment classes with sentiment confidence of at least 0.75.

🏆 Final Results

On the fixed combined Stage II test split:

System

Accuracy

Macro F1

Character TF-IDF + Logistic Regression

81.47%

80.42%

Standard AraBERT

87.36 ± 0.45%

86.83 ± 0.52%

Proposed Twitter-adapted AraBERT

88.22 ± 0.19%

87.73 ± 0.17%

Baseline + Proposed raw-logit ensemble

88.36 ± 0.26%

87.89 ± 0.30%

The proposed single model is the main practical application model. The ensemble is treated as an optional research extension.

🇪🇬 Egyptian-Focused Diagnostic

On the 322 ASTD examples contained inside the unseen Stage II test split:

Accuracy: 70.39 ± 1.00%
Macro F1: 67.19 ± 0.48%

Therefore, the main Stage II score should not be described as an exclusively Egyptian result.

🖥️ Application Features

Single Text Analysis

Arabic RTL interface

Real AraBERT inference

Arabic and English sentiment labels

Confidence score

Probabilities for all three classes

Visual sentiment compass

Input validation

Loading state

Professional result cards

CSV Batch Analysis

CSV upload

User-selectable text column

Real model inference

Processed text output

Sentiment and Arabic sentiment labels

Confidence

Negative / Neutral / Positive probabilities

Downloadable analyzed CSV

Project Information

The interface documents model information, preprocessing, classes, confidence interpretation, maximum sequence length, methodology, and project scope.

🎨 UI Design

The interface follows a premium sentiment-compass identity:

Arabic-first RTL layout

Modern AI dashboard

Dark / premium visual identity

Purple, blue, and cyan accents

Clear information hierarchy

Responsive cards

Professional result visualization

🏗️ Project Structure

arabic-sentiment-compass/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│
├── data/
│   └── ArSAS.txt
│
├── notebooks/
│   └── Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb
│
└── Proposed_Twitter_AraBERT_Tuned_best/
    └── local model checkpoint

The local model checkpoint is excluded from GitHub because of its size.

⚙️ Installation

git clone https://github.com/Randsalem19/arabic-sentiment-compass.git
cd arabic-sentiment-compass
pip install -r requirements.txt

Run the application:

python -m streamlit run app.py

The application loads the final model from Hugging Face.

📦 Main Dependencies

torch
transformers
sentencepiece
safetensors
streamlit

Current application versions:

Streamlit 1.61.1
PyTorch 2.13.0
Transformers 5.16.1

🔬 Reproducibility

Final configuration:

Checkpoint: aubmindlab/bert-base-arabertv02-twitter
Maximum sequence length: 128
Learning rate: 3e-5
Weight decay: 0.01
Maximum epochs: 5
Early stopping patience: 2
Per-device batch size: 16
Gradient accumulation: 2
Effective batch size: 32
Gradient clipping: 1.0
Data split seed: 42
Training seeds: 21, 42, 77
Primary metric: Macro F1
Loss: Class-weighted focal loss, gamma = 2

⚠️ Limitations

The main Stage II dataset is multi-source, not Egyptian-only.

ArSAS is auxiliary Arabic Twitter data and is not treated as a country-level Saudi dialect dataset.

Dataset differences in collection, annotation, topics, and dialect mixture cannot be completely removed.

Confidence is a raw Softmax score and is not calibrated.

The system should not be interpreted as perfect understanding of sarcasm, irony, or implicit sentiment.

Stronger Egyptian-specific evaluation and additional matched Arabic Twitter model comparisons remain valuable future work.

🚀 Future Work

Hugging Face Spaces deployment

Model quantization and inference optimization

Confidence calibration

More extensive Egyptian Arabic evaluation

Error-analysis dashboard

Historical sentiment tracking

Batch analytics and visual reports

API endpoint

Additional Arabic Twitter model comparisons

Explainability and token-level analysis

📚 Research Assets

The repository includes the final Stage II notebook:

Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb

It documents data preparation, evaluation, model comparison, multi-seed experiments, statistical analysis, predictions, and figures.

🔗 Project Links

GitHub: https://github.com/Randsalem19/arabic-sentiment-compass

Hugging Face Model: https://huggingface.co/randsalem/arabic-sentiment-compass-arabert

📄 License

This repository contains project code and experimental artifacts. Dataset and pretrained-model usage remains subject to the original licenses and terms of the respective resources.

⭐ Acknowledgment

This project builds on publicly available Arabic NLP resources and pretrained transformer research, especially the AraBERT family and Arabic Twitter sentiment datasets.

<p align="center">
  <strong>🧭 Arabic Sentiment Compass</strong><br>
  Arabic Sentiment Analysis powered by AraBERT
</p>
