# 🧭 Arabic Sentiment Compass

> **Arabic Sentiment Analysis powered by Twitter-adapted AraBERT**

An end-to-end Arabic Natural Language Processing system for classifying Arabic social-media text into **Negative, Neutral, and Positive** sentiment using a fine-tuned Twitter-adapted AraBERT transformer.

The project combines a reproducible NLP research pipeline with a professional **Arabic RTL Streamlit application** supporting both real-time single-text analysis and CSV batch inference.

---

## 🚀 Live Demo

### 🌐 Try Arabic Sentiment Compass

👉 [Launch the Live Streamlit Application](https://arabic-sentiment-compasss.streamlit.app/)

The application supports:

* Arabic single-text sentiment analysis
* Real-time transformer inference
* Arabic RTL interface
* Sentiment probabilities
* Confidence score
* Visual sentiment compass
* CSV batch sentiment analysis
* Downloadable prediction results

---

## 🎯 Project Overview

Arabic sentiment analysis presents several challenges that are less pronounced in standard English NLP tasks.

Arabic social-media text can contain:

* Dialectal vocabulary
* Spelling variations
* Character elongation
* Arabic diacritics
* Emojis
* Mentions
* URLs
* Informal writing
* Noisy user-generated content

This project investigates a reproducible three-class sentiment classification pipeline designed specifically for noisy Arabic Twitter-style text.

The final practical system uses a **Twitter-adapted AraBERT transformer** and provides an interactive application for real-world inference.

---

## 🧠 What Does the System Do?

The system takes Arabic text as input and predicts one of three sentiment classes:

|  ID | Sentiment | Arabic |
| --: | --------- | ------ |
| `0` | Negative  | سلبي   |
| `1` | Neutral   | محايد  |
| `2` | Positive  | إيجابي |

### Example

```text
Input:
الخدمة ممتازة والتجربة كانت رائعة جدًا!

Prediction:
Positive — إيجابي
```

The application also displays the model's scores for all three classes.

---

## 🤖 Model

### Base Model

```text
aubmindlab/bert-base-arabertv02-twitter
```

The model is specifically adapted to Twitter-style Arabic text, making it a strong fit for noisy social-media sentiment analysis.

### Final Application Model

```text
randsalem/arabic-sentiment-compass-arabert
```

The deployed application loads the trained tokenizer and sequence-classification model from the Hugging Face Model Hub.

### Architecture

```text
AutoModelForSequenceClassification
```

### Configuration

| Parameter               | Value          |
| ----------------------- | -------------- |
| Number of Classes       | 3              |
| Maximum Sequence Length | 128 tokens     |
| Primary Metric          | Macro F1       |
| Model Family            | AraBERT        |
| Domain Adaptation       | Twitter Arabic |

---

## 🔄 End-to-End Prediction Pipeline

```text
                  Arabic Text
                      │
                      ▼
              Input Validation
                      │
                      ▼
             Arabic Preprocessing
                      │
                      ▼
                  Tokenizer
                      │
                      ▼
          Twitter-adapted AraBERT
                      │
                      ▼
                    Logits
                      │
                      ▼
                  Softmax
                      │
                      ▼
             Class Probabilities
                      │
                      ▼
                 Argmax
                      │
                      ▼
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Negative      Neutral       Positive
      سلبي          محايد          إيجابي
```

### Confidence Interpretation

The displayed confidence represents the **maximum Softmax score** for the predicted class.

It should **not** be interpreted as a calibrated probability.

---

## 🧹 Arabic Text Preprocessing

The project uses a dedicated preprocessing function:

```text
preprocess_arabic_tweet
```

The preprocessing pipeline preserves important sentiment-bearing information while reducing common noise in Arabic social-media text.

It includes:

* URL normalization
* Mention normalization
* Arabic diacritic removal
* Tatweel removal
* Alef normalization
* `ى` normalization
* Excessive character repetition reduction
* Whitespace normalization

The same preprocessing logic is maintained between the trained system and the deployed application to reduce training/inference inconsistencies.

---

# 📊 Experimental Methodology

The research workflow was divided into two major stages.

## Stage I — ASTD-Centered Development

The first stage investigated several components of Arabic sentiment classification, including:

* Arabic preprocessing
* Class weighting
* Emoji handling
* Character normalization
* Training duration
* AraSarcasm augmentation
* Random oversampling
* Encoder selection

### Selected Light-Preprocessed AraBERT

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **73.80%** |
| Macro F1 | **71.49%** |

### Strongest Recorded ASTD-Only Encoder

MARBERT achieved:

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **75.60%** |
| Macro F1 | **73.34%** |

MARBERT was not selected as the final Stage II application model because Stage II was designed as a controlled comparison within the AraBERT family.

---

# 🧪 Stage II — Final Combined Experiment

The final experimental setup combined filtered examples from:

* ASTD
* ArSAS

The resulting corpus contained:

```text
13,915 unique examples
```

### Class Distribution

| Sentiment | Samples |
| --------- | ------: |
| Negative  |   5,880 |
| Neutral   |   5,062 |
| Positive  |   2,973 |

### Data Split

| Split      | Samples |
| ---------- | ------: |
| Training   |  11,132 |
| Validation |   1,391 |
| Test       |   1,392 |

ArSAS examples were retained for the three target sentiment classes using a sentiment-confidence threshold of at least `0.75`.

---

# 🏆 Final Model Results

Evaluation was performed on the fixed Stage II test split.

| System                                 |          Accuracy |          Macro F1 |
| -------------------------------------- | ----------------: | ----------------: |
| Character TF-IDF + Logistic Regression |        **81.47%** |        **80.42%** |
| Standard AraBERT                       | **87.36 ± 0.45%** | **86.83 ± 0.52%** |
| **Twitter-adapted AraBERT**            | **88.22 ± 0.19%** | **87.73 ± 0.17%** |
| Baseline + Raw-Logit Ensemble          | **88.36 ± 0.26%** | **87.89 ± 0.30%** |

### 🥇 Practical Application Model

The **Twitter-adapted AraBERT** is the main model used by the practical application.

The ensemble is treated as an optional research extension rather than the primary deployed model.

---

# 🇪🇬 Egyptian-Focused Evaluation

An additional diagnostic evaluation was performed on the **322 ASTD examples** contained within the unseen Stage II test split.

Results:

| Metric   |             Score |
| -------- | ----------------: |
| Accuracy | **70.39 ± 1.00%** |
| Macro F1 | **67.19 ± 0.48%** |

### Important Interpretation

The main Stage II result should **not** be described as an exclusively Egyptian Arabic result.

The final corpus is multi-source, and the Egyptian-focused evaluation is provided separately to measure performance on the available ASTD subset.

---

# 🖥️ Streamlit Application

The project includes a professional Arabic-first Streamlit interface.

## Single Text Analysis

The application supports:

* Arabic RTL interface
* Real AraBERT inference
* Arabic sentiment labels
* English sentiment labels
* Confidence score
* Probabilities for all three classes
* Visual sentiment compass
* Input validation
* Loading states
* Professional prediction cards

### Example Workflow

```text
Enter Arabic Text
       ↓
Preprocess
       ↓
AraBERT Inference
       ↓
Sentiment Prediction
       ↓
Probability Distribution
       ↓
Visual Result
```

---

# 📁 CSV Batch Analysis

The application also supports batch sentiment analysis.

Users can:

1. Upload a CSV file
2. Select the text column
3. Run the trained model
4. Generate sentiment predictions
5. View confidence scores
6. Inspect class probabilities
7. Download the analyzed CSV

The output includes:

```text
Processed Text
Sentiment
Arabic Sentiment
Confidence
Negative Probability
Neutral Probability
Positive Probability
```

This makes the application suitable for analyzing larger collections of Arabic social-media text.

---

# 🎨 UI & UX

Arabic Sentiment Compass follows a premium Arabic-first visual identity.

### Design Characteristics

* Arabic RTL layout
* Dark / premium interface
* Modern AI dashboard
* Purple, blue, and cyan visual accents
* Responsive information cards
* Clear prediction hierarchy
* Visual sentiment compass
* Professional result presentation

The interface is designed to make Arabic NLP inference accessible to both technical and non-technical users.

---

# 🏗️ Project Architecture

```text
                         User
                          │
              ┌───────────┴───────────┐
              │                       │
         Single Text              CSV File
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                Input Validation
                          │
                          ▼
               Arabic Preprocessing
                          │
                          ▼
                     Tokenizer
                          │
                          ▼
              Twitter-adapted AraBERT
                          │
                          ▼
                        Logits
                          │
                          ▼
                       Softmax
                          │
                          ▼
                Sentiment Prediction
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Negative     Neutral      Positive
             │            │            │
             └────────────┼────────────┘
                          ▼
                   Results & Export
```

---

# 📂 Project Structure

```text
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
└── notebooks/
    └── Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb
```

The large trained model checkpoint is not stored directly in the GitHub repository because of its size.

The deployed application loads the model from the Hugging Face Model Hub.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/arabic-sentiment-compass.git
```

## 2. Enter the Project

```bash
cd arabic-sentiment-compass
```

## 3. Create a Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate the Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 📦 Main Technologies

| Category             | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| NLP                  | Arabic NLP / Transformers |
| Transformer          | AraBERT                   |
| Deep Learning        | PyTorch                   |
| Model Library        | Hugging Face Transformers |
| Tokenization         | Hugging Face Tokenizer    |
| Application          | Streamlit                 |
| Data Processing      | Pandas / NumPy            |
| Model Hosting        | Hugging Face Model Hub    |
| Deployment           | Streamlit Community Cloud |
| Development          | Jupyter Notebook          |

---

# 🔬 Reproducibility

The final training configuration includes:

| Configuration           | Value                                     |
| ----------------------- | ----------------------------------------- |
| Base Checkpoint         | `aubmindlab/bert-base-arabertv02-twitter` |
| Maximum Sequence Length | `128`                                     |
| Learning Rate           | `3e-5`                                    |
| Weight Decay            | `0.01`                                    |
| Maximum Epochs          | `5`                                       |
| Early Stopping Patience | `2`                                       |
| Per-device Batch Size   | `16`                                      |
| Gradient Accumulation   | `2`                                       |
| Effective Batch Size    | `32`                                      |
| Gradient Clipping       | `1.0`                                     |
| Data Split Seed         | `42`                                      |
| Training Seeds          | `21, 42, 77`                              |
| Primary Metric          | Macro F1                                  |
| Loss                    | Class-weighted Focal Loss                 |
| Focal Gamma             | `2`                                       |

---

# ⚠️ Limitations

Despite the strong experimental results, several limitations should be considered.

### Dataset Composition

The main Stage II dataset is multi-source and should not be treated as an Egyptian-only dataset.

### Arabic Dialect Variation

Arabic social-media text contains substantial variation across dialects, regions, topics, and writing styles.

### Dataset Differences

Differences in:

* Collection methodology
* Annotation guidelines
* Topics
* Dialect mixture

cannot be completely eliminated.

### Confidence Calibration

The displayed confidence is a raw Softmax score and is **not calibrated**.

### Sarcasm & Irony

The system should not be interpreted as having perfect understanding of:

* Sarcasm
* Irony
* Implicit sentiment
* Context-dependent expressions

---

# 🚀 Future Work

Potential improvements include:

* Hugging Face Spaces deployment
* Model quantization
* Faster inference optimization
* Confidence calibration
* More extensive Egyptian Arabic evaluation
* Error-analysis dashboard
* Historical sentiment tracking
* Advanced batch analytics
* API endpoint
* Additional Arabic Twitter model comparisons
* Explainability analysis
* Token-level sentiment interpretation

---

# 📓 Research Notebook

The repository includes the final Stage II research notebook:

```text
Stage_II_Combined_ASTD_ArSAS_AraBERT_SentimentCompass_ipynb.ipynb
```

The notebook documents:

* Data preparation
* Dataset filtering
* Model training
* Evaluation
* Model comparison
* Multi-seed experiments
* Statistical analysis
* Predictions
* Visualizations

---

# 🎯 What This Project Demonstrates

This project demonstrates an end-to-end **Arabic NLP / Transformer-based Machine Learning workflow**:

```text
Arabic Social-Media Data
          ↓
Data Preparation
          ↓
Arabic Text Preprocessing
          ↓
Transformer Fine-Tuning
          ↓
Multi-Seed Evaluation
          ↓
Model Comparison
          ↓
Error & Diagnostic Analysis
          ↓
Production Model
          ↓
Streamlit Application
          ↓
Real-Time + Batch Inference
```

### Core Skills Demonstrated

```text
✓ Arabic NLP
✓ Sentiment Analysis
✓ Transformer Fine-Tuning
✓ AraBERT
✓ Hugging Face Transformers
✓ PyTorch
✓ Text Preprocessing
✓ Class Imbalance Handling
✓ Focal Loss
✓ Multi-Seed Evaluation
✓ Model Benchmarking
✓ Macro F1 Evaluation
✓ Streamlit
✓ Hugging Face Model Hub
✓ Production Inference
✓ Batch NLP Processing
```

---

# 🔗 Project Links

### GitHub Repository

[View the Source Code](https://github.com/AbdelrhmanAkl/arabic-sentiment-compass)

### Live Application

[Launch Arabic Sentiment Compass](https://arabic-sentiment-compasss.streamlit.app/)

### Hugging Face Model

[View the Arabic Sentiment Compass Model](https://huggingface.co/randsalem/arabic-sentiment-compass-arabert)

---

# 📄 License

This repository contains project code and experimental artifacts.

Dataset and pretrained-model usage remain subject to the original licenses and terms of their respective resources.

---

# ⭐ Acknowledgments

This project builds on publicly available Arabic NLP resources and pretrained transformer research, particularly:

* AraBERT
* Arabic Twitter sentiment datasets
* ASTD
* ArSAS
* Hugging Face Transformers

---

## 🧭 Arabic Sentiment Compass

**Arabic Sentiment Analysis powered by Twitter-adapted AraBERT.**

Built as an end-to-end Arabic NLP research and deployment project.
