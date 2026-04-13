# UPI Fraud Detection System 🔍

A real-time machine learning system that detects fraudulent UPI transactions using an ensemble of XGBoost and Isolation Forest models.

Built by **Umesh R Kale** during Machine Learning Internship at **EDXcellence**.

## The Problem

India loses over ₹1,087 crore to UPI fraud every year. Banks use basic rule engines that miss sophisticated fraud patterns. This project builds a smarter ML-based detector that:

- Catches fraud in **real time** before money is lost
- Explains **why** a transaction was flagged
- Handles **batch screening** of thousands of transactions
- Detects **new fraud patterns** it has never seen before

---

## Results

| Metric | Value |
|--------|-------|
| Fraud Detection Rate | **100%** |
| False Alarm Rate | **0.68%** |
| PR-AUC | **0.978** |
| ROC-AUC | **0.999** |
| Frauds Caught (test set) | **100 / 100** |
| False Alarms (test set) | **13 / 1,900** |

---

## Features

### Live Analyser
- Enter any transaction details and get an instant fraud verdict
- Risk score from 0–100 with confidence level
- Human-readable explanation of why it was flagged

### Model Dashboard
- Confusion matrix and feature importance charts
- Full model architecture overview
- Precision-Recall curve

### Batch Upload
- Upload a CSV of transactions
- Get all transactions scored and sorted by risk
- Download flagged results as CSV

---

## Fraud Patterns Detected

| Pattern | Description |
|---------|-------------|
| Large amount | Transaction 8–20× above user's average |
| Late night | Transfer between 1–4 AM to unknown merchant |
| Rapid fire | Multiple small transactions in quick succession |
| New device | Large amount from a device never used before |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Primary model | XGBoost (supervised) |
| Anomaly detection | Isolation Forest (unsupervised) |
| Class balancing | SMOTE |
| UI | Streamlit |
| Visualisation | Plotly |
| Data | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
upi-fraud-detector/
├── app.py                          # Streamlit UI
├── requirements.txt
├── README.md
├── data/
│   └── generate_dataset.py         # Synthetic UPI data generator
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── feature_engineering.py      # 25 ML features
│   ├── train_model.py              # XGBoost + Isolation Forest
│   ├── predict.py                  # Inference module
│   └── explainer.py                # Fraud explanation engine
└── models/
    ├── xgboost_model.pkl
    ├── isolation_forest.pkl
    └── scaler.pkl
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/UMESH-KALE0777/UPI-Fraud-Detection-System.git
cd UPI-Fraud-Detection-System
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate dataset and train models**
```bash
python data/generate_dataset.py
python src/feature_engineering.py
python src/train_model.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Sample Scenarios to Test

| Scenario | Amount | Hour | Merchant | Expected |
|----------|--------|------|----------|----------|
| Normal payment | ₹450 | 2 PM | Grocery | ✅ Safe |
| Suspicious | ₹12,000 | 11 PM | Unknown | ⚠️ Warning |
| Fraud | ₹45,000 | 2 AM | Unknown + new device | 🚨 Alert |

---

## Why Two Models?

Most student projects use only one model. This project uses two:

- **XGBoost** — learns from labelled fraud data with high precision
- **Isolation Forest** — unsupervised, catches NEW fraud patterns the labelled data hasn't seen

The ensemble (60% XGB + 40% IF) is more robust than either model alone — exactly how real fraud systems at Razorpay and PhonePe work.

---

## Internship

This project was built during a **Machine Learning with Python** internship at **EDXcellence** (Jan 2026 – Mar 2026).

Certificate ID: `INT-949-3285F20A`

---

## Connect

- LinkedIn: [linkedin.com/in/umesh-kale-a00520342](https://www.linkedin.com/in/umesh-kale-a00520342)
- GitHub: [github.com/UMESH-KALE0777](https://github.com/UMESH-KALE0777)
- Email: umesh.kale09192@gmail.com
