# 🛡️ UK Scam Message Detector

A machine learning text classifier that detects SMS scam messages targeting UK users — built with Naive Bayes, TF-IDF vectorization, and NLP preprocessing.

## 🎯 Project Overview

With SMS scam fraud costing UK consumers millions annually, this project builds a lightweight, interpretable classifier that identifies scam messages with **100% accuracy on the test set** using classical NLP techniques.

## 📊 Key Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Dataset Size | 100 labelled messages |
| Scam Messages | 50 |
| Legitimate Messages | 50 |
| Vectorization | TF-IDF (Unigram + Bigram) |
| Algorithm | Multinomial Naive Bayes |

> **Note on the 100% figure:** the test set is only 20 messages (a 100-message dataset with an 80/20 split), so this accuracy is a proof-of-concept result, not a claim that generalises to real-world traffic. A production system would need a much larger, more diverse, continuously-updated dataset to validate this kind of score.

## 🧠 How It Works

1. **Data Collection** — 100 labelled UK SMS messages (50 scam, 50 legitimate)
2. **Preprocessing** — Lowercasing, punctuation removal, stopword filtering
3. **Feature Extraction** — TF-IDF with unigram and bigram combinations
4. **Classification** — Multinomial Naive Bayes classifier
5. **Evaluation** — Accuracy, precision, recall, confusion matrix

## 🗂️ Repository Structure

```
uk-scam-message-detector/
│
├── uk_scam_detector.py       # Main classifier script
├── dataset.csv               # 100 labelled UK SMS messages
├── results.csv               # Prediction results
├── new_predictions.csv       # New message predictions
├── word_importance.csv       # Top TF-IDF features per class
├── model_report.txt          # Full classification report
├── charts.png                # Visualisation — class distribution & metrics
└── probability_chart.png     # Prediction probability chart
```

## 🔍 Sample Predictions

| Message | Prediction | Confidence |
|---------|------------|------------|
| "URGENT: Your HSBC account has been suspended. Verify now." | 🚨 SCAM | High |
| "Hi, are we still on for lunch tomorrow?" | ✅ Legitimate | High |
| "You've won a £500 Tesco voucher! Claim at..." | 🚨 SCAM | High |

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML Library:** scikit-learn
- **Vectorization:** TF-IDF (sklearn.feature_extraction.text)
- **Classifier:** MultinomialNB
- **Data Handling:** pandas, numpy
- **Visualisation:** matplotlib

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Moharaz01/uk-scam-message-detector.git
cd uk-scam-message-detector

# Install dependencies
pip install scikit-learn pandas numpy matplotlib

# Run the detector
python uk_scam_detector.py
```

## 📈 Results & Insights

- **Top scam indicators:** "urgent", "free", "claim", "winner", "verify", "suspended", "click"
- **Bigrams boost accuracy** by capturing phrases like "click here" and "your account"
- **Naive Bayes** is particularly well-suited for text classification due to its probabilistic word-frequency approach

## 👨‍💻 Author

**Mrithik Das Raz**
- 🔗 [LinkedIn](https://linkedin.com/in/mdrmrithik01)
- 🐙 [GitHub](https://github.com/Moharaz01)
- 📧 mrithikkantidasraz001@gmail.com

---

*Part of a portfolio of AI/ML and Data Science projects — built to solve real-world UK problems.*
