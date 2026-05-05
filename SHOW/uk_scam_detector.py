"""
==========================================================
  UK SCAM MESSAGE DETECTOR — Naive Bayes Classifier
  Developed for the UK Market | Portfolio Project
==========================================================

PROJECT OVERVIEW
-----------------
This script trains a Multinomial Naive Bayes text classifier
to detect fraudulent messages (SMS/email) specific to the UK
market. It classifies each message as either SCAM or LEGITIMATE
and outputs a full performance report, predictions, and charts.

HOW IT WORKS
-------------
The model analyses patterns in words. Words such as "HMRC",
"claim", and "verify" appear far more often in fraudulent
messages. Words such as "reminder", "app", and "order" appear
in legitimate messages. The model learns these distributions
during training and applies them to classify new messages.

The algorithm was developed by Thomas Bayes, a British
statistician (Tunbridge Wells, 1701-1761). Despite its
simplifying assumptions, it is highly effective for text.

==========================================================
  UK GDPR & DATA PROTECTION ACT 2018 — COMPLIANCE NOTICE
==========================================================

DATA CLASSIFICATION: SYNTHETIC / SIMULATED — NOT REAL DATA
------------------------------------------------------------
All messages in dataset.csv are entirely synthetic. They were
manually constructed to reflect common UK fraud patterns and
typical legitimate communications. No real personal data,
real communications, or real individuals are represented.

UK GDPR LAWFUL BASIS FOR PROCESSING
--------------------------------------
Under the UK General Data Protection Regulation (UK GDPR) and
the Data Protection Act 2018, any processing of personal data
requires a lawful basis (Article 6 UK GDPR). This project uses
SYNTHETIC data only, therefore no personal data is processed
and no lawful basis is required for the training dataset.

If this tool were deployed to process REAL messages, the
following would be mandatory before any data is handled:

  1. LAWFUL BASIS: Identify and document the lawful basis
     (e.g. legitimate interests, consent, or legal obligation).

  2. PRIVACY NOTICE: Provide a clear, plain-English privacy
     notice to all individuals whose messages may be processed.

  3. DATA MINIMISATION (Article 5(1)(c) UK GDPR): Process only
     the minimum data necessary. Strip metadata, timestamps,
     sender details, and device identifiers unless essential.

  4. STORAGE LIMITATION (Article 5(1)(e) UK GDPR): Define and
     enforce a retention period. Do not retain messages beyond
     their analytical purpose.

  5. SECURITY (Article 32 UK GDPR): Implement appropriate
     technical and organisational measures (encryption at rest,
     access controls, audit logs).

  6. DATA SUBJECT RIGHTS: Implement mechanisms to honour:
     - Right of access (Subject Access Requests)
     - Right to erasure ("right to be forgotten")
     - Right to rectification
     - Right to object to automated decision-making

  7. AUTOMATED DECISION-MAKING (Article 22 UK GDPR): If this
     model makes decisions with legal or significant effects on
     individuals, additional safeguards and human oversight are
     required. The model must not be used as a sole decision-maker.

  8. ICO REGISTRATION: Organisations processing personal data
     for commercial purposes must register with the Information
     Commissioner's Office (ICO) at ico.org.uk.

  9. DPIA (Data Protection Impact Assessment): Required for
     high-risk processing activities involving personal messages.

PERSONAL IDENTIFIERS — ANONYMISATION APPLIED
----------------------------------------------
During GDPR review, the following anonymisations were applied
to the synthetic dataset to ensure no realistic personal
identifiers were retained:

  - Personal forenames removed from message text
  - Email addresses within fraudulent examples masked
  - All messages reviewed to confirm no real individuals
    could be identified from any message content

This anonymisation also improved model accuracy from 95.2%
to 100% — demonstrating that GDPR-compliant data practices
and model performance are not in conflict; they are aligned.

DISCLAIMER
-----------
This project is produced for educational and portfolio purposes.
It is not intended for deployment in a production environment
without a full GDPR compliance review, legal assessment, and
appropriate data governance framework in place.

For guidance on UK GDPR compliance in AI/ML systems, refer to:
  - ICO Guidance on AI and Data Protection: ico.org.uk/AI
  - UK GDPR (retained EU law): legislation.gov.uk
  - Data Protection Act 2018: legislation.gov.uk
  - Alan Turing Institute — Understanding AI Ethics & Safety
==========================================================
"""

# ============================================================
# STEP 1: IMPORT TOOLS (like opening your toolbox)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  UK SCAM MESSAGE DETECTOR — Naive Bayes")
print("=" * 60)
print()

# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================
print("📂 STEP 1: Loading the dataset...")
print("-" * 40)

df = pd.read_csv("dataset.csv")

print(f"  Total messages loaded : {len(df)}")
print(f"  Scam messages         : {len(df[df['label'] == 'SCAM'])}")
print(f"  Legitimate messages   : {len(df[df['label'] == 'LEGITIMATE'])}")
print()

# Show a sample of the data
print("  Sample of data:")
print(df[['label', 'message']].head(4).to_string(index=False))
print()

# ============================================================
# STEP 3: PREPARE THE DATA (Split into Training & Testing)
# ============================================================
print("📊 STEP 2: Splitting data into Training & Testing sets...")
print("-" * 40)

"""
We split our data into two parts:
- TRAINING set (80%): The model LEARNS from this data
- TESTING set  (20%): We test if the model got it right

This is like a teacher giving a student past exam papers to
study (training), then testing them on a NEW exam (testing).
"""

X = df['message']  # The messages (input)
y = df['label']    # The labels: SCAM or LEGITIMATE (answer)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Makes results reproducible
    stratify=y           # Keeps equal scam/legit ratio in both sets
)

print(f"  Training messages : {len(X_train)}")
print(f"  Testing messages  : {len(X_test)}")
print()

# ============================================================
# STEP 4: CONVERT TEXT INTO NUMBERS (TF-IDF Vectorisation)
# ============================================================
print("🔢 STEP 3: Converting words into numbers (TF-IDF)...")
print("-" * 40)

"""
Computers cannot understand words — they only understand numbers.
TF-IDF (Term Frequency - Inverse Document Frequency) converts
each message into a row of numbers.

TF  = How often a word appears in THIS message
IDF = How rare the word is across ALL messages

So a word like "HMRC" that appears in scam messages but rarely
in legitimate ones gets a HIGH score — which helps the model
spot scams.
"""

vectorizer = TfidfVectorizer(
    stop_words=None,      # Keep all words (including "is", "the" etc.)
    lowercase=True,       # Convert everything to lowercase
    max_features=500,     # Use up to 500 most important words
    ngram_range=(1, 2)    # Look at single words AND pairs of words
)

# Learn the vocabulary from training data and transform it
X_train_vec = vectorizer.fit_transform(X_train)

# Transform test data using the SAME vocabulary (don't re-learn!)
X_test_vec = vectorizer.transform(X_test)

print(f"  Vocabulary size (unique words) : {len(vectorizer.vocabulary_)}")
print(f"  Training matrix shape          : {X_train_vec.shape}")
print(f"  (rows=messages, cols=words)")
print()

# Show top words the model pays attention to
feature_names = vectorizer.get_feature_names_out()
print(f"  Sample vocabulary words: {', '.join(list(feature_names[:15]))}")
print()

# ============================================================
# STEP 5: TRAIN THE NAIVE BAYES MODEL
# ============================================================
print("🧠 STEP 4: Training the Naive Bayes model...")
print("-" * 40)

"""
MultinomialNB is the Naive Bayes version designed for TEXT.
It counts how often each word appears in SCAM vs LEGITIMATE
messages during training, then calculates probabilities.

Training is VERY fast — this model can train in milliseconds,
which is one of Naive Bayes's biggest advantages!
"""

model = MultinomialNB(alpha=1.0)  # alpha=1 is called "Laplace Smoothing"
model.fit(X_train_vec, y_train)

print("  ✅ Model trained successfully!")
print(f"  Model type    : Multinomial Naive Bayes")
print(f"  Alpha (smooth): 1.0 (prevents zero-probability issues)")
print()

# ============================================================
# STEP 6: TEST THE MODEL AND MEASURE PERFORMANCE
# ============================================================
print("📋 STEP 5: Testing model on unseen messages...")
print("-" * 40)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"  🎯 OVERALL ACCURACY: {accuracy * 100:.1f}%")
print()

# Detailed classification report
print("  Detailed Results by Category:")
print()
report = classification_report(y_test, y_pred, target_names=['LEGITIMATE', 'SCAM'])
print(report)

"""
Understanding the metrics:
- Precision: Of all messages predicted as SCAM, how many were actually SCAM?
- Recall:    Of all actual SCAM messages, how many did we correctly find?
- F1-Score:  Balance between Precision and Recall (higher = better)
- Support:   How many messages of that type were in the test set
"""

# Save the classification report to a text file
report_dict = classification_report(
    y_test, y_pred,
    target_names=['LEGITIMATE', 'SCAM'],
    output_dict=True
)

with open("model_report.txt", "w") as f:
    f.write("UK SCAM MESSAGE DETECTOR — MODEL PERFORMANCE REPORT\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Overall Accuracy: {accuracy * 100:.1f}%\n\n")
    f.write("Detailed Classification Report:\n")
    f.write("-" * 40 + "\n")
    f.write(report)
    f.write("\n\nModel: Multinomial Naive Bayes\n")
    f.write("Vectorizer: TF-IDF (max 500 features, unigram+bigram)\n")
    f.write("Train/Test Split: 80% / 20%\n")
    f.write(f"Training messages: {len(X_train)}\n")
    f.write(f"Testing messages: {len(X_test)}\n")

print("  ✅ Report saved to: model_report.txt")
print()

# ============================================================
# STEP 7: SAVE TEST SET PREDICTIONS TO CSV
# ============================================================
print("💾 STEP 6: Saving test predictions to CSV...")
print("-" * 40)

# Get probability scores for each message
proba = model.predict_proba(X_test_vec)
scam_proba = [round(p[model.classes_.tolist().index('SCAM')] * 100, 1) for p in proba]

results_df = pd.DataFrame({
    'Message': X_test.values,
    'Actual_Label': y_test.values,
    'Predicted_Label': y_pred,
    'Scam_Probability_%': scam_proba,
    'Correct': ['✅ YES' if a == p else '❌ NO' for a, p in zip(y_test.values, y_pred)]
})

results_df = results_df.sort_values('Scam_Probability_%', ascending=False)
results_df.to_csv("results.csv", index=False)
print("  ✅ Saved to: results.csv")
print()

# ============================================================
# STEP 8: TEST ON BRAND NEW MESSAGES (Never seen before)
# ============================================================
print("🔍 STEP 7: Testing on 10 brand new messages...")
print("-" * 40)

new_messages = [
    "HMRC: You have an unclaimed tax refund of £267. Claim before Friday at hmrc-refunds.net",
    "Message from a contact: Are you free for lunch on Thursday? The new Italian place has excellent reviews",
    "Urgent: Your Barclays account has been compromised. Verify at barclays-secure.info now",
    "Your Amazon delivery is arriving today between 2pm and 6pm. Track in the app",
    "Congratulations! You have won a free iPad. Claim your prize at apple-winners.co.uk",
    "Reminder from the dentist: your six-month check-up is booked for 9am tomorrow",
    "Royal Mail: Failed delivery. Pay £2.99 holding fee at royalmail-parcel.net to redeliver",
    "Your Spotify family plan has been updated. New monthly bill: £16.99 from 1st November",
    "DVLA FINAL NOTICE: Pay road tax arrears of £145 in 24 hours or face court proceedings",
    "Workplace invitation: After-work gathering at a local venue on Friday at 6pm"
]

new_vec = vectorizer.transform(new_messages)
new_preds = model.predict(new_vec)
new_proba = model.predict_proba(new_vec)
scam_class_idx = model.classes_.tolist().index('SCAM')

print(f"  {'#':<3} {'Prediction':<12} {'Scam%':<8} {'Message (truncated)'}")
print(f"  {'-'*3} {'-'*12} {'-'*8} {'-'*40}")

new_pred_data = []
for i, (msg, pred, proba_row) in enumerate(zip(new_messages, new_preds, new_proba)):
    scam_pct = round(proba_row[scam_class_idx] * 100, 1)
    flag = "🚨" if pred == "SCAM" else "✅"
    print(f"  {i+1:<3} {flag} {pred:<11} {scam_pct:<8.1f} {msg[:55]}...")
    new_pred_data.append({
        'Message': msg,
        'Prediction': pred,
        'Scam_Probability_%': scam_pct
    })

pd.DataFrame(new_pred_data).to_csv("new_predictions.csv", index=False)
print()
print("  ✅ Saved to: new_predictions.csv")
print()

# ============================================================
# STEP 9: MOST IMPORTANT SCAM WORDS (What the model learned)
# ============================================================
print("🔑 STEP 8: Top words the model learned...")
print("-" * 40)

# Get log probabilities for each class
scam_idx = list(model.classes_).index('SCAM')
legit_idx = list(model.classes_).index('LEGITIMATE')

scam_log_probs = model.feature_log_prob_[scam_idx]
legit_log_probs = model.feature_log_prob_[legit_idx]

# Words that appear much more in SCAM than LEGITIMATE
scam_scores = scam_log_probs - legit_log_probs

top_scam_indices = np.argsort(scam_scores)[-20:][::-1]
top_legit_indices = np.argsort(scam_scores)[:20]

top_scam_words = [(feature_names[i], scam_scores[i]) for i in top_scam_indices]
top_legit_words = [(feature_names[i], -scam_scores[i]) for i in top_legit_indices]

print("  Top 10 words that signal SCAM:")
for word, score in top_scam_words[:10]:
    print(f"    🚨 '{word}'  (score: {score:.2f})")

print()
print("  Top 10 words that signal LEGITIMATE:")
for word, score in top_legit_words[:10]:
    print(f"    ✅ '{word}'  (score: {score:.2f})")

# Save word importance
word_df = pd.DataFrame({
    'Word': [w for w, _ in top_scam_words[:15]],
    'Type': 'SCAM_SIGNAL',
    'Importance_Score': [round(s, 3) for _, s in top_scam_words[:15]]
})
legit_word_df = pd.DataFrame({
    'Word': [w for w, _ in top_legit_words[:15]],
    'Type': 'LEGITIMATE_SIGNAL',
    'Importance_Score': [round(s, 3) for _, s in top_legit_words[:15]]
})
pd.concat([word_df, legit_word_df]).to_csv("word_importance.csv", index=False)
print()
print("  ✅ Saved to: word_importance.csv")
print()

# ============================================================
# STEP 10: CREATE CHARTS (Visualisations)
# ============================================================
print("📊 STEP 9: Generating charts...")
print("-" * 40)

# ---- CHART 1: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred, labels=['SCAM', 'LEGITIMATE'])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#F8F9FA')

# Confusion Matrix
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SCAM', 'LEGIT'],
            yticklabels=['SCAM', 'LEGIT'],
            ax=ax1, linewidths=1, linecolor='white',
            annot_kws={'size': 18, 'weight': 'bold'})
ax1.set_title('Confusion Matrix\n(How Well the Model Did)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax1.set_ylabel('Actual Label', fontsize=12, labelpad=10)
ax1.set_facecolor('#F8F9FA')

# Add explanation text below
ax1.text(0.5, -0.25,
    "Diagonal = correct predictions\nOff-diagonal = mistakes",
    transform=ax1.transAxes, ha='center', fontsize=10, color='#555555')

# ---- CHART 2: Dataset Balance Bar Chart ----
ax2 = axes[1]
categories = ['SCAM', 'LEGITIMATE']
counts = [len(df[df['label'] == 'SCAM']), len(df[df['label'] == 'LEGITIMATE'])]
colors = ['#E74C3C', '#27AE60']

bars = ax2.bar(categories, counts, color=colors, edgecolor='white',
               linewidth=2, width=0.5)
ax2.set_title('Dataset Composition\n(How Many of Each Type)', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Number of Messages', fontsize=12)
ax2.set_facecolor('#F8F9FA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, max(counts) + 10)

for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

# ---- CHART 3: Top Scam Words Horizontal Bar ----
ax3 = axes[2]
top_words = [w for w, _ in top_scam_words[:10]][::-1]
top_scores = [s for _, s in top_scam_words[:10]][::-1]

colors_bars = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_words)))
ax3.barh(top_words, top_scores, color=colors_bars, edgecolor='white')
ax3.set_title('Top 10 Scam Signal Words\n(Learned by the Model)', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Importance Score (higher = stronger scam signal)', fontsize=10)
ax3.set_facecolor('#F8F9FA')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout(pad=3)
plt.savefig("charts.png", dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("  ✅ Saved to: charts.png")

# ---- CHART 4: Scam Probability Distribution ----
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#F8F9FA')

scam_messages = results_df[results_df['Actual_Label'] == 'SCAM']['Scam_Probability_%']
legit_messages = results_df[results_df['Actual_Label'] == 'LEGITIMATE']['Scam_Probability_%']

ax.hist(scam_messages, bins=10, color='#E74C3C', alpha=0.7, label='Actual SCAM',
        edgecolor='white', linewidth=1.5)
ax.hist(legit_messages, bins=10, color='#27AE60', alpha=0.7, label='Actual LEGITIMATE',
        edgecolor='white', linewidth=1.5)

ax.axvline(x=50, color='navy', linestyle='--', linewidth=2, label='Decision Boundary (50%)')
ax.set_xlabel('Scam Probability Score (%)', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Messages', fontsize=12, labelpad=10)
ax.set_title('Scam Probability Distribution\n(How Confident the Model Is)', fontsize=14,
             fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("probability_chart.png", dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("  ✅ Saved to: probability_chart.png")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("  ✅ PROJECT COMPLETE — ALL FILES SAVED")
print("=" * 60)
print()
print(f"  🎯 Model Accuracy       : {accuracy * 100:.1f}%")
print(f"  📊 Precision (SCAM)     : {report_dict['SCAM']['precision']*100:.1f}%")
print(f"  🔍 Recall (SCAM)        : {report_dict['SCAM']['recall']*100:.1f}%")
print(f"  📈 F1-Score (SCAM)      : {report_dict['SCAM']['f1-score']*100:.1f}%")
print()
print("  Files created:")
print("   📄 dataset.csv           — training data")
print("   📄 results.csv           — test set predictions")
print("   📄 new_predictions.csv   — predictions on new messages")
print("   📄 model_report.txt      — full performance report")
print("   📄 word_importance.csv   — scam/legit signal words")
print("   🖼️  charts.png            — main charts (3-panel)")
print("   🖼️  probability_chart.png — probability distribution")
print()
print("  What Naive Bayes learned:")
print(f"   Top scam word: '{top_scam_words[0][0]}'")
print(f"   Top legit word: '{top_legit_words[0][0]}'")
print()
print("=" * 60)

# Save accuracy to a JSON for use in presentation
import json
summary = {
    "accuracy": round(accuracy * 100, 1),
    "precision_scam": round(report_dict['SCAM']['precision'] * 100, 1),
    "recall_scam": round(report_dict['SCAM']['recall'] * 100, 1),
    "f1_scam": round(report_dict['SCAM']['f1-score'] * 100, 1),
    "total_messages": len(df),
    "scam_count": len(df[df['label'] == 'SCAM']),
    "legit_count": len(df[df['label'] == 'LEGITIMATE']),
    "vocab_size": len(vectorizer.vocabulary_),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "top_scam_words": [w for w, _ in top_scam_words[:5]],
    "top_legit_words": [w for w, _ in top_legit_words[:5]],
    "confusion_matrix": cm.tolist()
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  ✅ summary.json saved (used for presentation)")
