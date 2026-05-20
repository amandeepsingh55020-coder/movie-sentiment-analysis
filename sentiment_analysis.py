# ============================================================
#   Movie Sentiment Analysis using Bag of Words + Naive Bayes
#   Author: Your Name
#   Description: Classifies movie reviews as Positive or Negative
# ============================================================

import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ─────────────────────────────────────────────
# STEP 1: Sample Dataset (Movie Reviews)
# ─────────────────────────────────────────────
# In real projects, you'd load a CSV like IMDB dataset
# Here we use a small built-in dataset to keep it simple

reviews = [
    "This movie was absolutely wonderful and amazing",
    "Great film, loved every moment of it",
    "Brilliant performances and excellent direction",
    "One of the best movies I have ever seen",
    "Fantastic storyline and superb acting",
    "I loved this movie so much, very entertaining",
    "Outstanding cinematography and great plot",
    "A masterpiece, highly recommend watching it",
    "Wonderful experience, felt every emotion",
    "Perfect movie for a family night, loved it",
    "This movie was terrible and very boring",
    "Worst film I have ever watched, complete waste",
    "Horrible acting and a very bad storyline",
    "Absolutely disgusting and a total disappointment",
    "Very dull and painfully slow, hated it",
    "Pathetic direction and terrible writing",
    "Such a bad movie, completely unwatchable",
    "Awful experience, do not waste your time",
    "Very poor script and bad cinematography",
    "Disaster of a film, deeply disappointing",
]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 1 = Positive
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0 = Negative
]

# ─────────────────────────────────────────────
# STEP 2: Text Preprocessing
# ─────────────────────────────────────────────

def preprocess_text(text):
    """
    Clean and normalize raw text.
    Steps:
      1. Lowercase everything
      2. Remove punctuation
      3. Remove extra whitespace
    """
    text = text.lower()                            # lowercase
    text = re.sub(r'[^\w\s]', '', text)            # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text

# Apply preprocessing
cleaned_reviews = [preprocess_text(r) for r in reviews]

print("=" * 55)
print("  MOVIE SENTIMENT ANALYSIS - Bag of Words + Naive Bayes")
print("=" * 55)
print(f"\n📦 Total Reviews: {len(cleaned_reviews)}")
print(f"   Positive: {labels.count(1)} | Negative: {labels.count(0)}")

# ─────────────────────────────────────────────
# STEP 3: Bag of Words (CountVectorizer)
# ─────────────────────────────────────────────
# CountVectorizer converts text into a matrix of token counts
# Each word becomes a feature (column)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)
y = np.array(labels)

print(f"\n📊 Vocabulary Size (unique words): {len(vectorizer.vocabulary_)}")
print(f"   Feature Matrix Shape: {X.shape}")
print(f"   (rows=reviews, cols=unique words)\n")

# Show a small sample of the vocabulary
sample_vocab = list(vectorizer.vocabulary_.keys())[:10]
print(f"   Sample words in vocabulary: {sample_vocab}")

# ─────────────────────────────────────────────
# STEP 4: Train-Test Split
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔀 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# STEP 5: Train Multinomial Naive Bayes
# ─────────────────────────────────────────────
# MultinomialNB is perfect for word count data (Bag of Words)
# It uses Bayes theorem: P(class | words) ∝ P(words | class) * P(class)

model = MultinomialNB()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# ─────────────────────────────────────────────
# STEP 6: Evaluate the Model
# ─────────────────────────────────────────────

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

print("🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   [[TN={cm[0][0]}  FP={cm[0][1]}]")
print(f"    [FN={cm[1][0]}  TP={cm[1][1]}]]")

# ─────────────────────────────────────────────
# STEP 7: Predict on New Reviews
# ─────────────────────────────────────────────

def predict_sentiment(review_text):
    """
    Predict sentiment of a new movie review.
    Returns: 'Positive 😊' or 'Negative 😞'
    """
    cleaned = preprocess_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = max(probability) * 100
    return sentiment, confidence

# Test on new reviews
print("\n" + "=" * 55)
print("  🎬 TESTING ON NEW REVIEWS")
print("=" * 55)

new_reviews = [
    "This film was absolutely breathtaking and beautiful",
    "I hated every minute of this terrible movie",
    "Not bad but could have been much better",
]

for rev in new_reviews:
    sentiment, confidence = predict_sentiment(rev)
    print(f"\n📝 Review  : {rev}")
    print(f"   Result  : {sentiment} (Confidence: {confidence:.1f}%)")

# ─────────────────────────────────────────────
# STEP 8: Save the Model
# ─────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n\n💾 Model saved as model.pkl and vectorizer.pkl")
print("    You can load these later to skip retraining!")
print("\n✅ Done! Project complete.\n")
