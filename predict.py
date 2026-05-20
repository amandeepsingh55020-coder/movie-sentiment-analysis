
import pickle
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print(" Model loaded successfully!\n")

# Interactive prediction loop
print(" Movie Sentiment Predictor")
print("Type a movie review and press Enter. Type 'quit' to exit.\n")

while True:
    review = input("Enter review: ").strip()
    if review.lower() == "quit":
        print("Goodbye! ")
        break
    if not review:
        continue

    cleaned = preprocess_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = max(proba) * 100

    sentiment = "Positive " if prediction == 1 else "Negative "
    print(f"→ Sentiment: {sentiment}  |  Confidence: {confidence:.1f}%\n")
