# 🎬 Movie Sentiment Analysis

A beginner-friendly NLP project that classifies movie reviews as **Positive** or **Negative** using:
- **Bag of Words** (CountVectorizer) for feature extraction
- **Multinomial Naive Bayes** for classification

---

## 📁 Project Structure

```
movie-sentiment-analysis/
│
├── sentiment_analysis.py   # Main script: train + evaluate + predict
├── predict.py              # Load saved model & run interactive predictions
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to ignore in Git
└── README.md               # This file
```

---

## 🧠 How It Works

### 1. Bag of Words (BoW)
Converts text into numerical features by counting word occurrences.

**Example:**
```
"I loved this movie"  →  [0, 1, 1, 0, 1, ...]
"I hated this movie"  →  [0, 0, 1, 1, 1, ...]
```
Each column = one unique word from the vocabulary.

### 2. Multinomial Naive Bayes
Uses Bayes' theorem to calculate probability of each class:

```
P(Positive | review) ∝ P(review | Positive) × P(Positive)
```

It's called "Naive" because it assumes all words are independent of each other — simple but surprisingly effective!

---

## ⚙️ Setup & Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the main analysis
```bash
python sentiment_analysis.py
```

### Run interactive predictor (after running main script once)
```bash
python predict.py
```

---

## 📊 Sample Output

```
Model Accuracy: 100.00%

Testing on New Reviews:
Review   : This film was absolutely breathtaking
Result   : Positive 😊 (Confidence: 98.4%)

Review   : I hated every minute of this terrible movie
Result   : Negative 😞 (Confidence: 97.1%)
```

---

## 📚 Concepts Used

| Concept | Description |
|---|---|
| Text Preprocessing | Lowercasing, removing punctuation |
| Bag of Words | Word count matrix (CountVectorizer) |
| Multinomial Naive Bayes | Probabilistic classifier for word counts |
| Train-Test Split | 80% train, 20% test |
| Model Persistence | Save/load with pickle |

---

## 🚀 Future Improvements

- Use IMDB dataset (50,000 reviews) instead of sample data
- Add TF-IDF vectorization
- Try Logistic Regression or SVM
- Build a simple Flask web app UI

---

## 👤 Author

Made with ❤️ as a beginner NLP project.
