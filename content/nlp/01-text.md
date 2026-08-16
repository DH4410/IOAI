---
title: Working with Text
track: nlp
order: 1
estimatedTime: 30
difficulty: beginner
---

# Working with Text

NLP (Natural Language Processing) is the branch of AI that deals with human language. IOAI competitions include text tasks: sentiment analysis, text classification, sequence labeling. This lesson covers how raw text becomes something a model can work with.

---

## 1. The Basic Problem

A model cannot read words. It needs numbers. So the first job in any NLP pipeline is to turn text into numbers.

The full pipeline looks like this:

```
Raw text -> Clean -> Tokenize -> Numericalize -> Model
```

Each step transforms the text into something closer to what the model needs.

---

## 2. Cleaning Text

Real text is messy. Cleaning is the first step:

```python
import re

def clean_text(text):
    text = text.lower()                      # lowercase
    text = re.sub(r'<[^>]+>', '', text)      # remove HTML tags
    text = re.sub(r'http\S+', '', text)      # remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text)         # collapse extra spaces
    return text.strip()

sample = "Check out http://example.com! This is <b>GREAT</b> :)"
print(clean_text(sample))
# 'check out  this is great '
```

What to clean depends on your task. For social media text, remove mentions and hashtags. For product reviews, keep the punctuation (a `!` can signal strong sentiment). Do not blindly apply all cleaning steps.

**Quick check:** You are classifying movie reviews. Should you remove all punctuation?
> Probably not. Exclamation marks and question marks carry sentiment. Try keeping them and see if accuracy changes.

---

## 3. Tokenization

**Tokenizing** means splitting text into smaller units (tokens). Usually tokens are words, but they can also be subwords or characters.

```python
# Simplest: split on spaces
text = "the cat sat on the mat"
tokens = text.split()
print(tokens)   # ['the', 'cat', 'sat', 'on', 'the', 'mat']

# Better: use NLTK for proper handling of punctuation
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

text = "I can't believe it's this good!"
tokens = word_tokenize(text)
print(tokens)   # ['I', 'ca', "n't", 'believe', 'it', "'s", 'this', 'good', '!']
```

Notice how `"can't"` became `['ca', "n't"]`. This is because `n't` is a meaningful unit (negation).

---

## 4. Stopwords and Stemming

**Stopwords** are common words that carry little meaning (`the`, `is`, `at`). Removing them reduces noise:

```python
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
filtered = [t for t in tokens if t not in stop_words]
print(filtered)   # ['cat', 'sat', 'mat']
```

**Stemming** reduces words to their root form. `running -> run`, `jumps -> jump`:

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ['running', 'runs', 'jumped', 'jumping', 'studies']
print([stemmer.stem(w) for w in words])
# ['run', 'run', 'jump', 'jump', 'studi']
```

Note `'studi'` is not a real word - stemming is a rough process. For cleaner results use **lemmatization** (finds the dictionary form) but it is slower.

**Quick check:** Should you always remove stopwords?
> No. For sentiment analysis, "not good" vs "good" matters - removing "not" would flip the meaning. For topic detection, stopwords can safely be removed.

---

## 5. Bag of Words

The simplest way to turn text into a vector: count how many times each word appears.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'I love this movie',
    'This movie is bad',
    'I hate this bad movie'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Each document becomes a vector where each position is a word in the vocabulary, and the value is the count.

**Problem:** Rare but important words get the same weight as common words. Solution: TF-IDF.

---

## 6. TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) downweights words that appear in many documents and upweights rare, important words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)   # keep top 1000 words
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```

TF-IDF vectors work well with logistic regression and SVM for text classification, especially with small datasets.

---

## 7. A Complete Text Classification Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
texts = [
    'I love this product', 'Amazing quality', 'Great value',
    'Terrible product', 'Waste of money', 'Very disappointing'
]
labels = [1, 1, 1, 0, 0, 0]   # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)   # same vectorizer, do NOT fit again

# Train and evaluate
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, preds):.2%}')
```

---

## Summary

| Step | Purpose | Tool |
|---|---|---|
| Clean | Remove noise (HTML, URLs) | `re` module |
| Tokenize | Split into words/subwords | `nltk.word_tokenize` |
| Remove stopwords | Drop common words | `nltk.corpus.stopwords` |
| Vectorize | Words to numbers | `CountVectorizer` or `TfidfVectorizer` |

For competition baseline: clean -> TF-IDF -> logistic regression. This often gets you 80%+ accuracy on simple text classification tasks and is your starting point before trying transformers.
