---
title: Text as Data
track: nlp
order: 1
estimatedTime: 40
difficulty: intermediate
---

# Text as Data

## Why Is NLP Hard?

Natural Language Processing (NLP) is the field of teaching computers to understand and work with human language. This sounds straightforward — humans do it effortlessly — but it turns out to be one of the hardest problems in AI.

Here is why.

### Ambiguity Is Everywhere

Consider the sentence: **"I saw the man with the telescope."**

Does this mean:
- I used a telescope to see the man, or
- I saw a man who was holding a telescope?

Both readings are grammatically correct. A human uses context to pick one. A computer has no easy way to do this without understanding the broader situation.

This is called **structural ambiguity** — the grammar of the sentence allows multiple parse trees.

There is also **lexical ambiguity**: the word "bank" can mean a financial institution or the side of a river. Words like "run", "light", "set" have dozens of meanings. English alone has many thousands of such words.

And then there is **referential ambiguity**: "The trophy would not fit in the suitcase because it was too big." What is "too big" — the trophy or the suitcase? Most humans say the trophy, but the grammar alone does not tell you.

### Context Changes Everything

The same words mean completely different things in different contexts.

```
"That movie was sick!"    # In 2010 slang: fantastic, great
"The patient was sick."   # Medical: ill, unwell
"That logic is sick."     # Possibly: flawed, wrong
```

Sarcasm, irony, and humor are all context-dependent. "Oh great, another Monday" does not mean the speaker loves Mondays.

Coreference chains stretch across paragraphs: "Alice told Bob that she would meet him later. She was already running late." Who does "she" refer to? You need to track discourse structure across multiple sentences.

### Language Variation

There is no single "English." There are:

- Regional dialects (British English, American English, Indian English)
- Social dialects (academic writing, Twitter slang, legal documents)
- Code-switching (mixing two languages in one sentence)
- Historical variation (Shakespeare vs. modern English)
- Domain-specific jargon (medical terminology, programming lingo)

A model trained on news articles may completely fail on tweets. A model trained on English may fail on Swahili. Even within one language, style variation is enormous.

### The Long Tail Problem

The English vocabulary has roughly 170,000 words in current use, but the frequency distribution follows a power law (Zipf's law): a small number of words appear constantly, and a huge number of words appear very rarely.

"The" appears in almost every sentence. But "defenestration" (throwing someone out of a window) might appear once in a million words of text. Building a model that handles both common and rare words is genuinely difficult.

New words are created constantly: "selfie", "cryptocurrency", "tweet" (as a verb), "COVID". Any fixed vocabulary becomes outdated quickly.

---

## The Raw Text to Numbers Pipeline

Computers cannot work with raw text directly. They need numbers. The NLP pipeline converts text into numerical representations that machine learning algorithms can process.

Here is the typical sequence:

```
Raw Text
   ↓
Text Cleaning      (lowercase, remove noise, handle punctuation)
   ↓
Tokenization       (split into units: words, subwords, characters)
   ↓
Normalization      (stemming, lemmatization, stopword removal)
   ↓
Vocabulary         (assign integer IDs to tokens)
   ↓
Vectorization      (Bag of Words, TF-IDF, or embeddings)
   ↓
Model Input        (numbers the algorithm can process)
```

Each step is a design choice. The choices you make affect what the model can learn and how well it generalizes.

---

## Tokenization Basics

Tokenization is the process of splitting text into discrete units called **tokens**.

### Character Tokenization

The simplest approach: every character is a token.

```python
text = "Hello, World!"
tokens = list(text)
print(tokens)
# ['H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!']
```

**Advantages:**
- No unknown tokens — any text can be represented
- Small, fixed vocabulary (26 letters + punctuation + digits ≈ 100 tokens)

**Disadvantages:**
- Very long sequences (10-character word = 10 tokens)
- No natural word boundaries — the model must learn what a "word" is from scratch
- Poor at capturing meaning (individual characters have little semantic content)

Character tokenization works well for some tasks like spell-checking or language identification, but poorly for most NLP tasks.

### Word Tokenization

Split on whitespace and punctuation.

```python
import re

text = "Hello, world! This is a simple sentence."

# Simple split on whitespace (bad — keeps punctuation attached)
tokens = text.split()
print(tokens)
# ['Hello,', 'world!', 'This', 'is', 'a', 'simple', 'sentence.']

# Better: use regex to split on whitespace and punctuation
tokens = re.findall(r"\b\w+\b", text)
print(tokens)
# ['Hello', 'world', 'This', 'is', 'a', 'simple', 'sentence']
```

The problem with word tokenization: what do you do with words you have never seen before (Out-Of-Vocabulary, or OOV words)?

```python
# Training vocabulary:
vocab = {"the", "cat", "sat", "on", "mat"}

# At test time:
new_word = "kitty"

if new_word not in vocab:
    token = "<UNK>"  # Replace with unknown token — information is lost!
```

OOV words are a serious problem. A model trained on Wikipedia will encounter countless words in product reviews that it never saw during training.

### Subword Tokenization

The modern solution: split words into meaningful pieces.

```python
# "unbelievable" might become:
# ["un", "##believ", "##able"]   (WordPiece style)
# ["un", "believ", "able"]       (BPE style)

# "running" might become:
# ["run", "##ning"]

# "COVID-19" might become:
# ["CO", "##VI", "##D", "-", "19"]
```

Subword tokenization handles unknown words by breaking them into known pieces. We cover this in detail in Lesson 2.

---

## Bag of Words (BoW)

The simplest way to convert text into a vector: count how many times each vocabulary word appears in the document. Ignore word order.

### Building BoW by Hand

```python
# Our mini corpus (collection of documents)
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
]

# Step 1: Build vocabulary (all unique words)
vocabulary = set()
for doc in documents:
    for word in doc.split():
        vocabulary.add(word)

vocabulary = sorted(vocabulary)  # Sort for consistent ordering
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

print("Vocabulary:", vocabulary)
print("Word-to-index:", word_to_idx)
# Vocabulary: ['cat', 'chased', 'dog', 'log', 'mat', 'on', 'sat', 'the']
# Word-to-index: {'cat': 0, 'chased': 1, 'dog': 2, 'log': 3, 'mat': 4, 'on': 5, 'sat': 6, 'the': 7}

# Step 2: Build the BoW vector for each document
import numpy as np

def document_to_bow(doc, word_to_idx):
    vocab_size = len(word_to_idx)
    vector = np.zeros(vocab_size, dtype=int)
    for word in doc.split():
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector

bow_vectors = np.array([document_to_bow(doc, word_to_idx) for doc in documents])

print("\nBoW Matrix (rows=documents, cols=words):")
print("Words:", vocabulary)
for i, (doc, vec) in enumerate(zip(documents, bow_vectors)):
    print(f"Doc {i}: {vec}  <- '{doc}'")
```

Output:
```
Words: ['cat', 'chased', 'dog', 'log', 'mat', 'on', 'sat', 'the']
Doc 0: [1, 0, 0, 0, 1, 1, 1, 2]  <- 'the cat sat on the mat'
Doc 1: [0, 0, 1, 1, 0, 1, 1, 2]  <- 'the dog sat on the log'
Doc 2: [1, 1, 1, 0, 0, 0, 0, 2]  <- 'the cat chased the dog'
```

Each document is now a vector of length 8 (vocabulary size). Similar documents will have similar vectors.

### BoW with scikit-learn CountVectorizer

In practice, you use scikit-learn's `CountVectorizer`:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "a dog and a cat can be friends",
]

# Create and fit the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# X is a sparse matrix — most entries are zero
print("Shape:", X.shape)               # (4, 12) — 4 docs, 12 unique words
print("Vocabulary:", vectorizer.vocabulary_)
print("Dense matrix:\n", X.toarray())
```

`CountVectorizer` options worth knowing:

```python
# Remove stopwords, limit vocabulary
vectorizer = CountVectorizer(
    stop_words='english',       # Remove common words like "the", "a", "on"
    max_features=1000,          # Keep only top 1000 most frequent words
    min_df=2,                   # Ignore words appearing in fewer than 2 documents
    max_df=0.95,                # Ignore words appearing in more than 95% of documents
    ngram_range=(1, 2),         # Include unigrams AND bigrams
)
```

### What BoW Loses

BoW intentionally throws away word order. This means:

```python
"dog bites man"   # BoW: {dog:1, bites:1, man:1}
"man bites dog"   # BoW: {man:1, bites:1, dog:1}
# Identical BoW vectors — very different meanings!
```

It also loses negation: "I do not like this movie" and "I like this movie" differ only in "not", which might be a stopword that gets removed.

Despite these limitations, BoW works surprisingly well for many tasks like spam detection, topic classification, and document retrieval.

---

## TF-IDF

**Term Frequency-Inverse Document Frequency** is a smarter way to weight word counts.

### The Problem BoW Has

In BoW, common words like "the", "is", "and" get high counts but carry little meaning. Rare but important words like "algorithm" or "vaccine" might have lower counts but tell you much more about what the document is about.

### The TF-IDF Formula

For term `t` in document `d`, with corpus of `N` documents:

```
TF(t, d)  = count(t in d) / total_words(d)       [how often in this doc?]

IDF(t)    = log(N / df(t)) + 1                    [how rare across all docs?]
              where df(t) = number of docs containing t

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Words appearing in EVERY document get a very low IDF (close to 0). Words appearing in only ONE document get a high IDF. The result: common filler words get downweighted, rare meaningful words get upweighted.

### Intuition

Imagine a corpus of 1000 news articles.

- "the" appears in all 1000 docs → IDF = log(1000/1000) + 1 = 0 + 1 = 1.0 (very low)
- "earthquake" appears in 5 docs → IDF = log(1000/5) + 1 = 5.3 + 1 = 6.3 (high!)
- "algorithm" appears in 20 docs → IDF = log(1000/20) + 1 = 3.9 + 1 = 4.9 (medium-high)

### Computing TF-IDF from Scratch

```python
import math
import numpy as np

def compute_tf(doc_tokens):
    """Term frequency for each word in a document."""
    tf = {}
    total_words = len(doc_tokens)
    for word in doc_tokens:
        tf[word] = tf.get(word, 0) + 1
    # Normalize by document length
    for word in tf:
        tf[word] = tf[word] / total_words
    return tf

def compute_idf(corpus_tokens):
    """Inverse document frequency across the corpus."""
    N = len(corpus_tokens)
    idf = {}
    # Count how many documents each word appears in
    all_words = set(word for doc in corpus_tokens for word in doc)
    for word in all_words:
        df = sum(1 for doc in corpus_tokens if word in set(doc))
        idf[word] = math.log(N / df) + 1
    return idf

def compute_tfidf(corpus):
    """Compute TF-IDF matrix for a corpus of documents."""
    # Tokenize
    corpus_tokens = [doc.lower().split() for doc in corpus]
    
    # Compute IDF over whole corpus
    idf = compute_idf(corpus_tokens)
    
    # Vocabulary
    vocab = sorted(idf.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Build TF-IDF matrix
    matrix = np.zeros((len(corpus), len(vocab)))
    for doc_idx, doc_tokens in enumerate(corpus_tokens):
        tf = compute_tf(doc_tokens)
        for word, tf_val in tf.items():
            if word in word_to_idx:
                col = word_to_idx[word]
                matrix[doc_idx, col] = tf_val * idf[word]
    
    return matrix, vocab, idf

# Example
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "dogs are loyal animals",
    "cats are independent animals",
]

matrix, vocab, idf = compute_tfidf(corpus)

print("IDF values (sorted by importance):")
for word, score in sorted(idf.items(), key=lambda x: -x[1])[:10]:
    print(f"  {word:15s}: {score:.3f}")

print("\nTop TF-IDF words in each document:")
for i, doc in enumerate(corpus):
    scores = [(vocab[j], matrix[i, j]) for j in range(len(vocab)) if matrix[i, j] > 0]
    scores.sort(key=lambda x: -x[1])
    print(f"  Doc {i} ('{doc[:30]}...'): {scores[:3]}")
```

### TF-IDF with scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Machine learning is a field of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning",
    "Computer vision is another AI application",
]

tfidf = TfidfVectorizer(smooth_idf=True, norm='l2')
X = tfidf.fit_transform(corpus)

# Show most important words per document
feature_names = tfidf.get_feature_names_out()
for doc_idx, row in enumerate(X):
    scores = zip(feature_names, row.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: -x[1])
    top_words = [(w, f"{s:.3f}") for w, s in sorted_scores if s > 0][:5]
    print(f"Doc {doc_idx}: {top_words}")
```

---

## Text Cleaning

Before vectorizing, you almost always clean the text. Here is each step explained.

### Lowercasing

```python
text = "The Quick Brown Fox JUMPS over the Lazy Dog"
text = text.lower()
print(text)  # "the quick brown fox jumps over the lazy dog"
```

Without lowercasing, "Cat" and "cat" would be treated as different words. This doubles vocabulary size for no benefit in most tasks.

**When NOT to lowercase:** Named entity recognition (Obama vs. obama matters), sentiment analysis where capitalization signals emphasis ("THIS IS AWFUL").

### Removing Punctuation

```python
import re

text = "Hello, World! How are you? I'm fine... thanks."
# Remove all non-alphanumeric characters except spaces
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
print(text)  # "Hello World How are you Im fine thanks"

# Or keep basic punctuation but remove special chars
text = "Hello, World! How are you?"
text = re.sub(r'[^\w\s.,!?]', '', text)
print(text)  # "Hello, World! How are you?"
```

**Note:** Apostrophes in contractions can cause issues. "don't" → "dont" changes meaning vs. "do" + "nt". Consider expanding contractions first.

```python
# Simple contraction expansion
contractions = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we've": "we have",
}

def expand_contractions(text):
    text = text.lower()
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

print(expand_contractions("I don't think it's right."))
# "i do not think it is right."
```

### Removing Stopwords

Stopwords are extremely common words that carry little semantic content.

```python
# Common English stopwords (a small subset)
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'can', 'could', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'it', 'this', 'that',
    'these', 'those', 'and', 'or', 'but', 'if', 'so', 'yet', 'nor',
    'not', 'no', 'very', 'just', 'also', 'about', 'into', 'through',
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she',
    'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
}

def remove_stopwords(tokens, stopwords=STOPWORDS):
    return [token for token in tokens if token not in stopwords]

text = "the cat sat on the mat and the dog sat next to the cat"
tokens = text.split()
filtered = remove_stopwords(tokens)
print("Original:", tokens)
print("Filtered:", filtered)
# Filtered: ['cat', 'sat', 'mat', 'dog', 'sat', 'next', 'cat']
```

**Warning:** Stopword removal can hurt on some tasks. Sentiment analysis relies on words like "not", "never", "very". Question answering relies on "who", "what", "where". Always think about whether removing them makes sense for your task.

### Stemming

Stemming reduces words to their root form by chopping off suffixes using simple rules. The result is not always a real word.

```python
# Implementing a simple (Porter-like) stemmer from scratch
def simple_stem(word):
    """A very basic stemmer — educational only, not production quality."""
    suffixes = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
        ('anci', 'ance'), ('izer', 'ize'), ('ising', 'ise'),
        ('izing', 'ize'), ('ation', 'ate'), ('ator', 'ate'),
        ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
        ('ousness', 'ous'), ('aliti', 'al'), ('ousli', 'ous'),
        ('ness', ''), ('ment', ''), ('ing', ''), ('tion', 'te'),
        ('ies', 'y'), ('es', ''), ('ed', ''), ('er', ''), ('ly', ''),
    ]
    word = word.lower()
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)] + replacement
    return word

# Test it
words = ["running", "runner", "runs", "easily", "fairness", "nationalization"]
for word in words:
    print(f"  {word:20s} → {simple_stem(word)}")
```

Output:
```
  running              → run
  runner               → runn
  runs                 → run
  easily               → easi
  fairness             → fair
  nationalization      → nationalize
```

Notice "runner" → "runn" — not a real word. Stemmers are fast but imprecise.

For production code, use NLTK's Porter Stemmer:
```python
# Reference only — requires nltk
# import nltk
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# stemmer.stem("running")  # → "run"
```

### Lemmatization

Lemmatization converts a word to its canonical form (the lemma) using vocabulary and morphological analysis. The output is always a valid word.

```python
# Manually coded mini-lemmatizer (educational)
LEMMA_MAP = {
    # Verbs
    "running": "run", "runs": "run", "ran": "run", "runner": "runner",
    "eating": "eat", "eats": "eat", "ate": "eat",
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "has": "have", "had": "have", "having": "have",
    # Nouns (plural → singular)
    "cats": "cat", "dogs": "dog", "children": "child",
    "mice": "mouse", "geese": "goose", "feet": "foot",
    "women": "woman", "men": "man", "people": "person",
    # Adjectives
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "bigger": "big", "biggest": "big",
}

def lemmatize(word):
    return LEMMA_MAP.get(word.lower(), word.lower())

words = ["running", "dogs", "better", "ate", "children", "was"]
for word in words:
    print(f"  {word:12s} → {lemmatize(word)}")
```

Output:
```
  running      → run
  dogs         → dog
  better       → good
  ate          → eat
  children     → child
  was          → be
```

**Stemming vs. Lemmatization:**
- Stemming is faster but produces non-words ("argue" → "argu")
- Lemmatization is slower but always produces valid words
- For most ML tasks with large data, stemming is fine
- For tasks where interpretability matters, use lemmatization

---

## Building a Vocabulary

The vocabulary is your model's "dictionary" — the fixed set of tokens it knows about.

```python
from collections import Counter

def build_vocabulary(corpus, max_size=None, min_freq=1):
    """
    Build a word vocabulary from a corpus.
    
    Args:
        corpus: list of strings (documents)
        max_size: maximum vocabulary size (None = unlimited)
        min_freq: minimum frequency to include a word
    
    Returns:
        word_to_idx: dict mapping word to integer ID
        idx_to_word: list mapping integer ID to word
    """
    # Count all words
    counter = Counter()
    for doc in corpus:
        tokens = doc.lower().split()
        counter.update(tokens)
    
    # Filter by minimum frequency
    valid_words = [(word, count) for word, count in counter.items() 
                   if count >= min_freq]
    
    # Sort by frequency (most common first)
    valid_words.sort(key=lambda x: -x[1])
    
    # Truncate to max_size
    if max_size is not None:
        valid_words = valid_words[:max_size - 3]  # Reserve space for special tokens
    
    # Add special tokens at fixed positions
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    all_tokens = special_tokens + [word for word, _ in valid_words]
    
    # Build mappings
    word_to_idx = {word: idx for idx, word in enumerate(all_tokens)}
    idx_to_word = all_tokens
    
    return word_to_idx, idx_to_word

# Example usage
corpus = [
    "the cat sat on the mat and the cat was happy",
    "the dog ran in the park and barked loudly",
    "a happy cat and a happy dog are good friends",
    "the park was full of happy children playing",
]

w2i, i2w = build_vocabulary(corpus, max_size=20, min_freq=1)

print("Vocabulary (first 15):")
for i, word in enumerate(i2w[:15]):
    print(f"  {i}: {word}")

print(f"\nVocabulary size: {len(w2i)}")

# Encode a sentence
def encode(text, word_to_idx):
    unk_idx = word_to_idx['<UNK>']
    return [word_to_idx.get(word.lower(), unk_idx) for word in text.split()]

sentence = "the happy cat chased a butterfly"
encoded = encode(sentence, w2i)
print(f"\nEncoded '{sentence}':")
print(f"  {encoded}")
print(f"  decoded: {[i2w[i] if i < len(i2w) else '?' for i in encoded]}")
```

---

## Sparse vs Dense Representations

There are two families of text representations.

### Sparse Representations (BoW, TF-IDF)

Each document is a vector of length V (vocabulary size). Most entries are zero.

```python
# A vocabulary of 50,000 words
# A document with 100 unique words
# → 49,900 zeros, 100 non-zeros = 0.2% dense

import numpy as np
from scipy.sparse import csr_matrix

# Sparse representation (memory-efficient)
V = 50000
doc_length = 100
sparse_vec = csr_matrix(np.zeros((1, V)))  # Uses very little memory

# Dense representation would be:
dense_vec = np.zeros(V)  # 50,000 floats × 4 bytes = 200KB per document
# For 1 million documents: 200GB!
```

**Properties of sparse representations:**
- Easily interpretable (which words matter)
- Memory-efficient with sparse matrices
- No training needed — just count
- Cannot capture semantic similarity (synonyms have orthogonal vectors)
- Fixed vocabulary — new words not in training are unknown

### Dense Representations (Embeddings)

Each word is a short, dense vector (e.g., 300 dimensions). Every dimension has a non-zero value.

```python
# Word2Vec-style dense embeddings (conceptual)
import numpy as np

# Pretend these are learned
word_vectors = {
    "cat":    np.array([0.23, -0.14, 0.87, 0.05, -0.32]),  # 5D example
    "feline": np.array([0.25, -0.13, 0.85, 0.07, -0.30]),  # Similar to cat!
    "dog":    np.array([0.21, -0.10, 0.82, 0.04, -0.28]),  # Also similar
    "table":  np.array([-0.50, 0.80, -0.20, 0.60, 0.10]),  # Very different
}

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Cat and feline are similar (they mean the same thing)
print(f"cat vs feline: {cosine_similarity(word_vectors['cat'], word_vectors['feline']):.3f}")
print(f"cat vs dog:    {cosine_similarity(word_vectors['cat'], word_vectors['dog']):.3f}")
print(f"cat vs table:  {cosine_similarity(word_vectors['cat'], word_vectors['table']):.3f}")
```

**Properties of dense representations:**
- Capture semantic similarity (synonyms have similar vectors)
- Require training (or loading pretrained weights)
- Fixed size regardless of vocabulary
- Less interpretable
- Much smaller: 300 dimensions vs 50,000 for a vocabulary

We cover word embeddings in depth in Lesson 3.

---

## Text Classification with BoW + Logistic Regression

Putting it all together: a complete text classification pipeline.

```python
import numpy as np
import re
from collections import Counter

# =============================================
# DATASET: Simple Sentiment Analysis
# =============================================

train_texts = [
    "I love this movie it was amazing",
    "Great film highly recommended",
    "Wonderful acting and beautiful story",
    "Best movie I have seen in years",
    "Fantastic performances and great plot",
    "Terrible waste of time avoid this",
    "Boring and predictable do not watch",
    "Awful acting and weak story",
    "Worst film ever made",
    "Disappointing and poorly written",
]

train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

test_texts = [
    "I enjoyed this wonderful film",
    "A complete disaster and waste of money",
    "Pretty good movie with solid acting",
]

test_labels = [1, 0, 1]

# =============================================
# STEP 1: TEXT CLEANING
# =============================================

STOPWORDS = {'i', 'a', 'and', 'the', 'in', 'it', 'this', 'to', 'of', 'my'}

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

# =============================================
# STEP 2: BUILD VOCABULARY
# =============================================

all_train_tokens = [clean(t) for t in train_texts]
counter = Counter(token for tokens in all_train_tokens for token in tokens)

vocab = ['<UNK>'] + [word for word, count in counter.most_common()]
word_to_idx = {word: i for i, word in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary size: {V}")

# =============================================
# STEP 3: BOW VECTORIZATION
# =============================================

def text_to_bow(text, word_to_idx):
    tokens = clean(text)
    v = np.zeros(len(word_to_idx))
    for token in tokens:
        idx = word_to_idx.get(token, 0)  # 0 = <UNK>
        v[idx] += 1
    return v

X_train = np.array([text_to_bow(t, word_to_idx) for t in train_texts])
X_test  = np.array([text_to_bow(t, word_to_idx) for t in test_texts])
y_train = np.array(train_labels)
y_test  = np.array(test_labels)

print(f"X_train shape: {X_train.shape}")  # (10, vocab_size)

# =============================================
# STEP 4: LOGISTIC REGRESSION FROM SCRATCH
# =============================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_train(X, y, lr=0.1, epochs=200):
    """Train logistic regression with gradient descent."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for epoch in range(epochs):
        # Forward pass
        logits = X @ weights + bias
        predictions = sigmoid(logits)
        
        # Compute cross-entropy loss
        eps = 1e-10  # Avoid log(0)
        loss = -np.mean(y * np.log(predictions + eps) + 
                        (1 - y) * np.log(1 - predictions + eps))
        
        # Backward pass (gradients)
        error = predictions - y
        grad_w = X.T @ error / n_samples
        grad_b = np.mean(error)
        
        # Update parameters
        weights -= lr * grad_w
        bias    -= lr * grad_b
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
    
    return weights, bias

def logistic_regression_predict(X, weights, bias, threshold=0.5):
    logits = X @ weights + bias
    probs = sigmoid(logits)
    return (probs >= threshold).astype(int), probs

print("\nTraining logistic regression:")
weights, bias = logistic_regression_train(X_train, y_train, lr=0.01, epochs=200)

print("\nPredictions on test set:")
preds, probs = logistic_regression_predict(X_test, weights, bias)
for text, pred, prob, true in zip(test_texts, preds, probs, y_test):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    true_label = "POSITIVE" if true == 1 else "NEGATIVE"
    correct = "✓" if pred == true else "✗"
    print(f"  {correct} {label} (p={prob:.2f}) | true={true_label} | '{text[:40]}'")

accuracy = np.mean(preds == y_test)
print(f"\nAccuracy: {accuracy:.1%}")

# =============================================
# STEP 5: INSPECT WHAT THE MODEL LEARNED
# =============================================

# The most positive and negative words
top_positive = sorted(zip(weights, vocab), reverse=True)[:5]
top_negative = sorted(zip(weights, vocab))[:5]

print("\nMost positive words:")
for w, word in top_positive:
    print(f"  {word:15s}: {w:+.3f}")

print("\nMost negative words:")
for w, word in top_negative:
    print(f"  {word:15s}: {w:+.3f}")
```

This complete example shows the full pipeline from raw text → cleaned tokens → BoW vectors → trained classifier → inspection.

---

## Key Takeaways

1. **NLP is hard** because of ambiguity (lexical, structural, referential), context-dependence, and massive language variation.

2. **The pipeline** is: clean → tokenize → normalize → vectorize → model.

3. **Tokenization** has three levels: character (tiny vocab, long sequences), word (natural but OOV problem), subword (best of both worlds).

4. **Bag of Words** converts text to word count vectors. Simple, fast, interpretable — but loses word order.

5. **TF-IDF** improves BoW by downweighting common words (low IDF) and upweighting rare but distinctive words (high IDF).

6. **Text cleaning** includes lowercasing, punctuation removal, stopword removal, and normalization (stemming/lemmatization).

7. **Sparse vs dense**: BoW/TF-IDF vectors are high-dimensional and sparse; word embeddings are low-dimensional and dense. Sparse is interpretable; dense captures semantics.

8. **BoW + logistic regression** is a strong baseline for text classification and often outperforms complex models on small datasets.

---

## Exercises

Head to the quiz section to practice:
- Building a BoW matrix from scratch
- Computing TF-IDF by hand
- Preprocessing a dataset end-to-end
