---
title: Word Embeddings
track: nlp
order: 3
estimatedTime: 50
difficulty: intermediate
---

# Word Embeddings

## The Word2Vec Revolution

In 2013, a team at Google published word2vec — a method that changed NLP forever. The headline result:

```
king - man + woman ≈ queen
```

You could do arithmetic with word meanings. This was not a trick or cherry-picked example. The model learned these relationships from reading billions of words of text, with no explicit supervision about word meanings.

Other relationships the model learned:
```
Paris - France + Germany ≈ Berlin    (capitals)
big - bigger + small ≈ smaller       (comparatives)
walking - walk + swim ≈ swimming     (verb tenses)
uncle - man + woman ≈ aunt           (gender)
doctor - hospital + law ≈ lawyer     (professions + domains)
```

The key insight: words with similar meanings appear in similar **contexts**. If you can learn to predict contexts, you learn word meaning for free.

---

## The Distributional Hypothesis

The distributional hypothesis, from linguistics (Harris, 1954):

> "You shall know a word by the company it keeps." — J.R. Firth, 1957

Words that appear in similar linguistic contexts tend to have similar meanings. Consider the word "dog":

```
"The dog barked loudly."
"The dog chased the cat."
"My dog likes to play fetch."
"She walked her dog in the park."
```

The words appearing around "dog" (barked, cat, play, walked, park) are very different from words around "table" (set, wooden, kitchen, dinner). A model that learns "what kinds of words surround this word" is implicitly learning meaning.

This is why we don't need labeled data to learn good word representations — we just need lots of text.

---

## The Embedding Matrix

A word embedding is a dense vector of real numbers. Instead of a one-hot vector (1 at the word's position, 0 everywhere else), we have a short vector where every dimension carries information.

```python
import numpy as np

# Vocabulary
vocab = ["cat", "dog", "car", "truck", "run", "drive"]
word_to_idx = {w: i for i, w in enumerate(vocab)}

# Embedding matrix: shape (vocab_size, embedding_dim)
# Each row is one word's embedding vector
embedding_dim = 4   # In practice: 50, 100, 300, 768...
np.random.seed(42)

# These would be LEARNED in a real model; we're just initializing randomly
E = np.random.randn(len(vocab), embedding_dim) * 0.1

print("Embedding matrix shape:", E.shape)
print("Embedding of 'cat':", E[word_to_idx['cat']])
print("Embedding of 'dog':", E[word_to_idx['dog']])

# Looking up an embedding
def get_embedding(word, E, word_to_idx):
    if word not in word_to_idx:
        raise ValueError(f"'{word}' not in vocabulary")
    idx = word_to_idx[word]
    return E[idx]

# After training, similar words have similar vectors
# (randomly initialized, so not meaningful yet — just showing the structure)
cat_emb = get_embedding('cat', E, word_to_idx)
dog_emb = get_embedding('dog', E, word_to_idx)
car_emb = get_embedding('car', E, word_to_idx)

print(f"\nRandom embeddings (before training):")
print(f"  cat: {cat_emb}")
print(f"  dog: {dog_emb}")
print(f"  car: {car_emb}")
```

After proper training on a large corpus, the embedding matrix captures semantic structure. We'll set up pretrained-like embeddings for demonstration:

```python
# Simulated "pretrained" embeddings (conceptual — real ones from word2vec/GloVe)
# Dimensions don't have clean interpretations, but here we'll use interpretable toy dims
# [animal-ness, vehicle-ness, size, action-ness]
pretrained = {
    "cat":    np.array([ 0.90,  0.01,  0.20,  0.15]),
    "dog":    np.array([ 0.88,  0.02,  0.35,  0.30]),
    "kitten": np.array([ 0.85,  0.01,  0.05,  0.10]),
    "puppy":  np.array([ 0.82,  0.02,  0.08,  0.20]),
    "car":    np.array([ 0.02,  0.92,  0.50,  0.60]),
    "truck":  np.array([ 0.01,  0.95,  0.85,  0.55]),
    "bus":    np.array([ 0.01,  0.90,  0.80,  0.40]),
    "run":    np.array([ 0.20,  0.05, -0.10,  0.95]),
    "sprint": np.array([ 0.15,  0.05, -0.12,  0.98]),
    "drive":  np.array([ 0.02,  0.80,  0.00,  0.90]),
    "king":   np.array([ 0.05,  0.05,  0.80,  0.20]),
    "queen":  np.array([ 0.05,  0.05,  0.78,  0.18]),
    "man":    np.array([ 0.10,  0.05,  0.50,  0.15]),
    "woman":  np.array([ 0.10,  0.05,  0.48,  0.13]),
}
```

---

## Cosine Similarity

The standard way to measure how similar two word vectors are. It measures the angle between them, not their magnitude.

```python
import numpy as np

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1]:
      1.0 = identical direction (most similar)
      0.0 = orthogonal (unrelated)
     -1.0 = opposite directions (most dissimilar)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# Test with our toy embeddings
pairs = [
    ("cat",   "dog"),      # Both animals — should be similar
    ("cat",   "kitten"),   # Very similar — adult vs young
    ("cat",   "car"),      # Animal vs vehicle — very different
    ("car",   "truck"),    # Both vehicles
    ("run",   "sprint"),   # Both fast movement
    ("run",   "drive"),    # Movement but different type
    ("king",  "queen"),    # Royalty — should be similar
    ("king",  "man"),      # Share gender but different status
]

print("Cosine similarities:")
print(f"{'Pair':30s} | Similarity")
print("-" * 45)
for w1, w2 in pairs:
    v1 = pretrained[w1]
    v2 = pretrained[w2]
    sim = cosine_similarity(v1, v2)
    print(f"  {w1:10s} vs {w2:10s}    | {sim:+.4f}")
```

Expected output (with our toy embeddings):
```
  cat        vs dog           | +0.998  (both animals, similar dims)
  cat        vs kitten        | +0.999
  cat        vs car           | very low (animal vs vehicle)
  car        vs truck         | +0.997
  run        vs sprint        | +0.999
  king       vs queen         | +0.999
```

### Finding the Most Similar Words

```python
def most_similar(word, embeddings_dict, top_k=5):
    """Find the top-k most similar words to the query word."""
    if word not in embeddings_dict:
        raise ValueError(f"'{word}' not in embeddings")
    
    query_vec = embeddings_dict[word]
    
    similarities = []
    for other_word, other_vec in embeddings_dict.items():
        if other_word == word:
            continue
        sim = cosine_similarity(query_vec, other_vec)
        similarities.append((other_word, sim))
    
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]

print("Most similar to 'cat':")
for word, sim in most_similar('cat', pretrained):
    print(f"  {word:15s}: {sim:.4f}")

print("\nMost similar to 'car':")
for word, sim in most_similar('car', pretrained):
    print(f"  {word:15s}: {sim:.4f}")
```

---

## Skip-Gram and CBOW: Intuition

Word2vec has two architectures. You don't need to understand every detail for IOAI, but you need the intuition.

### Skip-Gram: Predict Context from Word

Given a center word, predict the surrounding context words.

```python
# Training data generation for Skip-gram
# Window size = 2: look 2 words left and 2 words right

sentence = "the quick brown fox jumps over the lazy dog".split()
window_size = 2

print("Skip-gram training pairs (center_word → context_word):")
for center_idx, center_word in enumerate(sentence):
    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue
        context_idx = center_idx + offset
        if 0 <= context_idx < len(sentence):
            context_word = sentence[context_idx]
            print(f"  '{center_word}' → '{context_word}'")
```

Output:
```
  'the' → 'quick'
  'the' → 'brown'
  'quick' → 'the'
  'quick' → 'brown'
  'quick' → 'fox'
  'brown' → 'the'
  'brown' → 'quick'
  'brown' → 'fox'
  'brown' → 'jumps'
  ...
```

The model learns embeddings such that:
- `embedding(center) · embedding(context)` is HIGH for real pairs
- `embedding(center) · embedding(random_word)` is LOW for random "noise" pairs

This is called **negative sampling**. You train by maximizing:
```
P(context | center) over real pairs
while minimizing P(random | center) over noise pairs
```

### CBOW: Predict Word from Context

The reverse of Skip-gram: given context words, predict the center word.

```python
# CBOW training data: context_words → center_word

print("CBOW training pairs (context → center):")
for center_idx, center_word in enumerate(sentence):
    context = []
    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue
        context_idx = center_idx + offset
        if 0 <= context_idx < len(sentence):
            context.append(sentence[context_idx])
    
    if context:
        print(f"  {context} → '{center_word}'")
```

**Skip-gram vs CBOW:**
- Skip-gram works better on rare words (each rare word gets many training examples — one per context)
- CBOW is faster to train and works better on frequent words
- In practice, skip-gram with negative sampling is more commonly used

---

## GloVe: Global Co-Occurrence

**GloVe** (Global Vectors, Pennington et al., 2014) takes a different approach. Instead of predicting local context windows, it uses the global co-occurrence statistics of the entire corpus.

### The Co-Occurrence Matrix

```python
import numpy as np

# Toy example: build a co-occurrence matrix
sentences = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying",
    "deep learning requires data",
    "NLP is part of AI",
]

# Build vocabulary
all_words = list(set(word.lower() for s in sentences for word in s.split()))
all_words.sort()
word_to_idx = {w: i for i, w in enumerate(all_words)}
V = len(all_words)

print("Vocabulary:", all_words)

# Build co-occurrence matrix (window size = 1)
co_occur = np.zeros((V, V), dtype=float)
window_size = 2

for sentence in sentences:
    words = sentence.lower().split()
    for i, word in enumerate(words):
        for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
            if i != j:
                co_occur[word_to_idx[word]][word_to_idx[words[j]]] += 1

print("\nCo-occurrence matrix (first 5x5):")
print("      ", "  ".join(f"{w:8s}" for w in all_words[:5]))
for i, w in enumerate(all_words[:5]):
    row = co_occur[i, :5]
    print(f"{w:6s}", "  ".join(f"{v:8.1f}" for v in row))
```

GloVe then factorizes this matrix. The key insight: the ratio of co-occurrence probabilities contains richer information than the raw counts.

For words i, j, k:
```
P(ice | solid) / P(steam | solid)  is very high  (ice and solid co-occur a lot)
P(ice | gas)   / P(steam | gas)    is very low    (steam and gas co-occur a lot)
P(water | solid)/ P(water | gas)   is close to 1 (water relates to both)
```

GloVe's objective:
```
minimize: sum over all (i,j) pairs: (embedding_i · embedding_j + bias_i + bias_j - log(co_occur[i,j]))^2
```

(with a weighting function that downweights very frequent and very rare co-occurrences)

---

## The Analogy Reasoning Operation

The famous king - man + woman = queen works because of the linear structure of embedding space.

```python
import numpy as np

def analogy(a, b, c, embeddings, top_k=5):
    """
    Find d such that: a is to b as c is to d
    i.e., find d ≈ c + (b - a)
    
    Example: man is to king as woman is to d → queen
    """
    # Compute the offset vector
    offset = embeddings[b] - embeddings[a]
    
    # Target: c + offset
    target = embeddings[c] + offset
    
    # Exclude the query words from results
    exclude = {a, b, c}
    
    # Find nearest neighbor
    similarities = []
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        sim = cosine_similarity(target, vec)
        similarities.append((word, sim))
    
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]

# Test with our toy embeddings
print("Analogy: man is to king as woman is to ?")
results = analogy('man', 'king', 'woman', pretrained)
for word, sim in results:
    print(f"  {word:15s}: {sim:.4f}")

print("\nAnalogy: cat is to kitten as dog is to ?")
results = analogy('cat', 'kitten', 'dog', pretrained)
for word, sim in results:
    print(f"  {word:15s}: {sim:.4f}")

print("\nAnalogy: run is to sprint as drive is to ?")
results = analogy('run', 'sprint', 'drive', pretrained)
for word, sim in results:
    print(f"  {word:15s}: {sim:.4f}")
```

### Why Does This Work?

The key is that embedding spaces tend to have **linear structure** that reflects semantic relationships. The vector `king - man` captures "royalty without gender." Adding that to `woman` gives "royalty with female gender" = queen.

```python
# Visualize this geometrically (numerically)
king  = pretrained['king']
man   = pretrained['man']
woman = pretrained['woman']
queen = pretrained['queen']

royalty_offset = king - man
print("Royalty offset (king - man):", royalty_offset)

predicted_queen = woman + royalty_offset
print("Predicted queen:", predicted_queen)
print("Actual queen:   ", queen)
print("Cosine similarity:", cosine_similarity(predicted_queen, queen))
```

---

## Implementing a Toy Embedding Lookup

Let's build a complete, working embedding lookup:

```python
import numpy as np

class EmbeddingLayer:
    """
    A simple embedding lookup table.
    Given integer token IDs, returns their embedding vectors.
    This is the core of any NLP neural network.
    """
    
    def __init__(self, vocab_size, embedding_dim, seed=42):
        np.random.seed(seed)
        # The embedding matrix: shape (vocab_size, embedding_dim)
        # Initialized with small random values — will be learned during training
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids):
        """
        Look up embeddings for a sequence of token IDs.
        
        Args:
            token_ids: list or array of integers (token IDs)
        Returns:
            embeddings: array of shape (seq_len, embedding_dim)
        """
        # This is just a fancy array indexing operation
        return self.W[token_ids]
    
    def load_pretrained(self, word_to_idx, pretrained_dict):
        """
        Initialize embedding matrix from pretrained vectors.
        Words not in pretrained_dict keep their random initialization.
        """
        loaded = 0
        for word, vec in pretrained_dict.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                if len(vec) == self.embedding_dim:
                    self.W[idx] = vec
                    loaded += 1
        print(f"Loaded {loaded}/{len(pretrained_dict)} pretrained embeddings")
    
    def get_similar(self, word, word_to_idx, idx_to_word, top_k=5):
        """Find most similar words to a given word."""
        if word not in word_to_idx:
            return []
        query_idx = word_to_idx[word]
        query_vec = self.W[query_idx]
        
        # Compute cosine similarity to all words
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = self.W / norms
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        similarities = normalized @ query_norm
        similarities[query_idx] = -999  # Exclude self
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx_to_word[i], float(similarities[i])) for i in top_indices]
    
    def __repr__(self):
        return f"EmbeddingLayer(vocab_size={self.vocab_size}, dim={self.embedding_dim})"


# Build vocabulary
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a cat and a dog are friends",
    "cats are small animals",
    "dogs are loyal animals",
    "the quick brown fox",
]

all_words = list(set(w for s in sentences for w in s.split()))
all_words.sort()
word_to_idx = {w: i for i, w in enumerate(all_words)}
idx_to_word = all_words

# Create and use embedding layer
emb_layer = EmbeddingLayer(vocab_size=len(all_words), embedding_dim=8)

# Encode a sentence
sentence = "the cat sat on the mat"
token_ids = [word_to_idx[w] for w in sentence.split() if w in word_to_idx]
embeddings = emb_layer.forward(token_ids)

print(f"Embedding Layer: {emb_layer}")
print(f"\nSentence: '{sentence}'")
print(f"Token IDs: {token_ids}")
print(f"Embedding shape: {embeddings.shape}")  # (6, 8) — 6 tokens, 8 dimensions
print(f"First token embedding: {embeddings[0]}")

# Load pretrained (our toy ones)
# Expand pretrained to match our vocab
emb_layer.load_pretrained(word_to_idx, {
    'cat': np.array([0.9, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]),
    'dog': np.array([0.88, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]),
    'mat': np.array([0.0, 0.0, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
    'log': np.array([0.0, 0.0, 0.4, 0.0, 0.7, 0.0, 0.0, 0.0]),
})

print("\nMost similar to 'cat' (after loading pretrained):")
for w, s in emb_layer.get_similar('cat', word_to_idx, idx_to_word, top_k=5):
    print(f"  {w:10s}: {s:.4f}")
```

---

## Static vs Contextual Embeddings

### Static Embeddings (word2vec, GloVe)

Every word has exactly one vector, regardless of context.

```python
# Problem: polysemy (one word, many meanings)
sentences = [
    "I went to the bank to deposit money.",      # bank = financial
    "She sat on the bank of the river.",          # bank = riverbank
    "The plane banked sharply to the left.",      # bank = turn
]

# In word2vec, 'bank' has ONE vector that is an average/mixture of all its senses
# This vector is not optimal for any specific meaning
bank_vector = pretrained.get('bank', np.zeros(4))  # Not in our toy pretrained

print("Static embeddings: 'bank' has the same vector in all contexts")
print("This is a fundamental limitation of word2vec and GloVe")
```

### Contextual Embeddings (BERT, ELMo)

Every word gets a different vector depending on its surrounding context.

```python
# Conceptual illustration — not actual BERT
# BERT produces different embeddings for 'bank' in each sentence

# Sentence 1: "I went to the bank to deposit money."
# BERT embedding of 'bank': [0.2, 0.8, -0.3, 0.9, ...]  (financial meaning)

# Sentence 2: "She sat on the bank of the river."
# BERT embedding of 'bank': [0.9, 0.1,  0.7, 0.2, ...]  (river meaning)

# These vectors are different! The model uses the surrounding context
# to determine which meaning is intended.

print("Contextual embeddings: 'bank' gets different vectors in different contexts")
print("This is what makes BERT and modern transformers so powerful")
```

We cover how BERT achieves this in Lesson 5.

---

## Visualizing Embeddings with t-SNE

Real embeddings live in 300 or 768 dimensions — impossible to visualize directly. **t-SNE** (t-distributed Stochastic Neighbor Embedding) reduces them to 2D for visualization.

```python
import numpy as np

# t-SNE concept: preserve local structure
# Words that are similar in high-D space end up near each other in 2D

# The actual t-SNE algorithm is complex, but here's the intuition:
# 1. Compute pairwise similarities in high-D space (using Gaussian kernel)
# 2. Create a 2D random initialization
# 3. Compute pairwise similarities in 2D space (using t-distribution)
# 4. Minimize KL divergence between the two similarity distributions
# 5. Points that are close in high-D get attracted together in 2D

# What you see in real visualizations:
# - Animals cluster together: cat, dog, lion, horse, etc.
# - Countries cluster: France, Germany, Italy, Spain, etc.
# - Verb tenses cluster: run/ran, walk/walked, speak/spoke
# - Gender pairs align: king/queen, man/woman, actor/actress

# Using sklearn's t-SNE (reference — for visualization):
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# embeddings_2d = tsne.fit_transform(embedding_matrix)
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Let's do a "text scatter plot" using PCA instead (simple):
def project_to_2d(embeddings_dict):
    """Simple 2D projection using the first two principal components."""
    words = list(embeddings_dict.keys())
    matrix = np.array([embeddings_dict[w] for w in words])
    
    # Center the data
    matrix = matrix - matrix.mean(axis=0)
    
    # PCA: find top-2 principal components
    cov = matrix.T @ matrix / len(words)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by largest eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    top2_components = eigenvectors[:, idx[:2]]
    
    # Project
    projected = matrix @ top2_components
    return words, projected

words, coords_2d = project_to_2d(pretrained)

# Text "scatter plot"
print("2D projection of embeddings (text scatter plot):")
print("Each word's approximate position in PCA space:\n")

# Normalize to 0-20 range for display
x = coords_2d[:, 0]
y = coords_2d[:, 1]
x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10) * 40
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10) * 15

# Create a grid
grid_h, grid_w = 16, 41
grid = [[' '] * grid_w for _ in range(grid_h)]

for i, word in enumerate(words):
    col = min(int(x_norm[i]), grid_w - 1)
    row = min(int(y_norm[i]), grid_h - 1)
    # Place first char of word
    label = word[:4]
    for j, ch in enumerate(label):
        if col + j < grid_w:
            grid[row][col + j] = ch

print("  " + "+" + "-" * grid_w + "+")
for row in grid:
    print("  |" + ''.join(row) + "|")
print("  " + "+" + "-" * grid_w + "+")
print("  (Similar words appear near each other)")
```

---

## Using Embeddings for Text Classification

We can use embeddings to build a document representation by averaging word vectors:

```python
import numpy as np

# Toy pretrained embeddings
embeddings = {
    'good':      np.array([ 0.8, -0.2,  0.1]),
    'great':     np.array([ 0.9, -0.1,  0.2]),
    'excellent': np.array([ 0.95, -0.15, 0.25]),
    'bad':       np.array([-0.8,  0.2, -0.1]),
    'terrible':  np.array([-0.9,  0.3, -0.2]),
    'awful':     np.array([-0.85, 0.25, -0.15]),
    'movie':     np.array([ 0.0,  0.9,  0.1]),
    'film':      np.array([ 0.0,  0.88, 0.12]),
    'watch':     np.array([ 0.1,  0.5,  0.3]),
    'love':      np.array([ 0.7, -0.1,  0.4]),
    'hate':      np.array([-0.7,  0.1, -0.4]),
    'boring':    np.array([-0.6,  0.1, -0.5]),
    'exciting':  np.array([ 0.6, -0.1,  0.5]),
}

def document_to_embedding(text, embeddings, unknown_strategy='zero'):
    """
    Convert a document to a single embedding vector by averaging word vectors.
    
    Args:
        text: string
        embeddings: dict {word: vector}
        unknown_strategy: 'zero' (skip), 'random' (small random vector)
    Returns:
        embedding vector of same dimension as word vectors
    """
    words = text.lower().split()
    vectors = []
    
    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])
        elif unknown_strategy == 'random':
            dim = len(next(iter(embeddings.values())))
            vectors.append(np.random.randn(dim) * 0.01)
        # 'zero' strategy: just skip the unknown word
    
    if not vectors:
        dim = len(next(iter(embeddings.values())))
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

# Example texts
texts = [
    ("This movie is great and exciting", 1),   # positive
    ("Terrible film, really boring and bad", 0),  # negative
    ("I love this excellent film", 1),          # positive
    ("Awful movie I hate watching this", 0),    # negative
    ("The watch was good", None),               # ambiguous (watch = verb or noun?)
]

print("Document embeddings (average of word vectors):")
for text, label in texts:
    emb = document_to_embedding(text, embeddings)
    sentiment_indicator = emb[0]  # Our first dim was +positive/-negative
    pred = "POSITIVE" if sentiment_indicator > 0 else "NEGATIVE"
    true = {1:"POSITIVE", 0:"NEGATIVE", None:"?"}[label]
    print(f"  [{pred}] (true={true}) | '{text[:40]}'")
    print(f"    embedding: [{', '.join(f'{v:+.3f}' for v in emb)}]")
```

**Limitations of averaging:**
- Loses word order: "dog bites man" = "man bites dog"
- Rare words may dominate (if embeddings have large magnitudes)
- Short texts may be dominated by stopwords

Better approaches: use the [CLS] token from BERT (Lesson 5), or apply attention-weighted averaging.

---

## The Embedding Matrix as Learnable Parameters

In a neural network, the embedding matrix is just a parameter matrix that gets updated during training:

```python
import numpy as np

class SimpleEmbeddingClassifier:
    """
    Text classifier that learns embeddings from scratch during training.
    Architecture:
        Embedding lookup (vocab_size × emb_dim)
        Mean pooling (sequence → single vector)
        Linear layer (emb_dim → num_classes)
        Softmax
    """
    
    def __init__(self, vocab_size, emb_dim, num_classes, seed=42):
        np.random.seed(seed)
        # These are BOTH learned during training
        self.E = np.random.randn(vocab_size, emb_dim) * 0.01   # Embedding matrix
        self.W = np.random.randn(emb_dim, num_classes) * 0.01  # Classifier weights
        self.b = np.zeros(num_classes)                          # Bias
    
    def softmax(self, x):
        """Numerically stable softmax."""
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def forward(self, token_ids):
        """
        Forward pass: token_ids → class probabilities
        """
        # Look up embeddings
        word_embs = self.E[token_ids]       # (seq_len, emb_dim)
        
        # Pool: average over sequence length
        doc_emb = word_embs.mean(axis=0)    # (emb_dim,)
        
        # Linear classifier
        logits = doc_emb @ self.W + self.b  # (num_classes,)
        
        # Probabilities
        probs = self.softmax(logits)        # (num_classes,)
        
        return probs, doc_emb
    
    def predict(self, token_ids):
        probs, _ = self.forward(token_ids)
        return np.argmax(probs)


# Quick demo
vocab = ['<PAD>', 'good', 'bad', 'movie', 'great', 'terrible']
word_to_idx = {w: i for i, w in enumerate(vocab)}

classifier = SimpleEmbeddingClassifier(
    vocab_size=len(vocab), 
    emb_dim=4, 
    num_classes=2   # 0=negative, 1=positive
)

test_text = "great movie good"
token_ids = [word_to_idx.get(w, 0) for w in test_text.split()]
probs, doc_emb = classifier.forward(np.array(token_ids))

print(f"Text: '{test_text}'")
print(f"Token IDs: {token_ids}")
print(f"Document embedding: {doc_emb}")
print(f"Class probabilities: negative={probs[0]:.3f}, positive={probs[1]:.3f}")
print(f"Predicted class: {'positive' if np.argmax(probs) == 1 else 'negative'}")
print("\n(Random weights = random predictions — would need training to be useful)")
```

---

## Key Takeaways

1. **Word embeddings** are dense vectors that represent word meaning numerically.

2. **The distributional hypothesis**: words with similar meanings appear in similar contexts. Word2vec exploits this.

3. **Skip-gram** predicts context from a center word; **CBOW** predicts a center word from context. Both learn by maximizing prediction accuracy on billions of words.

4. **GloVe** uses global co-occurrence statistics rather than local context windows.

5. **Cosine similarity** measures the angle between vectors — the standard metric for comparing word embeddings.

6. **Analogy reasoning** (king - man + woman ≈ queen) works because embedding space has linear semantic structure.

7. **Static embeddings** give each word one vector (problem: polysemy). **Contextual embeddings** (BERT) give each word a different vector depending on its context.

8. **Document embeddings** can be built by averaging word vectors — simple but effective baseline.

9. The **embedding matrix** is just a parameter matrix in a neural network and can be fine-tuned for specific tasks.

---

## Exercises

Head to the quiz section to practice:
- Computing cosine similarity from scratch
- Performing analogy operations in numpy
- Implementing document embedding by averaging
