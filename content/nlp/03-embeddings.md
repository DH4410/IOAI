---
title: Word Embeddings
track: nlp
order: 3
estimatedTime: 30
difficulty: intermediate
---

# Word Embeddings

A word embedding is a dense vector of numbers that represents a word. Words with similar meanings have similar vectors. This is how NLP models "understand" language instead of just counting occurrences.

---

## 1. Why Embeddings?

Bag-of-words gives each word a one-hot vector (1 in its position, 0 everywhere else). For a vocabulary of 50,000 words, each word is a 50,000-dimensional vector. This is:
- Huge (50,000 dimensions)
- Sparse (only 1 non-zero value)
- Meaningless ("cat" and "dog" are no more similar than "cat" and "airplane")

Word embeddings fix all three problems by representing words as dense 100-300 dimension vectors where semantically related words are close together.

---

## 2. Word2Vec Intuition

Word2Vec learns embeddings by predicting context. Given the sentence:

```
"The quick brown fox jumps over the lazy dog"
```

The model predicts: "given the word 'fox', what words are likely nearby?" After training on millions of sentences, words that appear in similar contexts end up with similar vectors.

Famous result: `king - man + woman ≈ queen`

This arithmetic works because the model learns geometric relationships between concepts.

---

## 3. Using Pretrained Embeddings

Training embeddings from scratch requires huge amounts of text. Use pretrained ones instead:

```python
# Using gensim to load Word2Vec embeddings
# pip install gensim
import gensim.downloader as api

# Download a small pretrained model
model = api.load('glove-wiki-gigaword-50')  # 50-dimensional GloVe vectors

# Look up a word vector
print(model['king'].shape)    # (50,)

# Find similar words
similar = model.most_similar('python', topn=5)
for word, score in similar:
    print(f'{word}: {score:.3f}')

# Word arithmetic
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result)   # [('queen', 0.849)]
```

**Quick check:** You look up the vector for "happy" and "sad". Would they be close or far apart in embedding space?
> Relatively close! Words that appear in similar contexts (sentences about emotions, moods) get similar vectors, even if they are antonyms. GloVe does not distinguish synonyms from antonyms by proximity alone.

---

## 4. Using Embeddings in PyTorch

In PyTorch, `nn.Embedding` is a lookup table that maps token IDs to vectors:

```python
import torch
import torch.nn as nn

vocab_size = 10000   # number of unique tokens
embed_dim  = 128     # size of each vector

embedding = nn.Embedding(vocab_size, embed_dim)

# Input: batch of token ID sequences
token_ids = torch.LongTensor([[4, 20, 7, 3], [11, 2, 5, 1]])
# Shape: (2, 4) = (batch, sequence_length)

output = embedding(token_ids)
print(output.shape)   # (2, 4, 128) = (batch, seq_len, embed_dim)
```

Every time the model sees token ID 4, it looks up the same 128-dimensional vector. The embedding layer is trained alongside the rest of the model.

---

## 5. Loading Pretrained Embeddings into nn.Embedding

Instead of random initialization, you can start from pretrained GloVe vectors:

```python
import numpy as np

def load_glove(glove_path, vocab, embed_dim=50):
    vectors = np.zeros((len(vocab), embed_dim))
    word2idx = {word: i for i, word in enumerate(vocab)}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                vectors[word2idx[word]] = np.array(parts[1:], dtype=np.float32)

    return torch.FloatTensor(vectors)

# vocab = list of words in your dataset
# glove_vectors = load_glove('glove.6B.50d.txt', vocab)

# Create embedding layer with pretrained weights
embedding = nn.Embedding.from_pretrained(
    glove_vectors,
    freeze=False    # True = keep fixed, False = fine-tune during training
)
```

Starting from pretrained embeddings usually helps, especially for small datasets.

---

## 6. Contextual Embeddings

Word2Vec and GloVe give each word a single fixed vector regardless of context. But "bank" means different things in "river bank" and "bank account."

Modern models (BERT, GPT) produce **contextual embeddings**: the same word gets different vectors depending on the surrounding words. This is much better but requires running the full transformer model.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

text = "The bank was steep near the river"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state: embedding for each token
embeddings = outputs.last_hidden_state
print(embeddings.shape)   # (1, num_tokens, 768)

# Use [CLS] token as sentence embedding
sentence_embedding = embeddings[:, 0, :]
print(sentence_embedding.shape)   # (1, 768)
```

The `[CLS]` token embedding is commonly used as a fixed-size representation of the whole sentence.

---

## Summary

| Method | Dimensions | Context-aware? | When to use |
|---|---|---|---|
| Word2Vec / GloVe | 50-300 | No | Quick baseline, small data |
| FastText | 100-300 | No | Better for rare/misspelled words |
| BERT embeddings | 768 | Yes | When accuracy matters most |

In competition: start with a pretrained BERT and fine-tune on your task. Word2Vec is useful to understand concepts or as a fast baseline, but contextual embeddings from BERT almost always win.
