---
title: Attention Mechanism
track: nlp
order: 4
estimatedTime: 35
difficulty: advanced
---

# Attention Mechanism

Attention is the key idea behind all modern NLP models (BERT, GPT, T5). It lets the model look at every other word in the sentence when deciding what a word means. This lesson explains how it works.

---

## 1. The Problem Attention Solves

Before attention, NLP used RNNs (Recurrent Neural Networks). An RNN reads words one at a time and maintains a hidden state. For the sentence:

```
"The animal didn't cross the street because it was too tired"
```

What does "it" refer to? The word "animal" is 8 words back. By the time the RNN gets to "it", information about "animal" has been compressed and partially forgotten.

Attention says: when processing "it", look back at ALL words and decide which ones are relevant. The word "animal" gets high attention, "street" gets low attention.

---

## 2. How Attention Works

For each word (query), attention computes a weighted sum of all other words (values), where the weights depend on similarity between the query and each word (key).

```
Query: the word we are currently processing ("it")
Keys:  every word in the sequence
Values: the content of every word

Score = Query · Key (dot product)
Weight = softmax(Score / sqrt(d_k))
Output = sum of (Weight * Value) for all positions
```

The division by `sqrt(d_k)` keeps the dot products from getting too large.

Click each word below to see its attention pattern — which other words it "looks at":

```widget
{
  "type": "attention-heatmap",
  "title": "Attention weights — click a word to see where it attends",
  "words": ["The", "cat", "sat", "on", "the", "mat"],
  "weights": [
    [0.60, 0.12, 0.08, 0.07, 0.08, 0.05],
    [0.10, 0.55, 0.18, 0.07, 0.06, 0.04],
    [0.06, 0.28, 0.42, 0.09, 0.10, 0.05],
    [0.05, 0.06, 0.10, 0.68, 0.07, 0.04],
    [0.12, 0.10, 0.08, 0.12, 0.48, 0.10],
    [0.07, 0.28, 0.12, 0.08, 0.08, 0.37]
  ]
}
```

Notice: "on" (a preposition) attends heavily to itself (0.68) — it's a function word that mostly anchors itself. "sat" attends strongly to "cat" (0.28) — the verb connects to its subject.

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5   # (batch, seq, seq)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# Q, K, V shape: (batch, seq_len, d_k)
Q = torch.randn(1, 5, 64)
K = torch.randn(1, 5, 64)
V = torch.randn(1, 5, 64)

output = attention(Q, K, V)
print(output.shape)   # (1, 5, 64)
```

**Quick check:** After softmax, what do the attention weights sum to across a row?
> 1.0. For each position, the weights across all other positions sum to 1. It is a probability distribution over "how much to attend to each word."

---

## 3. Multi-Head Attention

Instead of one attention operation, the transformer runs several in parallel (called "heads"). Each head can focus on a different type of relationship.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape

        # Project and split into heads
        Q = self.W_q(Q).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, T, self.h, self.d_k).transpose(1, 2)
        # Shape: (batch, heads, seq, d_k)

        # Attention
        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        attn = weights @ V

        # Concatenate heads and project
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(attn)

mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)   # (batch=2, seq=10, d_model=512)
print(mha(x, x, x).shape)     # (2, 10, 512)
```

---

## 4. Why Multi-Head?

With 8 heads and 512 dimensions, each head gets 64 dimensions (`512/8`). Head 1 might learn syntactic relationships (subject-verb), Head 2 might learn coreference (pronouns and their referents), Head 3 might learn semantic similarity.

You can visualize attention weights to see what each head focuses on:

```python
import matplotlib.pyplot as plt

# After running forward pass with access to attention weights
# weights shape: (batch, heads, seq, seq)

def plot_attention(weights, tokens, head=0):
    attn = weights[0, head].detach().cpu().numpy()  # (seq, seq)
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f'Attention Head {head}')
    plt.tight_layout()
    plt.show()
```

---

## 5. Positional Encoding

Attention has no sense of order. "Dog bites man" and "Man bites dog" would look identical to pure attention. Positional encodings add position information.

The original transformer uses sine and cosine functions:

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]   # (seq_len, 1)
    dims = np.arange(0, d_model, 2)                 # even dimensions

    PE[:, 0::2] = np.sin(positions / 10000 ** (dims / d_model))
    PE[:, 1::2] = np.cos(positions / 10000 ** (dims / d_model))

    return torch.FloatTensor(PE).unsqueeze(0)  # (1, seq_len, d_model)

pe = positional_encoding(50, 512)
print(pe.shape)   # (1, 50, 512)
```

This gives each position a unique pattern. The model adds these encodings to the token embeddings before running attention.

**Quick check:** Why does positional encoding matter for attention?
> Without it, the model would produce the same output regardless of word order. Positional encodings let the model know where each token is in the sequence.

---

## 6. Self-Attention vs. Cross-Attention

**Self-attention:** Q, K, V all come from the same sequence. The sequence attends to itself. Used in encoders (BERT).

**Cross-attention:** Q comes from one sequence (decoder), K and V come from another (encoder). The decoder attends to the encoder output. Used in encoder-decoder models (T5, translation).

In most classification tasks you only need self-attention (encoder).

---

## Summary

| Concept | Key point |
|---|---|
| Attention | Compute weighted sum of all positions; weights from query-key similarity |
| Multi-head | Run 8 attention heads in parallel, each learns different relationships |
| Positional encoding | Add position info so the model knows word order |
| Self-attention | Sequence attends to itself (BERT) |
| Cross-attention | One sequence attends to another (translation, T5) |

You don't need to implement attention from scratch in competition. But understanding it helps you use models correctly and debug when things go wrong. The next lesson builds a full transformer using these pieces.
