---
title: The Attention Mechanism
track: nlp
order: 4
estimatedTime: 55
difficulty: advanced
---

# The Attention Mechanism

Attention is the single most important idea in modern NLP. It is the core of every large language model you have heard of — GPT, BERT, Claude, Llama. Before attention existed, sequence models hit a hard ceiling: the longer the input, the more they forgot. Attention broke that ceiling. This lesson explains the problem attention solves, how it works mathematically, and why the scaled dot-product form used in transformers became the standard.

---

## Section 1: The Problem — RNN Bottleneck

### Sequence-to-Sequence Models

Before attention, the dominant approach to tasks like machine translation was the **encoder-decoder** architecture built on Recurrent Neural Networks (RNNs) or Long Short-Term Memory networks (LSTMs).

The idea is conceptually clean:
1. An **encoder** reads the entire source sentence, one token at a time, and produces a sequence of hidden states $h_1, h_2, \ldots, h_T$.
2. After processing the last token, the final hidden state $h_T$ is handed to the **decoder** as a single fixed-size vector — the "context vector" $c$.
3. The **decoder** generates the output sequence one token at a time, conditioned on $c$.

```python
import numpy as np

# Simplified RNN encoder (conceptual)
def rnn_step(h_prev, x_t, W_h, W_x, b):
    """One RNN step: h_t = tanh(W_h @ h_prev + W_x @ x_t + b)"""
    return np.tanh(W_h @ h_prev + W_x @ x_t + b)

# After processing all T tokens, h_T becomes the context vector c.
# The decoder is initialized with c and generates the output.
```

### The Bottleneck Problem

The critical flaw: **all information about the source sentence must be compressed into a single vector** $c = h_T$.

Imagine translating "The cat sat on the mat because it was tired" into French. The word "it" at the end refers to "the cat" at the beginning. But by the time the encoder processes "tired", it has overwritten its hidden state many times. The information about "the cat" may be severely diluted.

This creates a hard empirical limit. Studies from 2014 showed that LSTM-based translators performed well on sentences up to ~20 words but degraded sharply on longer sequences. The BLEU score (a translation quality metric) dropped like a cliff around 30+ tokens.

```
Performance
  |  ****
  | *    *
  |*      ***
  |          ****
  |              ********
  +---------------------------------> Sentence Length
     5   10   15   20   25   30+
```

The model simply cannot store 30 words of meaning in a single 256-dimensional or 512-dimensional vector.

### What We Actually Want

When a human translator works on a long sentence, they don't memorize the entire source and then write the translation. They **look back**. When translating the verb, they glance at the subject. When translating a pronoun, they find its antecedent. The attention mechanism gives neural networks this same ability.

**Key takeaway:** The RNN bottleneck forces all source information into a fixed-size vector, causing information loss on long sequences. Attention replaces this single context vector with a dynamic, weighted combination of all encoder states.

---

## Section 2: The Attention Intuition

### Looking Back at All Encoder States

Instead of discarding all intermediate encoder hidden states and using only $h_T$, the attention mechanism keeps them all: $h_1, h_2, \ldots, h_T$.

When the decoder is generating the $i$-th output token, it computes a **weighted sum** over all encoder states:

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

The weights $\alpha_{ij}$ are called **attention weights**. They satisfy:

$$\sum_{j=1}^{T} \alpha_{ij} = 1, \quad \alpha_{ij} \geq 0$$

So $c_i$ is a probability-weighted mixture of encoder states. If $\alpha_{i3} = 0.8$, it means that when generating output token $i$, the decoder is paying 80% of its attention to the 3rd input token.

### The Intuition with a Concrete Example

Suppose we are translating "The black cat" → "Le chat noir" (French). French adjectives come after nouns, so the word order flips.

- When generating "Le" (the), attention should focus on "The"
- When generating "chat" (cat), attention should focus on "cat"
- When generating "noir" (black), attention should focus on "black"

The attention weights for each output token would look approximately like:

```
Output token | The  | black | cat
-------------|------|-------|-----
Le           | 0.85 | 0.05  | 0.10
chat         | 0.05 | 0.10  | 0.85
noir         | 0.05 | 0.85  | 0.10
```

The model learns these weights from data — no one hand-coded the word alignments. This is the magic of attention: **alignment emerges from training**.

### Attention as a Soft Lookup

A useful mental model: attention is like a **soft dictionary lookup**.

In a hard dictionary, you query a key and get exactly one value back. In attention:
- You have a **query** (what you are looking for)
- You have a set of **keys** (descriptors of what is stored)
- You have a set of **values** (the actual stored content)
- You compute similarity between your query and all keys, normalize to get weights, and return a weighted average of values

This query-key-value (QKV) framing becomes the standard vocabulary in the transformer.

**Key takeaway:** Attention computes a dynamic, learned weighted average of all encoder hidden states. The weights (called attention weights) indicate which parts of the input are most relevant for generating each output token.

---

## Section 3: Bahdanau (Additive) Attention

The first influential attention mechanism was proposed by Bahdanau et al. in 2014 ("Neural Machine Translation by Jointly Learning to Align and Translate"). Let us walk through it carefully.

### Setup

- Encoder hidden states: $h_1, \ldots, h_T \in \mathbb{R}^{d_h}$
- Decoder hidden state at step $i$: $s_{i-1} \in \mathbb{R}^{d_s}$

### Step 1: Compute Alignment Scores

For each encoder state $h_j$, compute a scalar score measuring how well it matches the current decoder state:

$$e_{ij} = \text{score}(s_{i-1}, h_j)$$

Bahdanau used a small feedforward network for this:

$$e_{ij} = v^T \tanh(W_s \cdot s_{i-1} + W_h \cdot h_j)$$

where $W_s \in \mathbb{R}^{d_a \times d_s}$, $W_h \in \mathbb{R}^{d_a \times d_h}$, and $v \in \mathbb{R}^{d_a}$ are learned parameters. $d_a$ is the "attention dimension," a hyperparameter (often 128 or 256).

This is sometimes called **additive attention** because it adds the projections of $s$ and $h$ before passing through tanh.

### Step 2: Normalize with Softmax

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

The softmax ensures weights are non-negative and sum to 1.

### Step 3: Compute Context Vector

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

### Step 4: Generate Output

The decoder uses $c_i$ concatenated with $s_{i-1}$ to produce the next hidden state and output token.

```python
import numpy as np

def bahdanau_attention(s_prev, encoder_states, W_s, W_h, v):
    """
    s_prev: (d_s,) decoder hidden state
    encoder_states: (T, d_h) all encoder hidden states
    W_s: (d_a, d_s)
    W_h: (d_a, d_h)
    v: (d_a,)
    Returns: context vector (d_h,) and attention weights (T,)
    """
    T = encoder_states.shape[0]

    # Project decoder state: (d_a,)
    proj_s = W_s @ s_prev  # broadcast over T later

    # Project each encoder state: (T, d_a)
    proj_h = encoder_states @ W_h.T

    # Add and pass through tanh: (T, d_a)
    energy = np.tanh(proj_s[np.newaxis, :] + proj_h)

    # Score with v: (T,)
    scores = energy @ v

    # Softmax
    scores_exp = np.exp(scores - scores.max())  # numerically stable
    alpha = scores_exp / scores_exp.sum()

    # Context vector: (d_h,)
    context = alpha @ encoder_states

    return context, alpha
```

### What the Parameters Learn

The parameters $W_s$, $W_h$, and $v$ are learned jointly with the encoder and decoder via backpropagation through time. No supervision on alignment is needed — the model discovers on its own which source words matter for each output word.

**Key takeaway:** Bahdanau attention uses a small neural network to score all (decoder state, encoder state) pairs, normalizes the scores with softmax, and computes a weighted sum of encoder states. The score network is trained end-to-end with the seq2seq model.

---

## Section 4: Self-Attention

Bahdanau attention connects a **decoder** query to **encoder** keys and values. Self-attention is a generalization: the queries, keys, AND values all come from the **same sequence**.

### Why Self-Attention?

Consider the sentence "The animal didn't cross the street because it was too tired."

What does "it" refer to? To a human, obviously "the animal." But for a model processing the sentence, resolving this coreference requires connecting "it" (position 8) to "animal" (position 2). Self-attention can do exactly this: when computing the representation of "it," attend strongly to "animal."

Self-attention also handles:
- Subject-verb agreement ("The keys that were on the table **are** missing")
- Modifier attachment ("I saw the man with the telescope")
- Long-range dependencies in general

### Self-Attention Formulation

Given a sequence of $n$ token embeddings $X \in \mathbb{R}^{n \times d}$, we compute:
- **Queries**: $Q = X W^Q$
- **Keys**: $K = X W^K$
- **Values**: $V = X W^V$

where $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$ are learned projection matrices.

The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Notice that $Q = K = V$ are all derived from the **same** input $X$ (via different linear projections). That is what makes it "self" attention.

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    """
    X: (n, d) input sequence
    W_Q, W_K: (d, d_k) projection matrices
    W_V: (d, d_v) value projection matrix
    Returns: (n, d_v) output
    """
    Q = X @ W_Q   # (n, d_k)
    K = X @ W_K   # (n, d_k)
    V = X @ W_V   # (n, d_v)

    d_k = Q.shape[-1]

    # Raw scores: (n, n)
    scores = Q @ K.T / np.sqrt(d_k)

    # Softmax over keys dimension
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    alpha = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Weighted sum of values: (n, d_v)
    output = alpha @ V

    return output, alpha
```

The output at position $i$ is a weighted mixture of all value vectors, where the weight between position $i$ and position $j$ is determined by the dot-product similarity of query $i$ and key $j$.

**Key takeaway:** Self-attention lets each position in a sequence attend to every other position, with learned weights. Queries, keys, and values all come from the same sequence via separate linear projections. This enables direct, non-sequential modeling of long-range dependencies.

---

## Section 5: Scaled Dot-Product Attention

The version of attention used in transformers is called **scaled dot-product attention**. It refines the basic QKV formulation with one crucial change: scaling by $\frac{1}{\sqrt{d_k}}$.

### Why Scale?

The dot product $Q K^T$ computes, for each query-key pair, the sum of $d_k$ element-wise products. If the vectors are initialized with roughly unit variance, the dot product has variance approximately $d_k$. For large $d_k$ (e.g., 64 or 128), these dot products can become very large.

Large dot products push the softmax into its **saturation region** — one entry gets weight close to 1 and all others near 0. In the saturation region, gradients are almost zero, and training stalls. Dividing by $\sqrt{d_k}$ brings the variance back to $\approx 1$ and keeps the softmax in a healthy regime.

### The Full Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Let $n$ be the sequence length and $d_k$ the key dimension:
- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{n \times d_k}$
- $V \in \mathbb{R}^{n \times d_v}$
- $Q K^T \in \mathbb{R}^{n \times n}$ — the attention score matrix
- After softmax: attention weight matrix $A \in \mathbb{R}^{n \times n}$
- Output: $A V \in \mathbb{R}^{n \times d_v}$

### Numerical Stability in Softmax

A common implementation trick: subtract the row maximum before exponentiating.

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

This prevents overflow without changing the output (since the constant cancels).

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., n, d_k)
    K: (..., m, d_k)
    V: (..., m, d_v)
    mask: optional boolean mask; True positions are masked (set to -inf)
    Returns: (..., n, d_v) output, (..., n, m) attention weights
    """
    d_k = Q.shape[-1]

    # (n, m) score matrix
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Numerically stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    alpha = np.exp(scores)
    alpha /= alpha.sum(axis=-1, keepdims=True)

    output = alpha @ V
    return output, alpha

# Example
np.random.seed(42)
n, d_k, d_v = 5, 8, 8
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

out, weights = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)          # (5, 8)
print("Weight matrix shape:", weights.shape)  # (5, 5)
print("Row sums (should be 1):", weights.sum(axis=-1))
```

### Masked Attention

In some settings we want to prevent certain positions from attending to others:
- **Causal masking** (used in GPT-style models): position $i$ can only attend to positions $\leq i$. This prevents the model from "seeing the future" during training.
- **Padding masking**: prevent attention to padding tokens.

The mask sets those scores to $-\infty$ before softmax, so their weights become exactly 0.

```python
def causal_mask(n):
    """Returns True where attention should be blocked (upper triangle)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)

n = 4
mask = causal_mask(n)
print(mask)
# [[False  True  True  True]
#  [False False  True  True]
#  [False False False  True]
#  [False False False False]]
```

**Key takeaway:** Scaled dot-product attention divides scores by $\sqrt{d_k}$ to prevent softmax saturation. A masking mechanism can block attention to certain positions, enabling causal (autoregressive) generation.

---

## Section 6: Multi-Head Attention

A single attention head computes one weighted mixture. But a sentence has many simultaneous relationships — syntactic, semantic, coreference, local n-gram patterns — and a single head may not capture all of them at once. Multi-head attention runs several attention computations in parallel, each potentially specializing in different relationship types.

### The Idea

Instead of one attention function with $d_k$-dimensional keys, use $h$ "heads," each operating in a lower-dimensional subspace $d_k / h$.

For each head $i = 1, \ldots, h$:
$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

where:
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $d_k = d_v = d_{\text{model}} / h$

Then concatenate all heads and project back:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

In the original transformer paper, $d_{\text{model}} = 512$, $h = 8$, so each head operates in $d_k = 64$ dimensions.

### Why Multiple Heads Work

Empirically, different heads in trained transformers attend to qualitatively different things:
- Some heads track syntactic dependencies (subject-verb)
- Some heads track local context (next/previous word)
- Some heads handle coreference
- Some heads seem to identify sentence boundaries

No one hand-designed these specializations — they emerged from training. By using multiple independent attention functions and concatenating, the model learns a richer representation than any single head could.

### Parameter Count

Each head has 3 projection matrices. For $h$ heads:
- Query projections: $h \times d_{\text{model}} \times d_k = d_{\text{model}}^2$ total
- Key projections: $d_{\text{model}}^2$
- Value projections: $d_{\text{model}}^2$
- Output projection: $h \times d_v \times d_{\text{model}} = d_{\text{model}}^2$

Total: $4 d_{\text{model}}^2$ parameters for one multi-head attention layer.

```python
import numpy as np

def multi_head_attention(Q, K, V, W_Q_heads, W_K_heads, W_V_heads, W_O):
    """
    Q, K, V: (n, d_model)
    W_Q_heads: list of h matrices, each (d_model, d_k)
    W_K_heads: list of h matrices, each (d_model, d_k)
    W_V_heads: list of h matrices, each (d_model, d_v)
    W_O: (h * d_v, d_model)
    Returns: (n, d_model)
    """
    heads = []
    for W_Q, W_K, W_V in zip(W_Q_heads, W_K_heads, W_V_heads):
        q = Q @ W_Q   # (n, d_k)
        k = K @ W_K   # (n, d_k)
        v = V @ W_V   # (n, d_v)
        head, _ = scaled_dot_product_attention(q, k, v)  # (n, d_v)
        heads.append(head)

    # Concatenate: (n, h * d_v)
    concat = np.concatenate(heads, axis=-1)

    # Project back: (n, d_model)
    return concat @ W_O

# Toy example with h=2 heads, d_model=8, d_k=d_v=4
h, d_model, d_k = 2, 8, 4
n = 5
np.random.seed(0)
X = np.random.randn(n, d_model)

W_Q_heads = [np.random.randn(d_model, d_k) / np.sqrt(d_model) for _ in range(h)]
W_K_heads = [np.random.randn(d_model, d_k) / np.sqrt(d_model) for _ in range(h)]
W_V_heads = [np.random.randn(d_model, d_k) / np.sqrt(d_model) for _ in range(h)]
W_O = np.random.randn(h * d_k, d_model) / np.sqrt(h * d_k)

out = multi_head_attention(X, X, X, W_Q_heads, W_K_heads, W_V_heads, W_O)
print("Multi-head attention output shape:", out.shape)  # (5, 8)
```

**Key takeaway:** Multi-head attention runs $h$ independent scaled dot-product attention computations in parallel, each in a lower-dimensional subspace. The outputs are concatenated and projected. Different heads can specialize in different types of relationships.

---

## Section 7: Attention in Practice — Visualizing What the Model Sees

### Attention Maps

For any given input sequence, we can extract and visualize the attention weight matrix $A \in \mathbb{R}^{n \times n}$ for any head in any layer. Each cell $(i, j)$ shows how much position $i$ attended to position $j$.

These visualizations are called **attention maps** (or attention heatmaps). They offer interpretability — a window into what the model "thinks is related."

**Famous examples from research:**

1. **Co-reference resolution heads**: When the input is "The animal didn't cross the street because it was too tired," one head in BERT's middle layers strongly attends from "it" to "animal."

2. **Positional heads**: Some heads show a diagonal pattern — each token mostly attends to its immediate neighbors. These act like local smoothing filters.

3. **Sentence boundary heads**: Some heads attend from every token in one sentence to the [CLS] token or [SEP] token.

```python
import numpy as np

def visualize_attention_text(tokens, weights, head=0):
    """
    Print attention weights as a simple text grid.
    tokens: list of strings, length n
    weights: (n, n) attention weight matrix
    """
    n = len(tokens)
    col_w = max(len(t) for t in tokens) + 1

    # Header
    header = " " * col_w + "".join(f"{t:>{col_w}}" for t in tokens)
    print(header)

    for i, row_tok in enumerate(tokens):
        row = f"{row_tok:<{col_w}}"
        for j in range(n):
            w = weights[i, j]
            # Use block characters for simple visualization
            if w > 0.5:
                block = "██"
            elif w > 0.2:
                block = "▓▓"
            elif w > 0.05:
                block = "░░"
            else:
                block = "  "
            row += f"{block:>{col_w}}"
        print(row)

tokens = ["The", "cat", "sat", "on", "it"]
np.random.seed(7)
# Synthetic weights — in reality these come from a trained model
weights = np.random.dirichlet(alpha=[2, 5, 1, 1, 1], size=5)
weights[4] = [0.1, 0.7, 0.1, 0.05, 0.05]  # "it" attends to "cat"
visualize_attention_text(tokens, weights)
```

### Limitations of Attention as Explanation

A word of caution: attention weights are **not** a perfect explanation of what the model uses to make decisions.

- High attention does not always mean causal importance.
- Different heads can cancel each other out.
- The value projections matter too — a highly-attended token might contribute little to the output if its value projection is small.

Research papers (e.g., "Attention is not Explanation," Jain & Wallace 2019) showed that you can sometimes replace the real attention weights with random weights and get similar predictions. Attention is a *useful* diagnostic, not a ground truth.

### Cross-Attention in Translation

In a trained seq2seq transformer for English→French translation, the cross-attention weights between decoder and encoder often show a near-diagonal pattern (aligned words), with some interesting shifts for language pairs with different word orders.

**Key takeaway:** Attention maps let you visualize which positions attend to which, providing qualitative interpretability. However, attention weights are not a perfect causal explanation — they are one lens, not the complete picture.

---

## Section 8: Computational Cost and Practical Considerations

### The O(n²) Problem

The most significant limitation of attention is its **quadratic complexity** in sequence length $n$.

Computing $Q K^T$ requires $n^2 \cdot d_k$ multiply-adds. For a sequence of length $n$:
- The attention matrix is $n \times n$
- Memory: $O(n^2)$ to store the matrix
- Compute: $O(n^2 d_k)$ floating-point operations

For a short sentence of 50 tokens, $n^2 = 2500$ — negligible. For a document of 2000 tokens, $n^2 = 4{,}000{,}000$. For a book of 100,000 tokens, $n^2 = 10^{10}$ — infeasible on standard hardware.

```python
import numpy as np

def attention_cost_analysis(seq_lengths, d_k=64, h=8, L=12):
    """
    Estimate the number of FLOPs for multi-head attention.
    seq_lengths: list of sequence lengths to analyze
    d_k: key dimension per head
    h: number of heads
    L: number of layers
    """
    print(f"{'Length':>8} | {'Attn Matrix':>15} | {'FLOPs (M)':>12}")
    print("-" * 45)
    for n in seq_lengths:
        matrix_entries = n * n
        # QK^T: n*d_k * n multiplications per head
        flops_per_layer = h * (2 * n * n * d_k + n * n * (d_k // h))
        total_flops = L * flops_per_layer / 1e6
        print(f"{n:>8} | {matrix_entries:>15,} | {total_flops:>12.1f}")

attention_cost_analysis([64, 256, 512, 1024, 2048, 4096])
```

Output (approximate):
```
  Length |     Attn Matrix |    FLOPs (M)
---------------------------------------------
      64 |           4,096 |          0.0
     256 |          65,536 |          0.5
     512 |         262,144 |          2.0
    1024 |       1,048,576 |          7.9
    2048 |       4,194,304 |         31.5
    4096 |      16,777,216 |        126.1
```

### Approximate Attention Methods

The research community has developed many methods to reduce the $O(n^2)$ cost:

| Method | Complexity | Idea |
|---|---|---|
| **Longformer** | $O(n \cdot w)$ | Sliding window + global tokens |
| **BigBird** | $O(n)$ | Random + window + global |
| **Linformer** | $O(n)$ | Low-rank approximation of attention matrix |
| **FlashAttention** | $O(n^2)$ compute, $O(n)$ memory | Reorders computation for GPU efficiency |
| **Linear Attention** | $O(n)$ | Kernel trick to avoid materializing $n \times n$ matrix |

**FlashAttention** deserves special mention: it does not reduce the theoretical complexity but dramatically reduces **memory** by never materializing the full $n \times n$ matrix. Instead, it uses tiling and fused CUDA kernels. This makes standard attention practical for sequences up to $\sim$8k-16k tokens without approximation, and it is now used in essentially every production LLM.

### Comparing Attention to RNNs

| Property | RNN | Self-Attention |
|---|---|---|
| Long-range dependencies | Difficult (vanishing gradients) | Direct — $O(1)$ path between any two positions |
| Sequential computation | Yes (must process token by token) | No (fully parallelizable) |
| Memory | $O(n)$ | $O(n^2)$ |
| Training speed | Slow (sequential) | Fast (parallel on GPU) |
| Context window | Theoretically unlimited (in practice ~100) | Limited by memory ($O(n^2)$) |

This parallelism is a key reason transformers trained faster and at larger scale than RNNs once GPUs became the standard training hardware.

### Comparing Bahdanau and Scaled Dot-Product Attention

| Property | Bahdanau (Additive) | Scaled Dot-Product |
|---|---|---|
| Score function | $v^T \tanh(W_s s + W_h h)$ | $q \cdot k / \sqrt{d_k}$ |
| Parameters | $W_s$, $W_h$, $v$ | Implicit in $W^Q$, $W^K$ |
| Compute | $O(n \cdot d_a)$ per query | $O(n \cdot d_k)$ per query |
| Practical speed | Slower | Faster (can use BLAS matrix multiply) |
| Used in | Seq2seq RNNs | Transformers |

**Key takeaway:** Attention is $O(n^2)$ in both memory and compute, which becomes the bottleneck for long sequences. Various approximations exist, with FlashAttention being the most widely adopted in practice. Compared to RNNs, attention enables full parallelism during training and direct modeling of long-range dependencies.

---

## Section 9: A Complete Worked Example

Nothing cements understanding like walking through the numbers by hand. Let us compute a complete scaled dot-product self-attention step with a tiny 3-token sequence.

### Setup

Suppose our sequence is ["I", "love", "NLP"] and after embedding + positional encoding we have:

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 4}$$

We use one head with $d_k = 2$ and very simple weight matrices:

$$W^Q = \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 \\ 0 & 1 \\ 1 & 0 \\ 1 & 0 \end{bmatrix}, \quad W^V = I_{4\times4}$$

```python
import numpy as np

# 3 tokens, embedding dimension 4
X = np.array([[1,0,1,0],
              [0,1,0,1],
              [1,1,0,0]], dtype=float)

W_Q = np.array([[1,0],[1,0],[0,1],[0,1]], dtype=float)
W_K = np.array([[0,1],[0,1],[1,0],[1,0]], dtype=float)
W_V = np.eye(4)   # identity: values = original embeddings

d_k = 2

# Step 1: Project to Q, K, V
Q = X @ W_Q   # (3, 2)
K = X @ W_K   # (3, 2)
V = X @ W_V   # (3, 4) = X

print("Q =\n", Q)
print("K =\n", K)
print("V =\n", V)
```

Output:
```
Q = [[1. 1.]
     [1. 1.]
     [2. 0.]]

K = [[1. 1.]
     [1. 1.]
     [0. 2.]]
```

### Step 2: Compute Scores

$$S = \frac{QK^T}{\sqrt{d_k}} = \frac{QK^T}{\sqrt{2}}$$

```python
# Step 2: raw scores
S = Q @ K.T / np.sqrt(d_k)
print("Scores S =\n", S)
# For row 0 ("I"):
# s[0,0] = (1*1 + 1*1)/√2 = 2/1.41 ≈ 1.41
# s[0,1] = (1*1 + 1*1)/√2 ≈ 1.41  ("I" and "love" have same Q and K)
# s[0,2] = (1*0 + 1*2)/√2 ≈ 1.41  (different pattern for "NLP")
```

### Step 3: Softmax to Get Attention Weights

```python
# Step 3: softmax along axis=-1 (over keys)
S_shifted = S - S.max(axis=-1, keepdims=True)  # numerical stability
exp_S = np.exp(S_shifted)
A = exp_S / exp_S.sum(axis=-1, keepdims=True)

print("Attention weights A =\n", np.round(A, 4))
print("Row sums:", A.sum(axis=-1))
```

### Step 4: Weighted Sum of Values

```python
# Step 4: output = A @ V
out = A @ V
print("Output =\n", np.round(out, 4))
# Each row is a weighted combination of the original embeddings
# High attention to similar tokens → representations become more similar
```

### Interpreting the Result

- "I" and "love" had identical Q and K vectors, so they attended to each other and to "NLP" roughly equally.
- "NLP" had a different Q vector, so its attention pattern differs.
- The output for each token is a mixture of the original embeddings, weighted by similarity.

This toy example shows that self-attention is essentially a **learned soft lookup** — similar queries and keys produce high attention weights, mixing their value representations.

**Key takeaway:** Working through a small numerical example reveals that scaled dot-product attention is a pipeline of matrix multiplications and a softmax. Every step has a concrete geometric interpretation: projection → similarity → normalization → mixing.

---

## Section 10: Attention Variants and the Modern Landscape

The original Bahdanau attention and the transformer's scaled dot-product attention spawned many variants. Understanding them helps you read research papers and choose the right tool.

### Luong (Multiplicative) Attention

Luong et al. (2015) proposed a simpler scoring function:

$$e_{ij} = s_i^T W_a h_j \quad \text{(general)}$$
$$e_{ij} = s_i^T h_j \quad \text{(dot)}$$

The "dot" version is essentially scaled dot-product attention without the scaling. The "general" version adds a learned matrix $W_a$ between the query and key.

Luong attention computes the context vector **after** generating the output, not before (a subtle architectural difference), and found that dot-product scoring often works as well as Bahdanau's MLP.

### Relative Position Attention

Standard transformers use absolute positional encodings. Several models improve on this:

**T5 Relative Bias**: instead of adding position embeddings to token embeddings, add a learned bias to each attention score based on the relative position between query and key positions:

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{|i - j|}$$

This is more generalizable — the bias $b_{|i-j|}$ encodes "how important is position-offset $\delta$?" rather than "what does absolute position 42 mean?"

**RoPE (Rotary Position Embeddings)**: used in Llama, GPT-NeoX, and most modern open-source LLMs. Rotates the query and key vectors by a position-dependent angle before the dot product:

$$q_m^{\text{rot}} = R_m q_m, \quad k_n^{\text{rot}} = R_n k_n$$

where $R_m$ is a rotation matrix depending on position $m$. The dot product then naturally captures relative position: $q_m^{\text{rot}} \cdot k_n^{\text{rot}} = q_m \cdot (R_m^T R_n k_n)$, and $R_m^T R_n$ depends only on $(m - n)$.

### Grouped-Query Attention (GQA)

Standard multi-head attention has $h$ query heads, $h$ key heads, and $h$ value heads. Grouped-query attention (used in Llama 2/3, Mistral) uses $h$ query heads but only $g < h$ key-value heads, where $g$ divides $h$ evenly.

$$\text{head}_i = \text{Attention}(Q_i, K_{\lfloor i/g \rfloor}, V_{\lfloor i/g \rfloor})$$

Each group of $h/g$ query heads shares the same K and V. This reduces the KV cache size by a factor of $h/g$ during inference — critical for serving large models.

| Method | Q heads | KV heads | KV cache size |
|---|---|---|---|
| Multi-Head Attention (MHA) | h | h | Full |
| Multi-Query Attention (MQA) | h | 1 | 1/h |
| Grouped-Query Attention (GQA) | h | g | g/h |

```python
import numpy as np

def grouped_query_attention(X, W_Qs, W_K_groups, W_V_groups, W_O, num_groups):
    """
    X: (n, d_model)
    W_Qs: list of h matrices, each (d_model, d_k) — one per query head
    W_K_groups: list of g matrices, each (d_model, d_k) — one per KV group
    W_V_groups: list of g matrices, each (d_model, d_v) — one per KV group
    W_O: (h*d_v, d_model)
    num_groups: g (number of KV groups)
    """
    h = len(W_Qs)
    heads_per_group = h // num_groups

    heads = []
    for i, W_Q in enumerate(W_Qs):
        group_idx = i // heads_per_group
        W_K = W_K_groups[group_idx]
        W_V = W_V_groups[group_idx]

        Q_i = X @ W_Q
        K_g = X @ W_K
        V_g = X @ W_V

        d_k = Q_i.shape[-1]
        scores = Q_i @ K_g.T / np.sqrt(d_k)
        scores -= scores.max(axis=-1, keepdims=True)
        alpha = np.exp(scores)
        alpha /= alpha.sum(axis=-1, keepdims=True)
        heads.append(alpha @ V_g)

    return np.concatenate(heads, axis=-1) @ W_O

# Toy: h=4 query heads, g=2 KV groups (2 queries share each KV)
np.random.seed(42)
n, d_model, d_k = 5, 8, 4
h, g = 4, 2
W_Qs = [np.random.randn(d_model, d_k)*0.1 for _ in range(h)]
W_Ks = [np.random.randn(d_model, d_k)*0.1 for _ in range(g)]
W_Vs = [np.random.randn(d_model, d_k)*0.1 for _ in range(g)]
W_O  = np.random.randn(h*d_k, d_model)*0.1
X = np.random.randn(n, d_model)
out = grouped_query_attention(X, W_Qs, W_Ks, W_Vs, W_O, num_groups=g)
print("GQA output shape:", out.shape)  # (5, 8)
print(f"KV cache reduction: {g}/{h} = {g/h:.0%} of MHA")
```

### Sparse Attention Patterns

For very long sequences, the full $O(n^2)$ attention is impractical. Sparse attention restricts which positions can attend to which:

- **Local/Sliding window**: each token attends only to the nearest $w$ tokens. Complexity $O(n \cdot w)$.
- **Strided**: every $k$-th token attends globally; others attend locally.
- **Random**: each token attends to $r$ randomly selected positions.
- **Global tokens**: special tokens (like `[CLS]`) attend to all positions.

**Longformer** (Beltagy et al., 2020) combines sliding window + global tokens and scales to 4,096+ tokens. **BigBird** (Zaheer et al., 2020) combines random + window + global and can theoretically handle sequences of any length.

### Flash Attention

FlashAttention (Dao et al., 2022) is not a different kind of attention — it is a **hardware-aware algorithm** for computing standard scaled dot-product attention faster with less memory.

The key observation: the $n \times n$ attention matrix is the bottleneck not because of FLOPs but because of **memory bandwidth**. Writing and reading the large matrix to HBM (high-bandwidth memory on the GPU) is slow. FlashAttention uses **tiling** — processing the attention matrix in small blocks that fit in SRAM (on-chip cache) — to avoid materializing the full matrix.

Result: $O(n)$ memory instead of $O(n^2)$, 2-4× faster in practice. FlashAttention 2 and 3 further improved on this and are now standard in PyTorch (`F.scaled_dot_product_attention`).

**Key takeaway:** The attention mechanism has spawned many variants — Luong attention, relative position encodings (RoPE, T5 bias), grouped-query attention for efficient inference, sparse patterns for long sequences, and FlashAttention for hardware efficiency. Understanding the basic scaled dot-product mechanism gives you the foundation to read and understand all of these extensions.

---

## Section 11: IOAI Competition Practice — Attention Calculations

IOAI problems often test your ability to compute attention weights by hand, identify failure modes, or reason about complexity. This section provides typical problem types with solutions.

### Problem Type 1: Manual Attention Computation

**Problem**: Given Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], V = [[5, 0], [0, 5]], d_k = 2. Compute the attention output for both positions.

**Solution**:

$$S = \frac{QK^T}{\sqrt{d_k}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1&0 \\ 0&1 \end{bmatrix} \begin{bmatrix} 1&0 \\ 0&1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1&0 \\ 0&1 \end{bmatrix} \approx \begin{bmatrix} 0.707 & 0 \\ 0 & 0.707 \end{bmatrix}$$

Softmax row 0: $\text{softmax}([0.707, 0])$. Let $a = e^{0.707} \approx 2.028$, $b = e^0 = 1$.
$$\alpha_{0} = [2.028 / 3.028,\; 1/3.028] \approx [0.670, 0.330]$$

Similarly, row 1: $\alpha_1 \approx [0.330, 0.670]$ (symmetric).

Output:
$$O_0 = 0.670 \times [5, 0] + 0.330 \times [0, 5] = [3.35, 1.65]$$
$$O_1 = 0.330 \times [5, 0] + 0.670 \times [0, 5] = [1.65, 3.35]$$

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.]])
V = np.array([[5., 0.], [0., 5.]])
d_k = 2

S = Q @ K.T / np.sqrt(d_k)
S_shifted = S - S.max(axis=-1, keepdims=True)
A = np.exp(S_shifted) / np.exp(S_shifted).sum(axis=-1, keepdims=True)
O = A @ V

print("Attention weights:\n", np.round(A, 3))
print("Output:\n", np.round(O, 3))
# Position 0 attends mostly to itself (0.670) but also to position 1 (0.330)
# Output mixes both value vectors accordingly
```

### Problem Type 2: Effect of Sequence Length on Memory

**Problem**: A transformer with 12 layers processes a batch of 8 sequences, each of length $n = 512$, with $d_k = 64$ and $h = 8$ heads. How much memory (in MB) is needed to store all attention weight matrices?

**Solution**:
- Attention weight matrix per head, per layer: $n \times n = 512 \times 512 = 262{,}144$ floats
- Number of such matrices: $L \times B \times h = 12 \times 8 \times 8 = 768$
- Total floats: $262{,}144 \times 768 = 201{,}326{,}592$
- Memory in MB (float32): $201{,}326{,}592 \times 4 / (1024^2) \approx 768$ MB

```python
def attention_memory_mb(n, L, B, h, dtype_bytes=4):
    """
    Compute memory required to store all attention weight matrices.
    n: sequence length
    L: number of layers
    B: batch size
    h: number of heads
    dtype_bytes: 4 for float32, 2 for float16
    """
    floats_per_matrix = n * n
    num_matrices = L * B * h
    total_bytes = floats_per_matrix * num_matrices * dtype_bytes
    return total_bytes / (1024 ** 2)

print(f"Memory (n=512,  L=12, B=8, h=8, fp32): {attention_memory_mb(512, 12, 8, 8):.0f} MB")
print(f"Memory (n=1024, L=12, B=8, h=8, fp32): {attention_memory_mb(1024, 12, 8, 8):.0f} MB")
print(f"Memory (n=2048, L=12, B=8, h=8, fp32): {attention_memory_mb(2048, 12, 8, 8):.0f} MB")
# Note the 4x increase each time n doubles (quadratic scaling!)
```

### Problem Type 3: Why Softmax? Why Not Normalize Directly?

A common conceptual question: could we just use $\alpha_{ij} = e_{ij} / \sum_k e_{ik}$ with raw scores (not exponentiated)?

**Answer**: Using raw scores (after normalizing to sum to 1) would sometimes produce **negative** attention weights. Attending "negatively" to a value would subtract that value's contribution rather than reducing it to zero. This would make attention much harder to interpret and train.

The softmax guarantees:
1. All weights are positive: $\alpha_{ij} > 0$
2. Weights sum to 1: $\sum_j \alpha_{ij} = 1$
3. The function is smooth and differentiable everywhere
4. Peaky distributions emerge naturally for high-confidence alignment

### Problem Type 4: Multi-Head Attention Parameter Count

**Problem**: A multi-head attention layer has $d_{\text{model}} = 512$, $h = 8$ heads, $d_k = d_v = 64$. How many learnable parameters does it have?

**Solution**:
- $h$ query projections: $h \times d_{\text{model}} \times d_k = 8 \times 512 \times 64 = 262{,}144$
- $h$ key projections: same = $262{,}144$
- $h$ value projections: $h \times d_{\text{model}} \times d_v = 262{,}144$
- Output projection: $(h \times d_v) \times d_{\text{model}} = 512 \times 512 = 262{,}144$

**Total**: $4 \times 262{,}144 = 1{,}048{,}576 \approx 1M$ parameters.

Note: this equals $4 d_{\text{model}}^2 = 4 \times 512^2$. Splitting into $h$ heads does NOT change the parameter count — it only changes how the parameters are used (parallel vs single computation).

```python
def mha_params(d_model, h):
    """Count parameters in multi-head attention."""
    d_k = d_model // h
    d_v = d_model // h

    q_proj = h * d_model * d_k
    k_proj = h * d_model * d_k
    v_proj = h * d_model * d_v
    o_proj = (h * d_v) * d_model

    total = q_proj + k_proj + v_proj + o_proj
    print(f"d_model={d_model}, h={h}, d_k={d_k}")
    print(f"  Q proj: {q_proj:,}")
    print(f"  K proj: {k_proj:,}")
    print(f"  V proj: {v_proj:,}")
    print(f"  O proj: {o_proj:,}")
    print(f"  Total:  {total:,} (= 4 × {d_model}² = {4*d_model**2:,})")
    return total

mha_params(512, 8)
```

**Key takeaway:** IOAI attention problems typically test: (a) manual softmax computation, (b) memory/compute complexity analysis, (c) parameter counting, and (d) conceptual understanding of why specific design choices were made. Practice these calculation types until they become fluent.

---

## Section 12: Gradient Flow and Training Dynamics

Understanding how gradients flow through the attention mechanism helps diagnose training failures and design better architectures.

### Backpropagation Through Softmax

The gradient of the cross-entropy loss with respect to the pre-softmax scores $s_j$ (before softmax) has a clean closed form:

$$\frac{\partial \mathcal{L}}{\partial s_j} = \alpha_j - \mathbb{1}[j = \text{target}]$$

This is the "softmax is self-defeating in gradient computation" property — the gradient at position $j$ equals the current attention weight minus whether $j$ was the ground-truth focus. Training drives correct attentions toward 1 and incorrect ones toward 0.

For scaled dot-product attention, the gradient further flows through $Q K^T / \sqrt{d_k}$:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial \mathcal{L}}{\partial S} K, \quad \frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial S}\right)^T Q$$

The $1/\sqrt{d_k}$ scaling appears in the gradient too — one reason it helps training stability.

### Residual Connections and Gradient Highways

Residual connections (also called skip connections) are critical for training deep transformers:

$$y = x + F(x) \implies \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

The "1" in the gradient means there is always a direct highway from the loss to earlier layers, regardless of how small $\partial F / \partial x$ becomes. Without residuals, a 12-layer transformer would suffer severe vanishing gradients.

```python
import numpy as np

def simulate_gradient_flow(depth, use_residual=True):
    """
    Simulate gradient magnitude decay with and without residuals.
    Assume each layer multiplies gradient by 'gain' (< 1 causes vanishing).
    """
    gain_per_layer = 0.7  # realistic for tanh activations
    grad = 1.0

    for layer in range(depth):
        layer_grad = gain_per_layer
        if use_residual:
            # Residual: gradient = original + layer gradient
            grad = grad * (1 + layer_grad)
        else:
            grad = grad * layer_grad

    return grad

depth = 12
grad_with = simulate_gradient_flow(depth, use_residual=True)
grad_without = simulate_gradient_flow(depth, use_residual=False)

print(f"Gradient at input (depth={depth}):")
print(f"  With residual:    {grad_with:.4f}")
print(f"  Without residual: {grad_without:.6f}")
print(f"  Ratio: {grad_with / grad_without:.0f}x larger with residuals")
```

### Pre-Norm vs Post-Norm

The original transformer used **post-norm**: `LayerNorm(x + Sublayer(x))`. Most modern models use **pre-norm**: `x + Sublayer(LayerNorm(x))`.

Pre-norm allows gradients to flow directly through the residual path without passing through LayerNorm, which can sometimes attenuate gradients. Pre-norm models are easier to train at large depths and generally require less careful learning rate tuning.

```python
def post_norm(x, sublayer_fn, gamma, beta):
    """Original transformer: normalize AFTER residual addition."""
    mu = (x + sublayer_fn(x)).mean(axis=-1, keepdims=True)
    std = (x + sublayer_fn(x)).std(axis=-1, keepdims=True)
    return gamma * ((x + sublayer_fn(x)) - mu) / (std + 1e-6) + beta

def pre_norm(x, sublayer_fn, gamma, beta):
    """Modern default: normalize BEFORE sublayer (inside residual branch)."""
    mu = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    x_norm = gamma * (x - mu) / (std + 1e-6) + beta
    return x + sublayer_fn(x_norm)
# Pre-norm: gradient highway from output to input is unobstructed
# Post-norm: gradient must pass through LayerNorm (additional transformation)
```

**Key takeaway:** Gradients flow through attention via the softmax gradient ($\alpha_j - \mathbb{1}[\text{target}]$) and through the scaling factor $1/\sqrt{d_k}$. Residual connections provide gradient highways preventing vanishing gradients in deep networks. Pre-norm (LayerNorm before the sublayer) is now the standard choice for training stability.

---

## Quick Reference: Attention Equations

The following equations are the core formulas you must know for IOAI. Memorize them and be able to derive the dimensions of each matrix.

**Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Dimensions: $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$, output $\in \mathbb{R}^{n \times d_v}$. For self-attention, $n = m$.

**Multi-Head Attention**:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Projection shapes: $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$; $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$; $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

**Total parameters in MHA**: $4 d_{\text{model}}^2$ (when $d_k = d_v = d_{\text{model}}/h$).

**Computational complexity**: $O(n^2 d_k)$ for computing $QK^T$; $O(n^2)$ memory.

**Bahdanau Alignment Score**:

$$e_{ij} = v^T \tanh(W_s s_{i-1} + W_h h_j)$$

**Context Vector**:

$$c_i = \sum_{j=1}^{T} \alpha_{ij} h_j, \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

**Self-Attention Projections**:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V, \quad X \in \mathbb{R}^{n \times d}$$

**Causal Masking**: set $S_{ij} = -\infty$ for all $j > i$ before softmax (prevents attending to future tokens).

**Stable Softmax**: $\text{softmax}(z)_i = e^{z_i - \max(z)} / \sum_j e^{z_j - \max(z)}$

**Attention Weight Properties**: $\sum_j \alpha_{ij} = 1$, $\alpha_{ij} \geq 0$ for all $i, j$.

**Complexity Summary**:

| Operation | Time Complexity | Memory |
|---|---|---|
| $QK^T$ computation | $O(n^2 d_k)$ | $O(n^2)$ for score matrix |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $A \cdot V$ | $O(n^2 d_v)$ | $O(n d_v)$ for output |
| FlashAttention | $O(n^2 d_k)$ compute | $O(n)$ memory (no stored $n \times n$) |
| Sparse attention | $O(n \cdot s \cdot d_k)$ | $O(n \cdot s)$ where $s \ll n$ |

**Multi-Head Parameter Breakdown** (for $d_{\text{model}} = 512$, $h = 8$, $d_k = 64$):

| Component | Shape | Parameters |
|---|---|---|
| $W_i^Q$ (per head, $h$ total) | $512 \times 64$ each → $8 \times 32768$ | $262{,}144$ |
| $W_i^K$ (per head, $h$ total) | $512 \times 64$ each | $262{,}144$ |
| $W_i^V$ (per head, $h$ total) | $512 \times 64$ each | $262{,}144$ |
| $W^O$ (output projection) | $512 \times 512$ | $262{,}144$ |
| **Total** | | **$1{,}048{,}576 = 4 \times 512^2$** |

**Attention in Different Architectures**:

| Architecture | Self-Attention Style | Cross-Attention | Masking |
|---|---|---|---|
| BERT Encoder | Bidirectional full | None | Padding mask only |
| GPT Decoder | Causal (left-to-right) | None | Causal (upper triangle) |
| Transformer (full) | Encoder: bidirectional | Decoder→Encoder | Encoder: padding; Decoder: causal + padding |
| T5 Encoder | Bidirectional | None | Padding mask only |
| T5 Decoder | Causal | Decoder→Encoder | Causal + padding |

---

## Summary

This lesson covered the attention mechanism from first principles:

1. **The RNN bottleneck**: forcing all source information into a single fixed-size vector degrades performance on long sequences.

2. **Attention intuition**: instead of one context vector, compute a dynamic weighted average of all encoder states, with weights determined by query-key similarity.

3. **Bahdanau attention**: uses a small MLP to compute alignment scores; trained end-to-end with no explicit alignment supervision.

4. **Self-attention**: query, key, and value all come from the same sequence; enables direct modeling of intra-sequence dependencies.

5. **Scaled dot-product attention**: $\text{softmax}(QK^T / \sqrt{d_k}) V$; scaling prevents softmax saturation.

6. **Multi-head attention**: $h$ parallel attention computations in lower-dimensional subspaces; concatenated and projected back; different heads specialize in different relationship types.

7. **Attention maps**: visualize what the model attends to; useful but not a perfect causal explanation.

8. **Computational cost**: $O(n^2)$ in sequence length; FlashAttention and sparse/approximate methods address this in practice.

9. **Worked examples and competition skills**: computing attention weights by hand, counting parameters, reasoning about memory costs, and identifying the role of each design choice.

10. **Attention variants**: RoPE (rotary position embeddings), grouped-query attention, sparse patterns, and FlashAttention extend the basic mechanism for efficiency and expressiveness.

11. **Gradient flow**: residual connections provide gradient highways; pre-norm (apply LayerNorm before the sublayer) is the modern standard for training stability.

### Self-Test Questions

Before moving on, verify you can answer these without looking back:

1. What is the shape of the attention weight matrix for a sequence of length $n$ with $h$ heads and $d_k$ dimensions per head?
2. Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention?
3. If $d_{\text{model}} = 512$ and $h = 8$, what is $d_k$?
4. What is the total number of parameters in a multi-head attention layer with $d_{\text{model}} = 768$?
5. What does the causal mask achieve, and in which setting is it necessary?
6. Why is $B$ initialized to zeros in LoRA? (Answer: so $\Delta W = BA = 0$ at init — no-op start)
7. What is the computational complexity of self-attention in terms of sequence length $n$?
8. Name two methods that reduce the $O(n^2)$ memory cost of attention.
9. What does "self-attention is permutation equivariant" mean in plain English?
10. Describe one concrete example of a relationship that a specific attention head might learn to encode.

Answers to all of these are covered in this lesson. If any feel uncertain, revisit the relevant section before proceeding to the Transformer architecture lesson.

### Connections to Upcoming Lessons

The attention concepts from this lesson appear throughout the rest of the course:

- **Lesson 5 (Transformers)**: The transformer encoder block wraps multi-head self-attention with a feed-forward network, residual connections, and layer normalization. The decoder adds a second attention layer (cross-attention between encoder states and decoder queries) and a causal mask.

- **Lesson 6 (Fine-Tuning)**: LoRA specifically targets the attention projection matrices ($W^Q$, $W^K$, $W^V$, $W^O$) because these are where the most task-specific adaptation happens. Understanding what these matrices do (project to query/key/value spaces) helps you understand why LoRA works where it does.

- **Lesson 7 (Generation)**: Causal masking (introduced in Section 5) is the mechanism that makes GPT's autoregressive generation possible. The KV cache (mentioned in Lesson 7) stores the key and value matrices computed at each step, avoiding recomputation.

- **Attention in vision (outside this course)**: Vision Transformers (ViT) apply exactly the same self-attention mechanism to image patches. Cross-modal models (CLIP, DALL·E, Flamingo) use cross-attention between text and image representations.

The attention mechanism is not just an NLP technique — it is a general-purpose learnable aggregation operation that has become fundamental across machine learning.

Attention is the foundation of the next lesson's topic — the Transformer architecture — which builds encoder and decoder blocks from multi-head attention, position-wise feedforward networks, residual connections, and layer normalization.
