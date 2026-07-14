---
title: Transformers and BERT
track: nlp
order: 5
estimatedTime: 55
difficulty: advanced
---

# Transformers and BERT

In 2017, Vaswani et al. published "Attention Is All You Need," introducing the Transformer architecture. It eliminated recurrence entirely and relied solely on attention. Within two years, BERT and GPT demonstrated that pre-training transformers on massive text corpora and then fine-tuning on downstream tasks was the dominant paradigm in NLP. This lesson builds the Transformer piece by piece, then explains how BERT and GPT use it.

---

## Section 1: From RNNs to Transformers — A Paradigm Shift

### The Problem with Sequential Models

RNNs process sequences token by token. Token $t$ cannot be processed until token $t-1$ is done. This sequential dependency has three major consequences:

1. **Training is slow**: you cannot use a GPU efficiently when each step depends on the previous one.
2. **Long-range dependencies are hard**: information must flow through many time steps, causing vanishing gradients.
3. **Memory is expensive**: to process a batch of variable-length sequences, you must pad or truncate.

LSTMs and GRUs mitigate (2) with gating mechanisms, but they do not eliminate the fundamental sequential bottleneck.

### The Transformer Hypothesis

What if we removed recurrence entirely? The Transformer does exactly this:
- Replace recurrent processing with **self-attention** — every position attends to every other position simultaneously.
- Process the entire sequence in parallel using matrix operations that GPUs are designed for.
- Add positional encoding to inject sequence order information (since attention itself is order-agnostic).

The result: models that train an order of magnitude faster, scale to billions of parameters, and achieve state-of-the-art on essentially every NLP benchmark.

### High-Level Architecture

The original Transformer has an **encoder-decoder** structure, designed for sequence-to-sequence tasks (e.g., machine translation).

```
Source Sequence                    Target Sequence
       ↓                                  ↓
[Embeddings + Positional Enc.]   [Embeddings + Positional Enc.]
       ↓                                  ↓
  Encoder Stack                    Decoder Stack
  (N=6 layers)                     (N=6 layers)
       ↓                                  ↓
  Encoder Output  ────────────→  Cross-Attention
                                          ↓
                                  Linear + Softmax
                                          ↓
                                  Output Probabilities
```

Each encoder block contains: Self-Attention → Add & Norm → FFN → Add & Norm

Each decoder block contains: Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm

BERT uses only the encoder. GPT uses only the decoder. Most modern LLMs (GPT-4, Claude, Llama) are decoder-only.

**Key takeaway:** The Transformer replaces sequential RNN processing with parallel self-attention, enabling much faster training and better scaling. The original design has an encoder-decoder structure; BERT and GPT are specialized encoder-only and decoder-only variants.

---

## Section 2: Positional Encoding

### Why Attention Needs Position Information

Self-attention is **permutation equivariant**: if you shuffle the input tokens, the output tokens shuffle in exactly the same way, with no other change. The attention score between positions $i$ and $j$ depends only on their content (via $Q_i \cdot K_j$), not on their absolute positions or relative distance.

This is a problem: "Dog bites man" and "Man bites dog" contain the same tokens and would produce identical attention outputs — but they mean completely different things.

We need to inject positional information before the first transformer layer.

### Sinusoidal Positional Encoding

The original Transformer uses sinusoidal functions of different frequencies for each dimension of the embedding:

$$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

where:
- $\text{pos}$ is the position in the sequence (0-indexed)
- $i$ is the dimension index ($0 \leq i < d_{\text{model}}/2$)
- $d_{\text{model}}$ is the embedding dimension (e.g., 512)

The positional encoding is **added** to the token embedding: $x_{\text{pos}} = e_{\text{pos}} + PE_{\text{pos}}$.

```python
import numpy as np

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Returns PE matrix of shape (max_len, d_model).
    """
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    i = np.arange(d_model // 2)[np.newaxis, :]     # (1, d_model/2)

    # Denominator term: 10000^(2i/d_model)
    div_term = 10000 ** (2 * i / d_model)

    PE[:, 0::2] = np.sin(positions / div_term)   # even dims
    PE[:, 1::2] = np.cos(positions / div_term)   # odd dims

    return PE

PE = sinusoidal_positional_encoding(max_len=100, d_model=16)
print("PE shape:", PE.shape)  # (100, 16)
print("PE[0]:", PE[0])        # All zeros at position 0 for sin, ones for cos
print("PE[1]:", PE[1])        # First position
```

### Why Sinusoids?

The sinusoidal choice has several nice properties:
1. **Bounded**: all values are in $[-1, 1]$, matching the scale of token embeddings.
2. **Different frequencies per dimension**: low-$i$ dimensions change rapidly across positions (high frequency), while high-$i$ dimensions change slowly (low frequency). Together they uniquely encode each position.
3. **Relative positions via linear transformation**: for any fixed offset $k$, $PE_{\text{pos}+k}$ can be expressed as a linear function of $PE_{\text{pos}}$. This means the model can, in principle, learn to compute relative positions.
4. **Generalizes to longer sequences**: the formula works for any length, even beyond the training length.

### Learned Positional Embeddings

An alternative: simply learn a separate embedding vector for each position. Most modern models (BERT, GPT-2) use learned embeddings rather than fixed sinusoids.

Trade-offs:
- Learned: more flexible, may fit training data better.
- Sinusoidal: works at any length, no extra parameters, slightly easier to analyze.

### Relative Positional Encodings

Recent models (T5, RoPE used by Llama) encode relative rather than absolute positions. Instead of "this token is at position 42," they encode "this token is 5 positions before that token." This generalizes better to longer sequences than seen during training. RoPE (Rotary Position Embeddings) in particular has become standard in recent open-source LLMs.

**Key takeaway:** Self-attention is order-agnostic, so positional information must be explicitly added. Sinusoidal encodings use sine/cosine functions of different frequencies to uniquely encode each position. Most modern models use learned or relative positional embeddings instead.

---

## Section 3: The Encoder Block

The Transformer encoder consists of $N$ identical blocks stacked on top of each other (the original paper uses $N=6$). Each block has two sub-layers:

1. **Multi-Head Self-Attention** (MHA)
2. **Position-wise Feed-Forward Network** (FFN)

Both sub-layers use a **residual connection** followed by **layer normalization**.

### Sub-layer 1: Multi-Head Self-Attention

We covered this in the previous lesson. In the encoder:
- Query, Key, Value all come from the **same** sequence (self-attention).
- All positions can attend to all other positions (no masking).
- $h = 8$ heads in the original paper, each with $d_k = d_v = 64$.

### Sub-layer 2: Position-wise FFN

After attention, each position is processed independently by a two-layer MLP:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

- First linear layer: $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, expands to $d_{ff}$ dimensions
- ReLU activation (or GELU in BERT)
- Second linear layer: $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, projects back to $d_{\text{model}}$

In the original paper: $d_{\text{model}} = 512$, $d_{ff} = 2048$ (a 4× expansion).

This FFN applies the same transformation to **every position independently** — "position-wise" means there is no mixing between positions in this sub-layer (that was done by attention). The FFN can be thought of as storing factual/world knowledge, while attention handles relational reasoning.

### Residual Connections

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

The input $x$ is added back to the sub-layer output before layer normalization. This pattern is called a **residual connection** (or skip connection), borrowed from ResNets in computer vision.

Residual connections help in two ways:
1. **Gradient flow**: gradients can flow directly from the output through the shortcut, bypassing the sub-layer. This prevents vanishing gradients in deep networks.
2. **Initialization**: at initialization, if the sub-layer outputs are near zero, the model starts as an approximate identity function, which is a better starting point than random chaos.

### Layer Normalization

Layer Norm normalizes across the feature dimension for each position independently:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \odot \gamma + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the $d_{\text{model}}$ features of a single token, and $\gamma$, $\beta$ are learned scale and shift parameters.

Compare to Batch Norm (used in vision): LayerNorm normalizes per-sample across features, while Batch Norm normalizes per-feature across the batch. LayerNorm works better for variable-length sequences where batch statistics are unreliable.

```python
import numpy as np

class EncoderBlock:
    """Simplified Transformer encoder block."""

    def __init__(self, d_model, d_ff, n_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize all weight matrices (normally done with He/Xavier init)
        scale = 1.0 / np.sqrt(d_model)
        # MHA weights (simplified: one set for all heads combined)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        # FFN weights
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

        # LayerNorm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mu) / np.sqrt(var + eps) + beta

    def ffn(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def attention(self, X):
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        scores = Q @ K.T / np.sqrt(self.d_model)
        scores -= scores.max(axis=-1, keepdims=True)
        alpha = np.exp(scores)
        alpha /= alpha.sum(axis=-1, keepdims=True)
        return (alpha @ V) @ self.W_O

    def forward(self, X):
        # Sub-layer 1: MHA + residual + LayerNorm
        X = self.layer_norm(X + self.attention(X), self.gamma1, self.beta1)
        # Sub-layer 2: FFN + residual + LayerNorm
        X = self.layer_norm(X + self.ffn(X), self.gamma2, self.beta2)
        return X

np.random.seed(42)
block = EncoderBlock(d_model=64, d_ff=256, n_heads=8)
X = np.random.randn(10, 64)  # 10 tokens, 64-dim embeddings
out = block.forward(X)
print("Encoder block output shape:", out.shape)  # (10, 64)
```

**Key takeaway:** Each encoder block applies: (1) multi-head self-attention with a residual connection and layer norm, then (2) a position-wise FFN with a residual connection and layer norm. Stacking 6-24 such blocks builds a powerful contextual representation.

---

## Section 4: The Decoder Block

The decoder generates output tokens one at a time during inference, conditioned on the encoder's output. Each decoder block has **three** sub-layers:

1. **Masked Multi-Head Self-Attention**
2. **Multi-Head Cross-Attention** (encoder-decoder attention)
3. **Position-wise FFN**

All three use residual connections and layer normalization.

### Sub-layer 1: Masked Self-Attention

The decoder attends to the target sequence generated so far. During training, the full target sequence is available, but we must prevent position $i$ from attending to positions $> i$ (future tokens). We enforce this with a **causal mask** (upper-triangular mask setting future scores to $-\infty$).

This is also called **autoregressive** masking.

### Sub-layer 2: Cross-Attention

This is where the decoder "reads" the encoder's output:
- **Query**: comes from the decoder's previous sub-layer
- **Key and Value**: come from the encoder's output

$$\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$$

This is the mechanism that lets the decoder focus on relevant parts of the source sentence when generating each output word. It is essentially Bahdanau attention, rewritten in the QKV framework.

### Autoregressive Decoding at Inference

During inference, the decoder generates tokens one at a time:

```python
def autoregressive_decode(encoder_output, start_token, max_len, model):
    """
    Pseudocode for autoregressive decoding.
    encoder_output: (n_src, d_model) — fixed after encoding
    """
    generated = [start_token]

    for step in range(max_len):
        # Embed and add positional encoding
        decoder_input = embed_and_encode(generated)

        # Decoder forward pass
        decoder_output = model.decoder(decoder_input, encoder_output)

        # Take only the last position's output
        logits = model.output_head(decoder_output[-1])  # (vocab_size,)
        next_token = logits.argmax()  # greedy decoding

        generated.append(next_token)
        if next_token == END_TOKEN:
            break

    return generated
```

During training, the entire target sequence is passed at once (with causal masking), and all positions are trained simultaneously — this is called **teacher forcing** and is what makes training efficient.

**Key takeaway:** The decoder adds a causal mask to self-attention (preventing the model from seeing future tokens) and a cross-attention layer (reading from the encoder). During inference, tokens are generated one at a time in a loop; during training, the full target sequence is used with masking for efficiency.

---

## Section 5: BERT — Bidirectional Encoder Representations from Transformers

BERT (Devlin et al., Google, 2018) is a landmark model that showed you can pre-train a transformer encoder on massive unlabeled text and then fine-tune it for essentially any NLP task.

### Architecture

BERT is a **stack of transformer encoder blocks** with no decoder. The original paper released two sizes:

| Model | Layers ($L$) | Hidden size ($H$) | Heads ($A$) | Parameters |
|---|---|---|---|---|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Input Representation

BERT's input is a concatenation of special tokens and subword tokens:

```
[CLS] The cat sat on the mat [SEP] It was tired [SEP]
```

- `[CLS]`: classification token, prepended to every input. Its final representation is used for sequence-level tasks.
- `[SEP]`: separator between segments (used for sentence-pair tasks).
- Segment embeddings: identify which sentence (A or B) each token belongs to.

Total embedding = token embedding + positional embedding + segment embedding.

### Pre-Training Task 1: Masked Language Modeling (MLM)

15% of input tokens are randomly selected:
- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% are kept unchanged

The model must predict the original token at masked positions. This is a "fill in the blank" task.

$$\mathcal{L}_{\text{MLM}} = -\sum_{\text{masked pos}} \log P(\text{token}_i \mid \text{context})$$

```
Input:  [CLS] The [MASK] sat on the mat [SEP]
Output:       →    cat  ←  (predict here)
```

MLM is **bidirectional**: the model sees context on both sides of the masked token. This is why "bidirectional" is in BERT's name, and it is what distinguishes BERT from GPT (which can only see the left context).

### Pre-Training Task 2: Next Sentence Prediction (NSP)

BERT is also trained to predict whether two sentences appear consecutively in text:

```
Input A: [CLS] The cat sat on the mat [SEP] It was tired [SEP]
Label: IsNext (True)

Input B: [CLS] The cat sat on the mat [SEP] The stock market crashed [SEP]
Label: NotNext (False)
```

The `[CLS]` token's representation is fed to a binary classifier.

**Note**: later research (RoBERTa, 2019) showed NSP provides minimal benefit, and many models dropped it. MLM is the key pre-training objective.

```python
import numpy as np

def mlm_mask(token_ids, vocab_size, mask_token_id, mask_prob=0.15, seed=42):
    """
    Apply MLM masking to a sequence of token IDs.
    Returns: masked_ids, labels (original IDs at masked positions, -100 elsewhere)
    """
    rng = np.random.RandomState(seed)
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, -100)  # -100 = ignore in loss

    n = len(token_ids)
    num_to_mask = max(1, int(n * mask_prob))
    mask_positions = rng.choice(n, size=num_to_mask, replace=False)

    for pos in mask_positions:
        labels[pos] = token_ids[pos]  # Store original token for loss
        r = rng.random()
        if r < 0.80:
            masked_ids[pos] = mask_token_id       # Replace with [MASK]
        elif r < 0.90:
            masked_ids[pos] = rng.randint(vocab_size)  # Random token
        # else: keep original (10% of cases)

    return masked_ids, labels

# Example
token_ids = np.array([101, 1996, 4937, 2938, 2006, 1996, 13523, 102])
# [CLS] The  cat   sat   on   the   mat  [SEP]
masked, labels = mlm_mask(token_ids, vocab_size=30522, mask_token_id=103)
print("Original:", token_ids)
print("Masked:  ", masked)
print("Labels:  ", labels)  # -100 everywhere except masked positions
```

### BERT Fine-Tuning

After pre-training on billions of tokens (BooksCorpus + Wikipedia), BERT is fine-tuned on specific tasks:

- **Classification** (e.g., sentiment): add a linear layer on top of `[CLS]`
- **Token classification** (e.g., NER): add a linear layer on top of each token's representation
- **Question answering** (SQuAD): add two linear layers to predict start and end token of the answer span
- **Sentence pair** (e.g., NLI): feed both sentences separated by `[SEP]`, classify `[CLS]`

Fine-tuning typically requires just a few hours on a single GPU and 2-5 epochs.

**Key takeaway:** BERT pre-trains a transformer encoder on masked language modeling (predict masked tokens, bidirectional context). The learned representations transfer remarkably well — fine-tuning BERT set new records on 11 NLP benchmarks simultaneously when it was released.

---

## Section 6: GPT — Generative Pre-Training

GPT (Radford et al., OpenAI, 2018) takes the opposite design choice: use only the **decoder** (autoregressive, left-to-right) and pre-train on language modeling.

### Decoder-Only Architecture

GPT uses a stack of transformer decoder blocks with:
- **Causal self-attention** (can only see left context)
- No cross-attention (no encoder)
- No `[SEP]` or `[CLS]` tokens

### Causal Language Modeling (CLM)

The training objective is simply: predict the next token given all previous tokens.

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

This is the classical language model objective. Given the text "The cat sat on the," the model must predict "mat."

```python
import numpy as np
from scipy.special import softmax

def causal_lm_loss(logits, targets):
    """
    Compute causal LM cross-entropy loss.
    logits: (T, vocab_size) — predictions for each position
    targets: (T,) — actual next tokens (shifted by 1)
    """
    T, V = logits.shape
    total_loss = 0.0

    for t in range(T):
        probs = softmax(logits[t])
        total_loss -= np.log(probs[targets[t]] + 1e-10)

    return total_loss / T

# Toy example
T, V = 5, 100
np.random.seed(42)
logits = np.random.randn(T, V)
targets = np.array([23, 5, 67, 12, 89])
loss = causal_lm_loss(logits, targets)
print(f"Causal LM loss: {loss:.3f}")
```

### GPT vs BERT: Key Differences

| Property | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM + NSP | Causal LM |
| Strengths | Understanding tasks | Generation tasks |
| Fine-tuning | Adds task head | Prompt-based or task head |
| Can generate | No (by design) | Yes |

BERT excels at tasks requiring understanding of the full context (e.g., NER, classification). GPT excels at generation — it can continue text, translate, summarize, and in its larger versions (GPT-3, GPT-4) answer questions through in-context learning.

### GPT-2 and GPT-3: Scaling Up

GPT-2 (2019, 1.5B parameters) showed that scaling GPT leads to impressive generative capabilities. GPT-3 (2020, 175B parameters) demonstrated **few-shot in-context learning**: by putting a few examples in the prompt, the model can perform tasks without any gradient updates. This shifted the paradigm from "fine-tune for every task" to "prompt for every task."

**Key takeaway:** GPT is a decoder-only transformer trained on causal language modeling (next-token prediction). It is fundamentally a generative model. Scaling GPT led to in-context learning and the modern large language model paradigm.

---

## Section 7: Pre-Training vs Fine-Tuning

### The Transfer Learning Paradigm

Before transformers, NLP models were trained from scratch on each task. This required large amounts of labeled data per task and ignored the vast knowledge in unlabeled text.

The transformer pre-training + fine-tuning paradigm:
1. **Pre-train** on a massive unlabeled corpus (e.g., Wikipedia, Common Crawl) using a self-supervised objective (MLM or CLM). This teaches the model language structure, facts, reasoning patterns.
2. **Fine-tune** on a small labeled dataset for the specific task. Since the model already understands language, it can adapt quickly with limited data.

This mirrors how humans learn: broad general knowledge (schooling) then domain-specific expertise (job training).

### Why Pre-Training Helps

The pre-trained model has learned:
- **Syntax**: sentence structure, dependencies, grammaticality
- **Semantics**: word meanings, synonymy, analogy
- **World knowledge**: facts, relationships, common sense
- **Discourse**: paragraph structure, topic coherence

All of this is encoded in the model's weights as a result of having to predict masked or next tokens across billions of examples. Fine-tuning then "unlocks" this knowledge for a specific task.

### The Data Efficiency Argument

Without pre-training, a classification model for sentiment analysis might need 100,000 labeled examples to perform well. With a pre-trained BERT, you might only need 1,000 — a 100× reduction in labeling cost.

```
Performance
    |                              *** Fine-tuned BERT
    |                       *****
    |                  *****
    |             *****
    |        *****
    |   *****
    |***
    |               *** From-scratch model
    |         *****
    |    *****
    | ***
    +----------------------------------------> Labeled Training Examples
          100   1K   10K   100K   1M
```

### The Pre-Training Corpus Matters

Models trained on different corpora develop different "priors":
- **BERT**: BooksCorpus + English Wikipedia (3.3B words)
- **RoBERTa**: + CC-News, OpenWebText, Stories (160GB total)
- **GPT-3**: 499B tokens from Common Crawl, WebText2, Books, Wikipedia
- **LLaMA**: 1.4T tokens from web data, code, books

More data generally helps, though quality matters as much as quantity. Low-quality web text (spam, gibberish) can hurt.

**Key takeaway:** Pre-training teaches the model general language understanding from unlabeled text. Fine-tuning then specializes the model for a task with far less labeled data than training from scratch. This transfer learning paradigm is the foundation of modern NLP.

---

## Section 8: Tokenization Recap and Scale

### Subword Tokenization

Before the transformer even sees text, it must be tokenized. Modern transformers use subword tokenization:

- **WordPiece** (BERT): builds a vocabulary by greedily merging frequent substrings.
- **BPE (Byte-Pair Encoding)** (GPT-2): similar, starts from characters and merges frequent pairs.
- **SentencePiece** (T5, Llama): language-agnostic, treats text as a sequence of Unicode characters.

Subword tokenization balances vocabulary size and coverage:
- Common words ("the", "is") get single tokens.
- Rare words ("astrophysicist") are split ("astro", "physic", "ist").
- Unknown words are never truly unknown — worst case, each character is a token.

```python
# Simplified BPE tokenization concept
def bpe_tokenize(text, vocab):
    """
    Conceptual BPE: split into characters, then apply merges.
    vocab: dict of (pair) -> merged_token, ordered by frequency
    """
    # Start: split into characters + end-of-word marker
    tokens = list(text) + ['</w>']

    # Apply merges in order
    for (left, right), merged in vocab.items():
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == left and tokens[i+1] == right:
                tokens = tokens[:i] + [merged] + tokens[i+2:]
            else:
                i += 1

    return tokens

# Example vocabulary (in practice, learned from data)
example_vocab = {
    ('t', 'h'): 'th',
    ('th', 'e'): 'the',
    ('c', 'a'): 'ca',
    ('ca', 't'): 'cat',
}
print(bpe_tokenize("the cat", example_vocab))
# ['the', '</w>', 'cat', '</w>']
```

### The Embedding Layer

Each subword token is mapped to a dense vector of dimension $d_{\text{model}}$ via an embedding matrix $E \in \mathbb{R}^{|V| \times d_{\text{model}}}$. This matrix is learned during pre-training. The output and input embedding matrices are often **tied** (shared weights) to reduce parameters and improve generalization.

### Scale: Parameters, Layers, d_model

The original Transformer:
- $d_{\text{model}} = 512$
- $d_{ff} = 2048$
- $h = 8$ heads
- $N = 6$ encoder + 6 decoder layers
- Parameters: ~65M

Modern scale:

| Model | Params | Layers | $d_{\text{model}}$ | Heads | Training tokens |
|---|---|---|---|---|---|
| BERT-Base | 110M | 12 | 768 | 12 | 3.3B |
| BERT-Large | 340M | 24 | 1024 | 16 | 3.3B |
| GPT-2 | 1.5B | 48 | 1600 | 25 | 40B |
| GPT-3 | 175B | 96 | 12288 | 96 | 300B |
| LLaMA-2 7B | 7B | 32 | 4096 | 32 | 2T |
| LLaMA-2 70B | 70B | 80 | 8192 | 64 | 2T |

### Parameter Counting for a Transformer Block

For a single encoder block with $d_{\text{model}}$ and $d_{ff}$:
- Multi-head attention (4 projection matrices): $4 d_{\text{model}}^2$
- FFN: $2 d_{\text{model}} d_{ff}$ (plus biases, small)
- Layer norms: $4 d_{\text{model}}$ (two layernorms, each $\gamma$ and $\beta$)

Total per block (approximately): $4 d_{\text{model}}^2 + 2 d_{\text{model}} d_{ff}$

For BERT-Base: $4 \times 768^2 + 2 \times 768 \times 3072 = 2{,}359{,}296 + 4{,}718{,}592 \approx 7.1M$ per block, times 12 blocks = $\sim$85M (remainder in embeddings).

```python
def count_transformer_params(vocab_size, max_seq_len, d_model, d_ff, n_heads, n_layers):
    """Approximate parameter count for a BERT-style encoder."""
    # Embedding layers
    token_emb = vocab_size * d_model
    pos_emb = max_seq_len * d_model
    seg_emb = 2 * d_model  # two segments A/B

    # Per encoder block
    mha = 4 * d_model * d_model       # Q, K, V, O projections
    ffn = 2 * d_model * d_ff          # two linear layers
    layernorms = 2 * 2 * d_model      # two LN per block, each gamma+beta
    per_block = mha + ffn + layernorms

    total = token_emb + pos_emb + seg_emb + n_layers * per_block

    print(f"Token embeddings: {token_emb:>12,}")
    print(f"Positional embs:  {pos_emb:>12,}")
    print(f"Segment embs:     {seg_emb:>12,}")
    print(f"Per block:        {per_block:>12,}")
    print(f"All blocks:       {n_layers * per_block:>12,}")
    print(f"Total:            {total:>12,}")
    return total

print("BERT-Base:")
count_transformer_params(
    vocab_size=30522, max_seq_len=512,
    d_model=768, d_ff=3072,
    n_heads=12, n_layers=12
)
```

**Key takeaway:** Transformer models are characterized by $d_{\text{model}}$, $d_{ff}$, number of heads, and number of layers. Parameters scale as $O(L \cdot d_{\text{model}}^2)$ (ignoring embeddings). Modern LLMs have 7B-175B+ parameters, trained on trillions of tokens.

---

## Section 9: BERT Variants and the RoBERTa Family

After BERT's success, researchers identified several ways to improve or adapt it. Understanding these variants illustrates what matters most in pre-training.

### RoBERTa (Robustly Optimized BERT)

Liu et al. (Facebook AI, 2019) retrained BERT with four key changes:
1. **More data**: 160GB of text (vs BERT's 16GB)
2. **Longer training**: 10× more training steps
3. **Larger batches**: 8,192 sequences per batch (vs 256)
4. **Dynamic masking**: generate new masks for each epoch (vs static once-for-all masks)
5. **Remove NSP**: drop next-sentence prediction entirely — it provides little benefit

RoBERTa matched or exceeded BERT-Large on all GLUE benchmarks using the BERT-Base architecture, demonstrating that training regime matters as much as architecture.

```python
# Conceptual: the ONLY difference in training is the config
roberta_training_config = {
    "pretrain_data": "160GB (vs BERT's 16GB)",
    "training_steps": 500_000,          # vs BERT's 1_000_000 but bigger batches
    "batch_size": 8_192,                # vs BERT's 256
    "masking": "dynamic",               # new masks each epoch
    "nsp": False,                       # removed
    "optimizer": "Adam",
    "lr": 4e-4,
    "warmup_steps": 24_000,
}
```

### DistilBERT (Knowledge Distillation)

DistilBERT (Sanh et al., Hugging Face, 2019) is a **compressed** version of BERT — 40% smaller, 60% faster, retaining 97% of BERT's performance.

Training technique: **knowledge distillation**. The student (DistilBERT, 6 layers) is trained to mimic the teacher (BERT, 12 layers) on its soft output probabilities, not just the hard one-hot labels.

Loss for distillation:
$$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, y_{\text{student}}) + (1-\alpha) \mathcal{L}_{\text{KD}}(p_{\text{teacher}}, p_{\text{student}})$$

where $\mathcal{L}_{\text{KD}}$ is KL-divergence between teacher and student soft probabilities.

The soft probabilities carry more information than hard labels — they encode the relative similarity between classes as the teacher learned them.

### ALBERT (A Lite BERT)

ALBERT (Lan et al., Google, 2019) reduces BERT's parameters via two techniques:

1. **Factorized embedding parameterization**: instead of $|V| \times d_{\text{model}}$ embedding matrix, use $|V| \times d_e \times d_e \times d_{\text{model}}$ (much smaller when $d_e \ll d_{\text{model}}$). For BERT-Large, this reduces embedding parameters from 32K×1024 to 32K×128 + 128×1024.

2. **Cross-layer parameter sharing**: share all parameters across Transformer layers. Instead of 24 independent sets of weights, all 24 layers use the same weights. Dramatic parameter reduction with modest performance cost.

ALBERT-xxlarge (235M unique parameters, but equivalent depth of 4096-dim) outperforms BERT-Large (340M parameters) on several benchmarks.

### DeBERTa (Decoding-Enhanced BERT with Disentangled Attention)

DeBERTa (He et al., Microsoft, 2020/2021) improves BERT with two ideas:

1. **Disentangled attention**: represent each word with two vectors — content and position — and compute attention on all four combinations: content-to-content, content-to-position, position-to-content, position-to-position.

2. **Enhanced mask decoder**: uses absolute position information only in the final layers (where positional information for predicting masked tokens is critical), not in all layers.

DeBERTa-v3 consistently achieves top results on GLUE and SuperGLUE and is a strong choice for NLP competitions.

**Key takeaway:** BERT spawned a family of variants — RoBERTa (better training regime), DistilBERT (knowledge distillation for speed), ALBERT (parameter sharing for efficiency), and DeBERTa (disentangled attention for better representation). Knowing these helps you pick the right base model for a task.

---

## Section 10: The Text-to-Text Framework and T5

### Every NLP Task as Text Generation

GPT treats every task as "complete this text." BERT treats every task as "classify this sequence." T5 (Raffel et al., Google, 2020) unifies everything under a single **text-to-text** framework:

- Input: a string
- Output: a string
- Training objective: predict the output string token by token

Every NLP task is reformulated as text-to-text:

| Task | Input | Output |
|---|---|---|
| Translation | "translate English to French: The cat sat." | "Le chat s'est assis." |
| Summarization | "summarize: [article text]" | "[summary]" |
| Classification | "sst2 sentence: It was great!" | "positive" |
| NER | "ner: Barack Obama visited Paris." | "Person: Barack Obama; Location: Paris" |
| QA | "question: Who visited Paris? context: Barack Obama visited Paris." | "Barack Obama" |

```python
# T5-style input formatting for different tasks
def format_t5_input(task, **kwargs):
    """Format an input for T5's text-to-text framework."""
    templates = {
        "translate_en_fr": "translate English to French: {text}",
        "summarize": "summarize: {text}",
        "sentiment": "sst2 sentence: {text}",
        "ner": "ner: {text}",
        "qa": "question: {question} context: {context}",
        "mnli": "mnli hypothesis: {hypothesis} premise: {premise}",
    }
    template = templates.get(task, "{text}")
    return template.format(**kwargs)

# Examples
print(format_t5_input("translate_en_fr",
    text="The cat sat on the mat."))
print(format_t5_input("sentiment",
    text="This movie was absolutely fantastic."))
print(format_t5_input("qa",
    question="Who visited Paris?",
    context="Barack Obama visited Paris in 2011."))
```

### T5 Architecture

T5 uses the standard encoder-decoder transformer with one key modification: **span corruption** as the pre-training objective instead of masked language modeling.

Span corruption masks contiguous spans of tokens (not individual tokens) and replaces each span with a sentinel token like `<extra_id_0>`. The decoder must reconstruct all masked spans.

```
Input:  "The <extra_id_0> sat on the <extra_id_1> mat."
Target: "<extra_id_0> cat <extra_id_1> velvet"
```

This is more efficient than MLM — masking 15% of individual tokens means most forward passes are "wasted" on predicting easy tokens. Masking 15% of tokens but in contiguous spans means each span is more predictive.

T5 is available in sizes: Small (60M), Base (220M), Large (770M), XL (3B), XXL (11B). The text-to-text framework means a single fine-tuning paradigm handles all tasks.

### Encoder-Decoder vs Decoder-Only

Modern LLMs are overwhelmingly decoder-only. Why did T5's encoder-decoder approach fall out of fashion?

1. **Decoder-only scales better**: empirically, decoder-only models benefit more cleanly from scale.
2. **Unification**: a decoder-only model can do both understanding (via prompting) and generation without architectural changes.
3. **In-context learning**: large decoder-only models (GPT-3) can perform many tasks by conditioning on examples in the prompt, without fine-tuning.
4. **Training simplicity**: one objective (CLM) vs two objectives (MLM encoder + CLM decoder).

However, encoder-decoder models (T5, Flan-T5) remain competitive for tasks where structured, bounded outputs are required (translation, structured prediction) and for scenarios where bidirectional context in the encoder matters.

```python
import numpy as np
from scipy.special import softmax

def span_corruption_masking(tokens, mask_prob=0.15, mean_span_length=3, seed=42):
    """
    T5-style span corruption.
    tokens: list of token IDs
    Returns: (corrupted_input, target_output)
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    num_to_mask = max(1, int(n * mask_prob))

    corrupted = list(tokens)
    masked_spans = []
    sentinel_id = 32000  # T5 uses <extra_id_0>, <extra_id_1>, ...

    i = 0
    sentinel_count = 0
    span_positions = set()

    # Select spans
    while len(span_positions) < num_to_mask and i < n:
        if i not in span_positions:
            span_len = max(1, int(rng.exponential(mean_span_length - 1)) + 1)
            span = list(range(i, min(i + span_len, n)))
            for pos in span:
                span_positions.add(pos)
            masked_spans.append((span, sentinel_id + sentinel_count))
            sentinel_count += 1
        i += 1

    # Build corrupted input and target
    corrupted_input = []
    target = []
    in_span = False
    for idx, tok in enumerate(tokens):
        span_info = next((s for s in masked_spans if idx in s[0] and s[0][0] == idx), None)
        if span_info:
            corrupted_input.append(span_info[1])  # sentinel
            target.append(span_info[1])
            for pos in span_info[0]:
                target.append(tokens[pos])
        elif idx not in span_positions:
            corrupted_input.append(tok)

    return corrupted_input, target

# Example (using integers as proxy token IDs)
tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
inp, tgt = span_corruption_masking(tokens, seed=7)
print("Original:  ", tokens)
print("Corrupted: ", inp)
print("Target:    ", tgt)
```

**Key takeaway:** T5 unifies all NLP tasks as text-to-text, using span corruption for pre-training. This elegant framework lets a single model handle translation, summarization, classification, and QA with the same training recipe. Modern LLMs moved toward decoder-only but T5 remains widely used for structured prediction tasks.

---

## Section 11: Understanding BERT's Representations

One of the most revealing analyses you can do with a trained BERT model is to probe its intermediate representations. This section explores what different layers encode and how to interpret the representations.

### The Layer-by-Layer View

Jawahar et al. (2019) analyzed BERT's 12 layers and found a consistent pattern:

- **Early layers (1-4)**: syntactic surface features — part-of-speech, morphology, local n-grams
- **Middle layers (5-9)**: syntactic relationships — dependency parsing, subject-verb agreement, chunking
- **Late layers (10-12)**: semantic features — co-reference, semantic roles, sentence meaning

This mirrors the classical NLP pipeline: morphology → syntax → semantics, but learned end-to-end.

### Probing Classifiers

A **probing classifier** is a simple linear model trained to predict a linguistic property from a model's intermediate representation. If the linear model achieves high accuracy, the property is "encoded" in the representation; if not, the representation doesn't contain that information.

```python
import numpy as np
from scipy.special import softmax

def train_probe(representations, labels, n_epochs=50, lr=0.01, seed=42):
    """
    Train a linear probe on intermediate representations.
    representations: (N, d) — frozen BERT features at some layer
    labels: (N,) — integer labels for a linguistic property
    Returns: trained weights, training accuracy history
    """
    rng = np.random.RandomState(seed)
    N, d = representations.shape
    num_classes = len(np.unique(labels))

    W = rng.randn(d, num_classes) * 0.01
    b = np.zeros(num_classes)
    history = []

    idx = np.arange(N)
    for epoch in range(n_epochs):
        rng.shuffle(idx)
        X = representations[idx]
        y = labels[idx]

        logits = X @ W + b
        probs = softmax(logits, axis=-1)

        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1
        dlogits /= N

        W -= lr * (X.T @ dlogits)
        b -= lr * dlogits.sum(axis=0)

        preds = (representations @ W + b).argmax(axis=-1)
        acc = (preds == labels).mean()
        history.append(acc)

    return W, b, history

# Simulate probing different BERT layers for POS tagging
# (In practice you'd extract actual BERT hidden states)
np.random.seed(42)
N, d = 500, 768

# Early layer: more POS-relevant features (higher probe accuracy)
early_repr = np.random.randn(N, d)
# Late layer: less POS-relevant (more semantic)
late_repr  = np.random.randn(N, d) * 0.5

pos_labels = np.random.randint(0, 5, N)  # 5 POS tags

_, _, history_early = train_probe(early_repr, pos_labels, n_epochs=20)
_, _, history_late  = train_probe(late_repr,  pos_labels, n_epochs=20)

print(f"Early layer probe final acc: {history_early[-1]:.3f}")
print(f"Late layer probe final acc:  {history_late[-1]:.3f}")
# In practice, early layers give higher POS probe accuracy
```

### Contextual vs Static Embeddings

A key property of transformer representations is **contextuality**: the same word gets a different representation depending on context.

Compare:
- "I went to the **bank** to deposit money." (financial institution)
- "We sat by the **bank** of the river." (riverbank)

In word2vec or GloVe, both "bank" occurrences have the **same** vector. In BERT, the two "bank" vectors are different — and we can measure how different:

$$\text{similarity} = \frac{h_{\text{bank, finance}} \cdot h_{\text{bank, river}}}{\|h_{\text{bank, finance}}\| \|h_{\text{bank, river}}\|}$$

Empirically, this cosine similarity is ~0.35 for BERT (quite different), whereas for static embeddings it would be 1.0 (identical).

```python
import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# Simulate contextual representations (from trained BERT)
# In practice: use transformers library to extract hidden states
np.random.seed(42)
d = 768

# Same word, very different contexts → different representations
bank_finance = np.random.randn(d)  # "bank" in financial context
bank_river   = np.random.randn(d)  # "bank" in geographic context
bank_copy    = bank_finance.copy() # exact same context

print(f"bank (finance) vs bank (river):  cos_sim = {cosine_similarity(bank_finance, bank_river):.3f}")
print(f"bank (finance) vs bank (same):   cos_sim = {cosine_similarity(bank_finance, bank_copy):.3f}")
# In practice: ~0.35 vs ~1.00 — dramatic disambiguation
```

### The CLS Token as a Sentence Representation

BERT's `[CLS]` token aggregates information from the entire sequence through 12 layers of attention. By the final layer, the `[CLS]` representation has "seen" every token in the sequence (via attention) and distilled global context.

However, research has shown that `[CLS]` is not always the best sentence representation. An alternative: **mean pooling** — average all non-padding token representations.

$$\text{sentence\_repr} = \frac{1}{n} \sum_{i=1}^{n} h_i^{(L)}$$

Models trained with **SimCSE** (Gao et al., 2021) or **sentence-BERT** (Reimers et al., 2019) are specifically optimized for sentence representations using contrastive learning, and significantly outperform vanilla BERT's `[CLS]` for similarity tasks.

**Key takeaway:** BERT's layers encode progressively more abstract features (surface → syntax → semantics). Probing classifiers reveal what each layer encodes. The same word gets different representations in different contexts (contextuality). For sentence-level tasks, `[CLS]` or mean-pooled representations can be used, with specialized sentence embedding models (sentence-BERT) being significantly better for similarity and retrieval tasks.

---

## Section 12: Choosing the Right Pre-Trained Model

With dozens of pre-trained transformers available on Hugging Face Hub, choosing the right one is a practical skill. Here is a structured guide.

### Decision Framework

```
What type of task?
├── Sequence classification / regression
│   └── Need generation? → No → bert-base-uncased, roberta-base, deberta-v3-base
├── Token classification (NER, POS)
│   └── roberta-base or bert-base-uncased (fine-tune with per-token head)
├── Question answering (extractive)
│   └── roberta-base or deberta-v3-base (strong QA baselines)
├── Text generation / completion
│   └── gpt2, mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf
├── Translation
│   └── Helsinki-NLP/opus-mt-en-fr (MarianMT), t5-base, mbart-large-50
├── Summarization
│   └── facebook/bart-large-cnn, t5-base, google/pegasus-large
└── Sentence embeddings / retrieval
    └── sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5
```

### Model Size vs Performance Trade-Off

```python
import numpy as np

# Approximate GLUE benchmark scores (higher = better, scale 0-100)
# Source: various papers and Hugging Face leaderboards (approximate)
models = {
    "DistilBERT-base":   {"params_M": 66,   "glue_avg": 77.0, "speed_ms": 20},
    "BERT-base":         {"params_M": 110,  "glue_avg": 79.6, "speed_ms": 40},
    "RoBERTa-base":      {"params_M": 125,  "glue_avg": 86.4, "speed_ms": 45},
    "BERT-large":        {"params_M": 340,  "glue_avg": 80.5, "speed_ms": 100},
    "RoBERTa-large":     {"params_M": 355,  "glue_avg": 88.9, "speed_ms": 120},
    "DeBERTa-v3-base":   {"params_M": 184,  "glue_avg": 88.4, "speed_ms": 60},
    "DeBERTa-v3-large":  {"params_M": 440,  "glue_avg": 91.4, "speed_ms": 160},
}

print(f"{'Model':<25} {'Params':>8} {'GLUE':>8} {'Speed (ms)':>12}")
print("-" * 58)
for name, info in models.items():
    print(f"{name:<25} {info['params_M']:>7}M {info['glue_avg']:>8.1f} {info['speed_ms']:>11}")
```

### Domain-Specific Pre-Trained Models

For domain-specific NLP tasks, general pre-trained models often underperform models pre-trained on domain text:

| Domain | Model | Base |
|---|---|---|
| Biomedical | BioBERT, PubMedBERT | BERT continued on PubMed |
| Clinical | ClinicalBERT | BERT on clinical notes |
| Legal | LegalBERT | BERT on EU legislation, court decisions |
| Scientific | SciBERT | BERT on Semantic Scholar papers |
| Code | CodeBERT, GraphCodeBERT | BERT on GitHub code |
| Finance | FinBERT | BERT on financial news |

The pattern: take BERT (or RoBERTa), continue pre-training on domain text, then fine-tune. This often gives 3-5 point improvements on domain-specific benchmarks over general BERT.

### Multilingual Models

For non-English NLP:

- **mBERT** (multilingual BERT): trained on 104 languages simultaneously using MLM. Surprisingly good cross-lingual transfer — train on English NER, evaluate on French NER.
- **XLM-RoBERTa**: RoBERTa on 100 languages with more data. Generally stronger than mBERT.
- **mT5**: multilingual T5, covers 101 languages.

Cross-lingual transfer works because languages share structural features (syntactic patterns, morphological regularities) that a multilingual model encodes in shared representation space.

```python
def model_selection_guide(task, data_size, latency_sensitive, domain):
    """
    Simple heuristic for model selection.
    task: 'classification', 'ner', 'generation', 'similarity'
    data_size: number of labeled training examples
    latency_sensitive: bool — is inference speed critical?
    domain: 'general', 'biomedical', 'legal', 'code'
    """
    recommendations = []

    if domain != 'general':
        recommendations.append(f"Consider domain-specific model (e.g., BioBERT for biomedical)")

    if latency_sensitive:
        recommendations.append("Use DistilBERT or a quantized model for lower latency")
    elif task in ['classification', 'ner']:
        if data_size < 500:
            recommendations.append("DeBERTa-v3-base with LoRA (avoids overfitting)")
        elif data_size < 10000:
            recommendations.append("RoBERTa-base full fine-tune")
        else:
            recommendations.append("DeBERTa-v3-large full fine-tune for best performance")
    elif task == 'generation':
        recommendations.append("Llama-2-7B or Mistral-7B with LoRA fine-tuning")
    elif task == 'similarity':
        recommendations.append("sentence-transformers/all-mpnet-base-v2 or BAAI/bge-large")

    return recommendations

recs = model_selection_guide("classification", data_size=2000,
                              latency_sensitive=False, domain="general")
for r in recs:
    print(f"  - {r}")
```

**Key takeaway:** Choosing the right pre-trained model depends on task type, data size, latency constraints, and domain. For most English NLP classification tasks, DeBERTa-v3-base is the strongest base model. DistilBERT is best for latency-sensitive deployment. Domain-specific models outperform general ones by 3-5 points on in-domain benchmarks. XLM-RoBERTa is the go-to for multilingual tasks.

---

## Summary

This lesson built the Transformer from the ground up:

1. **Motivation**: RNNs are slow to train (sequential) and struggle with long-range dependencies. Transformers replace recurrence with attention.

2. **Positional encoding**: since attention is order-agnostic, sinusoidal or learned positional embeddings are added to token embeddings.

3. **Encoder block**: MHA + FFN, each with residual connections and layer normalization. Stacking 6-24 blocks builds rich contextual representations.

4. **Decoder block**: adds a causal mask to self-attention and a cross-attention sub-layer for reading the encoder output.

5. **BERT**: encoder-only, pre-trained with masked language modeling. Bidirectional context = better understanding.

6. **GPT**: decoder-only, pre-trained with causal language modeling. Autoregressive = natural generation. Scaling leads to in-context learning.

7. **Pre-training vs fine-tuning**: pre-train on massive unlabeled data, fine-tune efficiently on labeled tasks.

8. **Scale**: BERT-Base has 110M parameters; modern LLMs have 7B-175B+, trained on trillions of tokens.

The next lesson covers how to take a pre-trained transformer and adapt it to your specific task through fine-tuning — including full fine-tuning, feature extraction, and parameter-efficient methods like LoRA.
