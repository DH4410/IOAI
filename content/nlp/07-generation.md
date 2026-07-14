---
title: Text Generation and GPT
track: nlp
order: 7
estimatedTime: 55
difficulty: advanced
---

# Text Generation and GPT

Language models generate text by predicting one token at a time. The way we sample from those predictions — the **decoding strategy** — dramatically affects the quality, diversity, and coherence of the output. This lesson dives into the mathematics of text generation: temperature, top-k, nucleus sampling, beam search, and how reinforcement learning from human feedback (RLHF) transforms raw language models into helpful assistants. We also examine the modern landscape of large language models and the hallucination problem.

---

## Section 1: How GPT Generates Text — The Next-Token Prediction Loop

### The Core Loop

GPT (and all autoregressive language models) generate text through a simple iterative loop:

1. Start with a prompt (context tokens)
2. Pass the context through the model to get a probability distribution over the vocabulary
3. Sample (or select) the next token from that distribution
4. Append the new token to the context
5. Repeat from step 2 until you hit a stop condition (end-of-sequence token, max length, etc.)

```python
import numpy as np
from scipy.special import softmax

def generate(prompt_ids, model, max_new_tokens=50, eos_token_id=2):
    """
    Conceptual autoregressive generation loop.
    prompt_ids: list of integer token IDs (the prompt)
    model: callable, takes list of token IDs, returns logits (vocab_size,)
             for the next token position
    Returns: list of generated token IDs (not including prompt)
    """
    context = list(prompt_ids)
    generated = []

    for _ in range(max_new_tokens):
        # Forward pass: get logits for next token
        logits = model(context)  # (vocab_size,)

        # Convert logits to probabilities
        probs = softmax(logits)

        # Sample from the distribution
        next_token = np.random.choice(len(probs), p=probs)
        generated.append(next_token)
        context.append(next_token)

        if next_token == eos_token_id:
            break

    return generated
```

### From Logits to Probabilities

The final layer of a transformer language model is a linear projection from the hidden state $h \in \mathbb{R}^{d_{\text{model}}}$ to the vocabulary:

$$\text{logits} = h W_{\text{vocab}}^T + b, \quad \text{logits} \in \mathbb{R}^{|V|}$$

where $|V|$ is the vocabulary size (e.g., 32,000 for Llama, 50,257 for GPT-2).

The logits are then converted to a probability distribution with softmax:

$$P(x_t = v \mid x_{<t}) = \frac{e^{\text{logits}_v}}{\sum_{v'} e^{\text{logits}_{v'}}}$$

A key insight: the absolute values of the logits don't matter — only the differences between them. Adding a constant to all logits doesn't change the probabilities. This is why numerical stability tricks (subtracting the max) work.

### The Computational Bottleneck: KV Cache

Naively, generating each new token requires running a full forward pass over the entire context. For a 1000-token context, that means processing 1000 tokens per step — and at step 1000, the total work is $1000 \times 1000 = 10^6$ token-steps.

The **KV cache** solves this: we cache the key and value matrices from all previous attention steps. At each new token, we only compute the query for the new token and attend over the cached keys and values. This reduces the per-step computation dramatically.

```python
class KVCache:
    """Simple KV cache for efficient autoregressive generation."""

    def __init__(self):
        self.keys = []    # List of (1, d_k) arrays, one per step
        self.values = []  # List of (1, d_v) arrays, one per step

    def update(self, new_key, new_value):
        """Add a new key-value pair from the latest token."""
        self.keys.append(new_key)
        self.values.append(new_value)

    def get_all(self):
        """Return all cached keys and values stacked."""
        if not self.keys:
            return None, None
        return np.stack(self.keys, axis=0), np.stack(self.values, axis=0)

    def length(self):
        return len(self.keys)
```

**Key takeaway:** GPT generates text token by token: run forward pass → get logits → sample/select next token → append to context → repeat. The KV cache makes this efficient by avoiding redundant computation over already-processed tokens.

---

## Section 2: Temperature — Sharpening and Flattening Distributions

### What Temperature Does

Temperature is a scalar $T > 0$ that divides the logits before softmax:

$$P(x_t = v \mid x_{<t}) = \frac{e^{\text{logits}_v / T}}{\sum_{v'} e^{\text{logits}_{v'} / T}}$$

- $T = 1.0$: standard softmax, no change
- $T < 1.0$ (e.g., 0.5): **sharpens** the distribution — high-probability tokens become even more dominant, low-probability tokens are suppressed. Output is more deterministic and repetitive.
- $T > 1.0$ (e.g., 1.5): **flattens** the distribution — probabilities become more uniform. Output is more diverse and creative, but also more chaotic/incoherent.
- $T \to 0$: becomes greedy decoding (always pick the argmax)
- $T \to \infty$: becomes uniform random sampling over all vocabulary tokens

```python
import numpy as np
from scipy.special import softmax

def apply_temperature(logits, temperature):
    """Scale logits by temperature before softmax."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature

def temperature_effect(logits, temperatures):
    """Show how temperature affects the probability distribution."""
    print(f"{'Token':>10} | " + " | ".join(f"T={t:4.2f}" for t in temperatures))
    print("-" * (12 + 12 * len(temperatures)))

    token_names = ["apple", "banana", "cherry", "date", "elderberry"]

    for t in temperatures:
        scaled = apply_temperature(logits, t)
        probs = softmax(scaled)
        if t == temperatures[0]:
            header = f"{'':>10}"
        row = " | ".join(f"{p:>8.4f}" for p in probs)
        print(f"{'':>10} | {row}" if t != temperatures[0] else None)

    for i, name in enumerate(token_names):
        row_parts = []
        for t in temperatures:
            scaled = apply_temperature(logits, t)
            probs = softmax(scaled)
            row_parts.append(f"{probs[i]:>8.4f}")
        print(f"{name:>10} | " + " | ".join(row_parts))

# A distribution that is initially spread with one clear favorite
logits = np.array([2.5, 1.0, 0.5, -0.5, -1.5])
temperature_effect(logits, [0.3, 0.7, 1.0, 1.5, 2.0])
```

### The Entropy Interpretation

Temperature directly controls the **entropy** of the output distribution:

$$H(P) = -\sum_v P(v) \log P(v)$$

- Low temperature → low entropy → predictable, concentrated output
- High temperature → high entropy → diverse, surprising output

For factual queries ("What is the capital of France?"), you want low temperature to get the correct answer reliably. For creative writing, higher temperature encourages exploration.

Typical values used in practice:
- Factual Q&A: $T = 0.2$ to $T = 0.5$
- Chat / general purpose: $T = 0.7$ to $T = 1.0$
- Creative writing: $T = 1.0$ to $T = 1.4$

**Key takeaway:** Temperature $T$ divides logits before softmax. $T < 1$ concentrates probability on the top tokens (safer, more deterministic). $T > 1$ spreads probability across many tokens (more creative, more chaotic). Temperature is a trade-off between accuracy and diversity.

---

## Section 3: Top-k Sampling

### The Problem with Pure Sampling

Even at reasonable temperatures, the model assigns nonzero probability to every token in the vocabulary — including completely nonsensical ones. At $T = 1.0$, there is a small chance of sampling any of the 32,000 tokens in the vocabulary, including very rare or contextually inappropriate ones. Occasionally, this produces jarring errors.

### Top-k Sampling

Top-k sampling restricts sampling to only the **$k$ most probable tokens** at each step, zeroing out all others before renormalizing.

Algorithm:
1. Compute logits and sort tokens by descending probability
2. Keep only the top-$k$ tokens
3. Set all other logit values to $-\infty$ (so they get zero probability after softmax)
4. Sample from the renormalized distribution

```python
import numpy as np
from scipy.special import softmax

def top_k_sampling(logits, k, temperature=1.0):
    """
    Apply top-k filtering and sample from the result.
    logits: (vocab_size,)
    k: number of top tokens to keep
    Returns: sampled token index
    """
    # Apply temperature
    scaled_logits = logits / temperature

    # Find the k-th largest logit value
    sorted_indices = np.argsort(scaled_logits)[::-1]  # descending
    top_k_indices = sorted_indices[:k]

    # Create filtered logits: -inf everywhere except top-k
    filtered_logits = np.full_like(scaled_logits, -np.inf)
    filtered_logits[top_k_indices] = scaled_logits[top_k_indices]

    # Softmax to get probabilities (only top-k tokens are nonzero)
    probs = softmax(filtered_logits)

    # Sample
    next_token = np.random.choice(len(probs), p=probs)
    return next_token, probs

# Example
np.random.seed(42)
vocab_size = 20
logits = np.random.randn(vocab_size)
k = 5

token, probs = top_k_sampling(logits, k=k)
print(f"Sampled token: {token}")
print(f"Non-zero probs: {np.sum(probs > 0)} (should be {k})")
print(f"Sum of probs: {probs.sum():.6f} (should be 1.0)")
```

### Choosing k

Common values: $k = 40$ to $k = 100$.

- Small $k$ (e.g., $k=5$): more focused, less diverse
- Large $k$ (e.g., $k=200$): approaches unconstrained sampling

**Limitation of top-k**: the number of reasonable next tokens varies wildly by context. After "The capital of France is," there may be only 1-2 reasonable completions ("Paris," "located"). After "I like," there could be hundreds. Using a fixed $k=40$ means sometimes including bad options (when there are really only 2 good ones) and sometimes excluding good options (when there are 100 good ones).

This motivates top-p sampling.

**Key takeaway:** Top-k sampling restricts each generation step to the $k$ most probable tokens. It prevents sampling from the very low-probability tail of the distribution. However, a fixed $k$ is context-insensitive.

---

## Section 4: Top-p (Nucleus) Sampling

### The Nucleus Idea

Instead of a fixed number of tokens, top-p sampling (Holtzman et al., 2020) keeps the **smallest set of tokens whose cumulative probability is at least $p$**.

This adapts dynamically: when the model is confident (one token has 90% probability), the nucleus is tiny. When the model is uncertain (probabilities spread across many tokens), the nucleus is large.

### Algorithm

1. Sort tokens by descending probability
2. Compute cumulative probability: $c_1 = p_1$, $c_2 = p_1 + p_2$, ...
3. Find the smallest set $S$ such that $\sum_{v \in S} p_v \geq p$
4. Zero out all tokens not in $S$, renormalize, sample

```python
import numpy as np
from scipy.special import softmax

def top_p_sampling(logits, p, temperature=1.0):
    """
    Apply nucleus (top-p) filtering and sample.
    logits: (vocab_size,)
    p: cumulative probability threshold (e.g., 0.9)
    Returns: sampled token index, filtered probability distribution
    """
    # Apply temperature
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)

    # Sort by descending probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)

    # Find the cutoff: indices where cumulative prob exceeds p
    # We keep tokens up to and including the one that pushes cumsum >= p
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    nucleus_indices = sorted_indices[:cutoff_idx]

    # Create filtered distribution
    filtered_probs = np.zeros_like(probs)
    filtered_probs[nucleus_indices] = probs[nucleus_indices]
    filtered_probs /= filtered_probs.sum()  # renormalize

    # Sample
    next_token = np.random.choice(len(filtered_probs), p=filtered_probs)
    return next_token, filtered_probs, len(nucleus_indices)

# Example: show how nucleus size adapts to distribution shape
np.random.seed(7)
vocab_size = 50

# Scenario 1: confident model (one dominant token)
logits_confident = np.full(vocab_size, -5.0)
logits_confident[3] = 5.0  # token 3 is very dominant

# Scenario 2: uncertain model (spread distribution)
logits_uncertain = np.random.randn(vocab_size)

for name, logits in [("Confident", logits_confident),
                      ("Uncertain", logits_uncertain)]:
    _, _, nucleus_size = top_p_sampling(logits, p=0.9)
    print(f"{name}: nucleus size = {nucleus_size} tokens for p=0.9")
```

### Top-p vs Top-k: Which to Use?

| Scenario | Top-k | Top-p |
|---|---|---|
| Confident model step | k tokens regardless | few tokens (small nucleus) |
| Uncertain model step | k tokens regardless | many tokens (large nucleus) |
| Adaptivity | None | Automatic |
| Common default | k=40 | p=0.9 or p=0.95 |

In practice, **top-p is generally preferred** because of its adaptivity. Many production systems combine them: apply top-k first (remove the very long tail), then apply top-p within the remaining tokens.

**Key takeaway:** Top-p (nucleus) sampling keeps the smallest set of tokens covering at least cumulative probability $p$. Unlike top-k, the nucleus size adapts automatically — small when the model is confident, large when uncertain. This is the most widely used sampling method in production LLMs.

---

## Section 5: Decoding Strategies Compared

### Greedy Decoding

At each step, choose the single most probable token:

$$x_t = \arg\max_v P(x_t = v \mid x_{<t})$$

- Fast, deterministic, reproducible
- Does not search over the space of sequences
- Often produces repetitive, dull, or suboptimal text
- A locally good choice can lead to a globally bad sequence

```python
def greedy_decode(logits_sequence):
    """
    logits_sequence: (T, vocab_size) — one set of logits per step
    Returns: sequence of token IDs
    """
    return [np.argmax(logits) for logits in logits_sequence]
```

### Beam Search

Beam search maintains $B$ partial sequences (the "beam") at each step, expanding each by the top $B$ candidates, and keeping the top $B$ total:

```python
def beam_search(model, prompt_ids, beam_width=5, max_len=20):
    """
    Conceptual beam search.
    model: callable, takes token list, returns logits (vocab_size,)
    Returns: best sequence found
    """
    from scipy.special import softmax

    # Initialize beam: list of (log_prob, sequence)
    beams = [(0.0, list(prompt_ids))]

    for step in range(max_len):
        candidates = []

        for log_prob, seq in beams:
            logits = model(seq)
            log_probs = np.log(softmax(logits) + 1e-10)

            # Expand this beam with all top-B next tokens
            top_B = np.argsort(log_probs)[-beam_width:]
            for token in top_B:
                candidates.append((log_prob + log_probs[token],
                                   seq + [token]))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Return the best sequence
    return beams[0][1]
```

**Beam search properties:**
- Approximate maximum-probability sequence (not exact — exponential search space)
- Good for machine translation (stable, high-quality output)
- Produces "safe," generic text; can be repetitive for open-ended generation
- Beam width $B = 4$ to $B = 10$ is typical

### Sampling (Temperature / Top-k / Top-p)

Stochastic sampling introduces randomness, producing more diverse and creative text.

**Properties:**
- Non-deterministic (different seed → different output)
- Can produce surprising, creative continuations
- Sometimes produces incoherent text
- Works well for creative writing, dialogue, storytelling

### Comparing Outputs (Schematic)

Given prompt: "The scientist looked at the data and"

| Method | Generated Text (Schematic) |
|---|---|
| Greedy | "said that the results were consistent with the previous findings. The data showed that the..." |
| Beam ($B=5$) | "concluded that the findings were significant. The results, which had been carefully analyzed,..." |
| Temp $T=0.7$ | "frowned. Something didn't add up. She had expected the numbers to match, but there it was..." |
| Temp $T=1.5$ | "grinned explosively at the rainbow-covered monkey statistics ..." |

**Key takeaway:** Greedy is fast but suboptimal. Beam search improves over greedy by maintaining multiple hypotheses but produces safe, repetitive text. Sampling introduces diversity and creativity at the cost of occasional incoherence. Production systems typically combine top-p sampling with temperature $T \approx 0.7$.

---

## Section 6: Repetition Penalty

### The Repetition Problem

Language models have a well-known tendency to get "stuck" in loops, repeating the same phrases:

*"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat..."*

This happens because high-probability tokens tend to remain high-probability in similar contexts — once the model generates a phrase, the context makes it likely to generate the same phrase again.

### The Repetition Penalty

One simple fix: penalize tokens that have already appeared in the generated text.

$$\text{logits}_v^{\text{adj}} = \begin{cases} \text{logits}_v / \theta & \text{if } v \text{ appeared in context} \\ \text{logits}_v & \text{otherwise} \end{cases}$$

where $\theta > 1$ is the penalty factor (typical value: 1.2 to 1.5).

For tokens with positive logits, dividing by $\theta > 1$ makes them smaller (less likely). For tokens with negative logits, dividing by $\theta$ makes them less negative (more likely) — an unintended side effect.

```python
import numpy as np
from scipy.special import softmax

def apply_repetition_penalty(logits, generated_ids, penalty=1.3):
    """
    Penalize tokens that have already been generated.
    logits: (vocab_size,) current logits
    generated_ids: set of token IDs already generated
    penalty: repetition penalty factor (> 1.0)
    Returns: modified logits
    """
    adjusted = logits.copy()
    for token_id in generated_ids:
        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty
    return adjusted

# Example
np.random.seed(42)
vocab_size = 10
logits = np.array([3.0, 1.5, 0.2, -0.5, 2.8, -1.0, 0.8, 1.2, -0.3, 2.1])
generated = {0, 4}  # tokens 0 and 4 have already been generated

original_probs = softmax(logits)
adjusted = apply_repetition_penalty(logits, generated, penalty=1.3)
new_probs = softmax(adjusted)

print(f"{'Token':>6} | {'Original':>10} | {'Penalized':>10} | {'Already seen':>12}")
for i in range(vocab_size):
    seen = "YES" if i in generated else ""
    print(f"{i:>6} | {original_probs[i]:>10.4f} | {new_probs[i]:>10.4f} | {seen:>12}")
```

### Frequency and Presence Penalties (OpenAI API Style)

The OpenAI API exposes two related parameters:
- **presence_penalty**: a flat additive penalty for any token that appeared at least once
- **frequency_penalty**: a penalty proportional to how many times the token appeared

$$\text{logits}_v^{\text{adj}} = \text{logits}_v - \alpha_P \cdot \mathbb{1}[v \in \text{context}] - \alpha_F \cdot \text{count}(v, \text{context})$$

where $\alpha_P$ and $\alpha_F$ are the presence and frequency penalty strengths.

**Key takeaway:** Repetition penalty reduces the probability of generating tokens that have already appeared in the context. It is a simple heuristic that significantly reduces loop-like repetition, at the cost of potentially suppressing legitimate repetition (e.g., proper nouns that should appear multiple times).

---

## Section 7: GPT Architecture and Scaling Laws

### The GPT Architecture

GPT is a stack of transformer decoder blocks (using only causal self-attention — no encoder or cross-attention):

```
Input Tokens
     ↓
Token Embeddings + Positional Embeddings
     ↓
[Causal Self-Attention Block] × L
     ↓
Layer Norm
     ↓
Linear (d_model → vocab_size)
     ↓
Softmax → Next Token Probabilities
```

Each causal self-attention block:
1. LayerNorm (pre-norm in modern models)
2. Causal Multi-Head Self-Attention (with upper-triangular mask)
3. Residual connection
4. LayerNorm
5. FFN (with GELU activation in GPT-2+)
6. Residual connection

Modern GPT-family models use **pre-norm** (LayerNorm before the sublayer) rather than post-norm (LayerNorm after), which improves training stability at large scales.

### Scaling Laws

Kaplan et al. (OpenAI, 2020) discovered empirical power law relationships between model performance and scale:

$$L(N) \propto N^{-0.076}, \quad L(D) \propto D^{-0.095}, \quad L(C) \propto C^{-0.050}$$

where $L$ is the test loss, $N$ is parameter count, $D$ is dataset size in tokens, and $C$ is total compute (FLOPs). The exponents are approximate.

**Key finding**: Performance improves smoothly and predictably as you scale up model size, data, and compute. The improvements follow power laws over many orders of magnitude.

```python
import numpy as np

def scaling_law_loss(N=None, D=None, C=None, base_loss=3.0):
    """
    Approximate scaling law predictions.
    Provide exactly one of N (params), D (tokens), or C (FLOPs).
    Returns approximate test loss.
    """
    if N is not None:
        # Kaplan et al. approximate exponent for parameters
        return base_loss * (1e8 / N) ** 0.076
    elif D is not None:
        # Approximate exponent for data
        return base_loss * (5e9 / D) ** 0.095
    elif C is not None:
        # Approximate exponent for compute
        return base_loss * (1e20 / C) ** 0.050
    raise ValueError("Provide N, D, or C")

model_sizes = [1e7, 1e8, 1e9, 1e10, 1e11]
print("Scaling with model size:")
for N in model_sizes:
    print(f"  N={N:.0e} params: loss ≈ {scaling_law_loss(N=N):.3f}")
```

### The Chinchilla Scaling Law

Hoffmann et al. (DeepMind, 2022) refined the Kaplan scaling laws and found the optimal compute allocation:

**For a given compute budget $C$, the optimal training uses roughly $N^* \approx \sqrt{C / 6}$ parameters and $D^* = 2 N^*$ training tokens.**

This means: for a given compute budget, you should train a smaller model for more steps (more data), rather than a larger model for fewer steps. This was contrary to the practice at the time (e.g., GPT-3 was undertrained relative to Chinchilla-optimal). The Llama model family was designed with Chinchilla-optimal data ratios.

**Key takeaway:** GPT is a decoder-only transformer trained with causal language modeling. Scaling laws predict smooth power-law improvements in loss as you increase parameters, data, and compute. Chinchilla optimal training suggests training smaller models on more data.

---

## Section 8: RLHF — Reinforcement Learning from Human Feedback

### The Alignment Problem

A raw pre-trained language model (even GPT-3) is a next-token predictor — it will continue any text in a plausible style, including harmful, biased, or factually wrong content. If trained on web text, the model has learned to produce all kinds of content that humans wrote online.

We want models that are:
- **Helpful**: answer questions, complete tasks
- **Harmless**: refuse to produce dangerous or offensive content
- **Honest**: acknowledge uncertainty, don't make up facts

These properties cannot be taught by next-token prediction alone — they require human value alignment. RLHF is the dominant method for achieving this alignment.

### Phase 1: Supervised Fine-Tuning (SFT)

Start with a pre-trained model and fine-tune it on a dataset of demonstration data: high-quality examples of the desired behavior, written by human contractors.

Examples:
```
User: Explain photosynthesis in simple terms.
Assistant: Photosynthesis is how plants make food using sunlight...

User: Write a Python function to reverse a string.
Assistant: def reverse_string(s): return s[::-1]
```

The SFT model learns to produce the style and format of a helpful assistant. But it is still limited — the demonstrations are finite, and the model may not generalize correctly.

### Phase 2: Reward Model Training

Collect **preference data**: for the same prompt, generate multiple responses (from the SFT model or other models), and have human raters rank them from best to worst.

Train a **reward model** $r_\phi(x, y)$ that predicts the human preference score for a response $y$ to prompt $x$.

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

where $y_w$ is the preferred ("winner") response and $y_l$ is the less preferred ("loser") response. This is the **Bradley-Terry model** for pairwise preference.

```python
import numpy as np
from scipy.special import expit  # sigmoid

def reward_model_loss(r_winner, r_loser):
    """
    Pairwise preference loss for reward model training.
    r_winner: scalar reward score for preferred response
    r_loser: scalar reward score for less-preferred response
    Returns: loss (should be minimized)
    """
    # We want r_winner > r_loser
    # Loss = -log(sigmoid(r_winner - r_loser))
    margin = r_winner - r_loser
    loss = -np.log(expit(margin) + 1e-10)
    return loss

# Example: reward model correctly ranks winner higher
print(f"Loss when winner > loser by 2.0: {reward_model_loss(1.0, -1.0):.4f}")
print(f"Loss when tie (0.0, 0.0):        {reward_model_loss(0.0, 0.0):.4f}")
print(f"Loss when winner < loser (bad):  {reward_model_loss(-1.0, 1.0):.4f}")
```

### Phase 3: PPO Fine-Tuning

Use the reward model to fine-tune the SFT model further with **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm.

The language model is treated as a **policy** $\pi_\theta(y \mid x)$ that generates responses. The reward is the reward model score $r_\phi(x, y)$.

The PPO objective maximizes:

$$\mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y)\right] - \beta \cdot \text{KL}\!\left[\pi_\theta(\cdot|x) \| \pi_{\text{SFT}}(\cdot|x)\right]$$

The KL penalty prevents the RL policy from drifting too far from the SFT model — without it, the model would find degenerate ways to "game" the reward model (e.g., gibberish that happens to score well).

```
SFT Model (frozen as reference)
       ↓
PPO update loop:
  1. Sample prompt x from dataset
  2. Generate response y with policy π_θ
  3. Score with reward model: r = r_φ(x, y)
  4. Compute KL penalty: KL(π_θ || π_SFT)
  5. Compute PPO advantage and update θ
  Repeat until convergence
```

### RLHF in Practice

RLHF is expensive and complex:
- Human preference data collection requires significant human labor
- Reward model training can introduce reward hacking
- PPO is unstable and sensitive to hyperparameters

Recent alternatives:
- **DPO (Direct Preference Optimization)**: skips the explicit reward model, directly optimizes preference data with a simpler objective
- **RLAIF (RL from AI Feedback)**: uses a strong AI model (Claude, GPT-4) instead of humans to provide preference labels

**Key takeaway:** RLHF involves three phases: (1) SFT on human-written demonstrations, (2) train a reward model from human preference comparisons, (3) fine-tune the SFT model with PPO to maximize the reward. This aligns the model with human values and produces the "helpful assistant" behavior.

---

## Section 9: The Modern LLM Landscape

### Major Models

The LLM landscape as of 2024-2025:

| Model | Organization | Params | Open? | Key Innovation |
|---|---|---|---|---|
| GPT-4 | OpenAI | Unknown (~1T est.) | No | Multimodal, strong reasoning |
| Claude 3 / 3.5 | Anthropic | Unknown | No | Constitutional AI, safety |
| Gemini 1.5 | Google | Unknown | No | 1M token context window |
| LLaMA 3 | Meta | 8B / 70B / 405B | Yes | Strong open-source base |
| Mistral 7B | Mistral | 7B | Yes | Sliding window attention |
| Qwen 2.5 | Alibaba | 7B-72B | Yes | Strong multilingual |

### Decoder-Only Dominance

Despite the original transformer having both encoder and decoder, modern large language models are almost all **decoder-only** (like GPT). Why?

1. Decoder-only models can be used for both understanding (via prompting) and generation natively.
2. Scaling works very well — decoder-only models benefit cleanly from more compute and data.
3. Unified training objective (CLM) is simple and works at scale.

Encoder-only models (BERT-family) remain popular for embedding tasks (search, similarity) but less common as general-purpose models.

### Context Windows

A critical practical parameter is the **context window** — how many tokens the model can process at once.

| Model | Context Window |
|---|---|
| GPT-2 | 1,024 tokens |
| GPT-3 | 4,096 tokens |
| GPT-4 | 128K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Claude 3.5 | 200K tokens |

Larger context windows are enabled by:
- Relative positional encodings (RoPE, ALiBi) that generalize beyond training length
- Efficient attention approximations (FlashAttention, grouped-query attention)
- Extended context fine-tuning on long documents

**Key takeaway:** Modern LLMs are almost exclusively decoder-only transformers. They vary enormously in size (7B to 1T+ parameters), openness (open-weight vs closed-source), and context window size (4K to 1M tokens). The field is advancing extremely rapidly.

---

## Section 10: Hallucination — What It Is and How to Mitigate It

### What Is Hallucination?

**Hallucination** refers to language models confidently generating factually incorrect information. The model does not "know" it is wrong — it is just predicting plausible-sounding text.

Examples of hallucinations:
- "Albert Einstein was born in 1985." (Wrong birth year)
- Citing a paper that doesn't exist, with a plausible-sounding title and authors
- Stating that a real person said something they never said
- Making up statistics that sound reasonable

### Why Does Hallucination Happen?

1. **Training objective mismatch**: the model is trained to predict the next token, not to be accurate. A convincing lie is just as likely as a true statement if both are common in training data.

2. **Knowledge cutoff**: the model's weights encode knowledge up to the training cutoff. It has no access to newer information.

3. **Interpolation in embedding space**: for rare facts, the model may "interpolate" between patterns it has seen, generating plausible-sounding but wrong combinations.

4. **No epistemic uncertainty modeling**: standard LM training does not explicitly train the model to say "I don't know." The model generates something regardless.

5. **RLHF can sometimes amplify hallucination**: if human raters prefer confident, fluent answers over hedged ones, the reward model may learn to reward confidence — including confident wrong answers.

### Mitigation Strategies

**1. Retrieval Augmented Generation (RAG)**

Instead of relying purely on the model's parametric memory, retrieve relevant documents at inference time and include them in the context:

```
System: Answer only based on the provided documents.
Documents: [Retrieved relevant text...]
User: What year was Einstein born?
```

The model has explicit source material to ground its answer. Hallucinations decrease dramatically when the answer is in the retrieved context.

**2. Chain-of-Thought Prompting**

Ask the model to "think step by step" before giving an answer. This forces explicit reasoning that can be checked:

```
Question: Is 17 × 23 = 391? Think step by step.
Answer: 17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391. Yes.
```

**3. Calibrated Uncertainty**

Fine-tune or prompt the model to express uncertainty:

```
If you are not confident, say "I'm not sure about this" rather than guessing.
```

**4. Factual Consistency Checking**

Use a separate model or retrieval system to verify claims in the generated output.

```python
def check_factual_consistency(claim, retrieved_passages):
    """
    Pseudocode: use an NLI model to check if the claim is
    entailed by the retrieved passages.
    Returns: 'entailed', 'contradicted', or 'not_enough_info'
    """
    # In practice: use a fine-tuned NLI model
    # nli_model(premise=retrieved_text, hypothesis=claim)
    for passage in retrieved_passages:
        if entails(passage, claim):
            return "entailed"
        elif contradicts(passage, claim):
            return "contradicted"
    return "not_enough_info"

def generate_with_verification(prompt, model, retriever, nli_model):
    """Pseudocode for RAG + NLI verification pipeline."""
    # Step 1: Retrieve relevant context
    passages = retriever.retrieve(prompt)
    context = "\n\n".join(passages)

    # Step 2: Generate response grounded in context
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    response = model.generate(augmented_prompt)

    # Step 3: Verify each claim in the response
    claims = extract_claims(response)
    verified = all(
        check_factual_consistency(c, passages) == "entailed"
        for c in claims
    )

    return response, verified
```

**5. Constrained Generation**

For structured outputs, use constrained decoding to ensure the output is valid (e.g., always valid JSON, always a date in the right format). This doesn't prevent factual hallucination but prevents format hallucination.

### The Fundamental Tension

There is an inherent tension: the properties that make LLMs fluent and helpful (predicting plausible completions from vast training data) are also the root cause of hallucination. Models that say "I don't know" frequently are less useful; models that never say it hallucinate more.

Mitigation is an active research area, and the best production systems typically combine RAG, uncertainty calibration, and human oversight for high-stakes use cases.

**Key takeaway:** Hallucination is the tendency of LLMs to generate confident but incorrect information. It arises from the next-token prediction objective, training data biases, and lack of explicit factual grounding. The most effective mitigations are RAG (ground responses in retrieved documents), chain-of-thought reasoning, and calibrated uncertainty expression.

---

## Section 11: Speculative Decoding and Efficient Inference

### The Generation Bottleneck

Autoregressive text generation is **serial** — you must generate token $t$ before you can generate $t+1$. This means that even on very fast hardware, generating 1,000 tokens requires 1,000 sequential forward passes through the model.

For a 7B-parameter model, each forward pass (with KV cache) takes ~10–30ms on a single GPU. Generating 1,000 tokens takes 10–30 seconds. For a 70B model, it's 100–300 seconds. This latency is a fundamental challenge for production LLM serving.

### Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) dramatically speeds up inference using a clever two-model trick:

1. **Draft model** (small, fast): generate $k$ candidate tokens speculatively in $k$ forward passes.
2. **Target model** (large, slow): verify all $k$ candidates in **one** forward pass.
3. **Accept or reject**: accept the candidate tokens that match what the target model would have generated. If token $i$ is rejected, sample a corrected token and discard $i+1, \ldots, k$.

```python
import numpy as np
from scipy.special import softmax

def speculative_decode_step(draft_tokens, draft_probs, target_probs):
    """
    One round of speculative decoding.
    draft_tokens: (k,) candidate tokens from draft model
    draft_probs: (k, vocab) probabilities from draft model
    target_probs: (k+1, vocab) probabilities from target model
                  (target runs one forward pass over all k+1 positions)
    Returns: accepted tokens, whether to sample a new token
    """
    accepted = []
    for i in range(len(draft_tokens)):
        t = draft_tokens[i]
        q = target_probs[i, t]   # target prob of draft token
        p = draft_probs[i, t]    # draft prob of draft token

        # Accept with probability min(1, q/p)
        ratio = min(1.0, q / (p + 1e-10))
        if np.random.random() < ratio:
            accepted.append(t)
        else:
            # Reject: sample from adjusted distribution
            adjusted = np.maximum(target_probs[i] - draft_probs[i], 0)
            adjusted /= adjusted.sum() + 1e-10
            new_token = np.random.choice(len(adjusted), p=adjusted)
            accepted.append(new_token)
            return accepted, False  # stop here

    # All accepted: sample one more from target's prediction
    bonus_token = np.random.choice(len(target_probs[-1]), p=target_probs[-1])
    accepted.append(bonus_token)
    return accepted, True

# The key insight: if the draft model is correct ~80% of the time
# and we speculate k=4 tokens per round, we get ~3.2 tokens per
# "expensive" target model call instead of 1.
# Speedup ≈ k * acceptance_rate / (1 + cost_ratio)

k = 5
acceptance_rate = 0.8
cost_ratio = 0.1  # draft model is 10% the cost of target
speedup = k * acceptance_rate / (1 + cost_ratio)
print(f"Expected tokens per target call: {k * acceptance_rate:.1f}")
print(f"Approximate speedup: {speedup:.1f}x")
```

The key property: speculative decoding produces **exactly** the same distribution as the target model alone — it is not an approximation. It is a sample-exact acceleration technique.

Typical speedups: 2–4× on standard text, with good small draft models. Draft models are often 7B for a 70B target, or a dedicated 1–2B model.

### Batched Inference and Throughput

For serving LLMs to many users simultaneously, throughput (tokens per second across all users) matters as much as latency (tokens per second per user).

**Continuous batching**: instead of waiting for all sequences in a batch to finish before starting new ones, dynamically swap completed sequences out and new requests in. This keeps GPU utilization high.

**Key numbers** (approximate, depending on hardware):
- A single A100 80GB GPU can serve a 7B model at ~2,000-5,000 tokens/sec for a batch of 32 requests.
- The same GPU serving one user at a time: ~200 tokens/sec.
- Batching provides ~10-25× throughput improvement.

### Chain-of-Thought Sampling

Recent work shows that generating longer reasoning chains before the answer dramatically improves accuracy on complex tasks. Instead of directly outputting the answer, prompt the model:

```
"Let's think step by step."
```

This leads to what is sometimes called "test-time compute scaling" — spending more inference-time compute (generating more tokens) to get better answers, analogously to how training-time scaling improves the model.

Models trained with RLHF specifically on chain-of-thought traces (like OpenAI's o1/o3 and DeepSeek-R1) show dramatic improvements on mathematical and logical reasoning.

```python
def chain_of_thought_prompt(question, few_shot_examples=None):
    """
    Format a question with chain-of-thought prompting.
    """
    cot_template = """Q: {question}
A: Let's think step by step.
{reasoning}
Therefore, the answer is: {answer}"""

    if few_shot_examples:
        examples_str = "\n\n".join([
            cot_template.format(**ex) for ex in few_shot_examples
        ])
        return f"{examples_str}\n\nQ: {question}\nA: Let's think step by step."

    return f"Q: {question}\nA: Let's think step by step."

# Example for math
examples = [
    {
        "question": "If a train travels 120 miles in 2 hours, what is its speed?",
        "reasoning": "Speed = distance / time. Distance = 120 miles. Time = 2 hours. Speed = 120/2 = 60.",
        "answer": "60 miles per hour."
    }
]
prompt = chain_of_thought_prompt(
    "If a train travels 180 miles in 3 hours, what is its speed?",
    few_shot_examples=examples
)
print(prompt)
```

**Key takeaway:** Speculative decoding uses a small draft model to generate candidate tokens verified by a large target model in one pass, achieving 2-4× speedups while preserving exact output distribution. Chain-of-thought sampling shows that allocating more generation steps to reasoning (not just the final answer) significantly improves accuracy on complex tasks.

---

## Section 12: Evaluation of Generated Text

Evaluating generated text is fundamentally harder than evaluating classification — there is no single correct answer for "write a poem about the ocean."

### Automatic Metrics

**BLEU (Bilingual Evaluation Understudy)**: measures n-gram overlap between generated text and reference translations. Used for machine translation.

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is the precision of $n$-grams (fraction of generated $n$-grams that appear in the reference) and BP is a brevity penalty (penalizes very short outputs).

```python
import numpy as np
from collections import Counter

def ngrams(tokens, n):
    """Extract all n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(candidate, reference, max_n=4):
    """
    Compute BLEU-4 score.
    candidate: list of tokens (generated)
    reference: list of tokens (ground truth)
    """
    candidate = list(candidate)
    reference = list(reference)

    # Brevity penalty
    c = len(candidate)
    r = len(reference)
    bp = 1.0 if c >= r else np.exp(1 - r/c)

    log_precision_sum = 0.0
    weights = [1/max_n] * max_n

    for n in range(1, max_n + 1):
        cand_ngrams = Counter(ngrams(candidate, n))
        ref_ngrams = Counter(ngrams(reference, n))

        # Clipped count: can't exceed reference count
        matches = sum(min(count, ref_ngrams[gram])
                      for gram, count in cand_ngrams.items())

        total = sum(cand_ngrams.values())
        precision = matches / total if total > 0 else 0.0

        if precision > 0:
            log_precision_sum += weights[n-1] * np.log(precision)
        else:
            return 0.0  # if any n-gram precision is 0, BLEU is 0

    return bp * np.exp(log_precision_sum)

# Example
ref   = "the cat sat on the mat".split()
cand1 = "the cat sat on the mat".split()    # perfect
cand2 = "a dog lay on the floor".split()   # poor
cand3 = "the cat sat on the floor".split() # close

for name, cand in [("perfect", cand1), ("poor", cand2), ("close", cand3)]:
    score = bleu_score(cand, ref)
    print(f"BLEU ({name}): {score:.4f}")
```

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: measures n-gram recall, used for summarization.

**BERTScore**: embeds both generated and reference text with BERT and measures cosine similarity. Correlates better with human judgments than n-gram metrics.

**Perplexity**: for language models themselves, perplexity measures how surprised the model is by held-out text. A model with perplexity $P$ is as confused as if it had to choose uniformly among $P$ equally-likely options at each step:

$$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\right)$$

```python
import numpy as np
from scipy.special import softmax

def perplexity(logits_list, token_ids):
    """
    Compute perplexity of a token sequence under the model.
    logits_list: list of (vocab_size,) arrays, one per step
    token_ids: list of actual token IDs (one per step)
    """
    log_probs = []
    for logits, token in zip(logits_list, token_ids):
        probs = softmax(logits)
        log_probs.append(np.log(probs[token] + 1e-10))

    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

# Lower perplexity = model assigns higher probability to the text
# GPT-2 on Penn Treebank: ~35. GPT-3: ~20. LLaMA 2 70B: ~5-8.
np.random.seed(42)
# Simulate a model that's quite confident about 5 tokens
vocab = 100
logits = [np.random.randn(vocab) for _ in range(5)]
tokens = [np.argmax(softmax(l)) for l in logits]  # greedy tokens
ppl_greedy = perplexity(logits, tokens)
print(f"Perplexity on greedy tokens: {ppl_greedy:.2f}")  # should be low

# Random tokens have high perplexity
tokens_random = [np.random.randint(vocab) for _ in range(5)]
ppl_random = perplexity(logits, tokens_random)
print(f"Perplexity on random tokens: {ppl_random:.2f}")  # should be high
```

### Human Evaluation

For open-ended generation, human evaluation remains the gold standard:
- **Preference studies**: show two model outputs to humans, ask which is better
- **Likert scale ratings**: rate output on dimensions (coherence, factuality, helpfulness) from 1-5
- **Chatbot Arena**: crowd-sourced comparisons where users vote for preferred model responses

Automatic metrics like BLEU and ROUGE correlate poorly with human preferences for open-ended generation. BERTScore and model-based evaluators (using a strong LLM as the judge, "LLM-as-a-judge") correlate better.

**Key takeaway:** BLEU measures n-gram precision (translation), ROUGE measures n-gram recall (summarization), perplexity measures how well the model predicts a fixed test set, and BERTScore uses embedding similarity for semantic matching. For open-ended generation, human evaluation or LLM-as-a-judge remains the most reliable signal.

---

## Summary

This lesson covered the complete picture of text generation with modern LLMs:

1. **The generation loop**: autoregressive next-token prediction. Logits → softmax → sample/select → append → repeat.

2. **Temperature**: divides logits before softmax. Low $T$ → confident, repetitive; high $T$ → creative, chaotic. Typical range: 0.5–1.2.

3. **Top-k sampling**: restrict to $k$ most probable tokens. Fixed cutoff, context-insensitive.

4. **Top-p sampling**: restrict to smallest set covering $\geq p$ cumulative probability. Adapts to the model's confidence. Most widely used. Typical $p = 0.9$.

5. **Decoding strategies**: greedy (fast, dull), beam search (quality, safe), sampling (diverse, creative). Production systems typically use temperature + top-p sampling.

6. **Repetition penalty**: divide logits of already-generated tokens by a penalty factor $\theta > 1$ to reduce loops.

7. **Scaling laws**: loss improves as power law with parameters ($N$), data ($D$), and compute ($C$). Chinchilla optimal: smaller model + more data beats larger model + less data.

8. **RLHF**: SFT → reward model training (pairwise preferences) → PPO. Aligns raw LMs with human values. DPO and RLAIF are modern variants.

9. **LLM landscape**: decoder-only dominates. Models range from 7B to 1T+ params, 4K to 1M token context windows. Open-source (LLaMA, Mistral) and closed-source (GPT-4, Claude, Gemini) ecosystems.

10. **Hallucination**: models confidently generate false information. Root cause: next-token prediction doesn't reward accuracy. Main mitigations: RAG, chain-of-thought, calibrated uncertainty.
