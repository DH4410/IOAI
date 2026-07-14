---
title: Fine-Tuning Language Models
track: nlp
order: 6
estimatedTime: 55
difficulty: advanced
---

# Fine-Tuning Language Models

Pre-trained language models like BERT and GPT are general-purpose — they understand language deeply but are not specialized for any particular task. Fine-tuning is the process of adapting a pre-trained model to a specific task. This lesson covers the spectrum from full fine-tuning (updating every weight) to parameter-efficient methods (updating only a tiny fraction of weights), with concrete guidance on when to use each.

---

## Section 1: Why Fine-Tune? Transfer Learning in NLP

### The Transfer Learning Hypothesis

A pre-trained language model has encoded an enormous amount of knowledge:
- Syntactic structure (subject-verb agreement, phrase boundaries)
- Semantic relationships (synonymy, analogy, entailment)
- World knowledge (capitals, historical facts, scientific concepts)
- Discourse structure (topic coherence, argument organization)

The hypothesis: this knowledge, encoded in the model's weights, will transfer to downstream tasks. Instead of training a new model for sentiment analysis, named entity recognition, or question answering, we start from the pre-trained weights and adapt.

The empirical result: this works spectacularly well. BERT fine-tuned for 3 epochs on the Stanford Sentiment Treebank (a 67K sentence dataset) achieves state-of-the-art performance. Training an LSTM from scratch on the same data achieves substantially worse results.

### The Landscape of Adaptation Strategies

There is a spectrum of how much of the model you update:

```
Full Fine-Tuning ←────────────────────────────────→ Prompting
(update all weights)                           (update nothing)

Full FT → Feature Extraction → Adapters → LoRA → Prefix Tuning → Prompt Tuning
          (freeze backbone)   (add small  (add   (add learned   (soft
                               modules)   low-   prefix tokens)  prompts)
                                          rank
                                          updates)
```

The trend in recent years has been toward the right side of this spectrum. With models reaching 70B parameters, updating all weights is often impractical, and parameter-efficient methods can match full fine-tuning quality at a fraction of the cost.

### When You Need Fine-Tuning vs When You Don't

Before fine-tuning, always ask: can prompting alone solve the problem?

- **Prompting works well**: text classification with a few examples, summarization, translation.
- **Fine-tuning is necessary**: precise structured output (NER, slot filling), tasks where consistency and format control matter, low-latency inference, tasks requiring deep domain knowledge.

**Key takeaway:** Pre-trained models encode general language knowledge that transfers to downstream tasks. Fine-tuning adapts this knowledge with task-specific data. The spectrum runs from full fine-tuning to zero-shot prompting, with parameter-efficient methods offering a middle ground.

---

## Section 2: Full Fine-Tuning

### What It Is

Full fine-tuning: take a pre-trained model, add a task-specific output head, and train **all parameters** on labeled task data.

For a BERT-based classifier:
1. Take BERT-Base (110M parameters)
2. Add a linear layer: $W \in \mathbb{R}^{768 \times C}$ (where $C$ is the number of classes)
3. Feed the `[CLS]` token representation through this layer → logits
4. Train with cross-entropy loss on labeled examples
5. Backpropagate through the **entire** model — all 110M params get updated

```python
import numpy as np
from scipy.special import softmax

class FineTunedBERT:
    """
    Conceptual BERT fine-tuning for classification.
    In practice, use transformers.BertForSequenceClassification.
    """
    def __init__(self, pretrained_bert, num_classes, d_model=768):
        self.bert = pretrained_bert  # pre-trained encoder
        self.num_classes = num_classes
        # Task head: randomly initialized
        self.W_head = np.random.randn(d_model, num_classes) * 0.01
        self.b_head = np.zeros(num_classes)

    def forward(self, input_ids, attention_mask):
        # Run the full BERT encoder
        hidden_states = self.bert(input_ids, attention_mask)

        # Take [CLS] token representation (position 0)
        cls_repr = hidden_states[:, 0, :]  # (batch, d_model)

        # Linear classification head
        logits = cls_repr @ self.W_head + self.b_head  # (batch, num_classes)
        return logits

    def predict(self, logits):
        return softmax(logits, axis=-1).argmax(axis=-1)

def cross_entropy_loss(logits, labels):
    """
    logits: (batch, C)
    labels: (batch,) integer class indices
    """
    batch_size = logits.shape[0]
    probs = softmax(logits, axis=-1)
    correct_probs = probs[np.arange(batch_size), labels]
    return -np.mean(np.log(correct_probs + 1e-10))

# Example
batch_size, d_model, num_classes = 4, 768, 3
np.random.seed(42)
logits = np.random.randn(batch_size, num_classes)
labels = np.array([0, 1, 2, 1])
loss = cross_entropy_loss(logits, labels)
print(f"Cross-entropy loss: {loss:.4f}")
```

### Practical Full Fine-Tuning Process

1. Load pre-trained weights from a checkpoint (e.g., `bert-base-uncased`)
2. Add the task head (classification, token labeling, span extraction, etc.)
3. Prepare data: tokenize, create DataLoader with batching and shuffling
4. Train for 2-5 epochs with a small learning rate (1e-5 to 5e-5)
5. Evaluate on a validation set; use the best checkpoint
6. Evaluate final performance on test set (only once!)

### Advantages and Disadvantages

**Advantages:**
- Maximum expressiveness: every parameter can adapt to the task
- Generally achieves the best performance (given sufficient data)
- Well-understood, reliable

**Disadvantages:**
- High memory: must store model + gradients + optimizer state (3× model size for Adam)
- High compute: takes hours to days for large models
- Each task requires a separate fine-tuned checkpoint (storage)
- Risk of overfitting on small datasets

**Key takeaway:** Full fine-tuning updates all model weights and generally achieves the highest task performance. It requires significant memory and compute, and each task needs its own model checkpoint.

---

## Section 3: Feature Extraction — Freezing the Backbone

### The Idea

Instead of fine-tuning all parameters, treat the pre-trained model as a **fixed feature extractor**:

1. Freeze the pre-trained backbone (no gradient updates to it)
2. Pass your data through the frozen model to get contextualized embeddings
3. Train only the task-specific head (a much smaller model) on top

```python
import numpy as np
from scipy.special import softmax

def extract_bert_features(bert_model, texts):
    """
    Extract [CLS] embeddings from a FROZEN BERT model.
    bert_model: callable, returns (batch, seq_len, d_model)
    texts: list of preprocessed token sequences
    Returns: (N, d_model) feature matrix
    """
    features = []
    for text in texts:
        with_no_grad:  # In practice: torch.no_grad()
            hidden = bert_model(text)  # (1, seq_len, 768)
            cls_feat = hidden[0, 0, :]  # (768,)
        features.append(cls_feat)
    return np.stack(features)  # (N, 768)

# After extracting features, train a simple classifier
class LinearClassifier:
    def __init__(self, input_dim, num_classes, lr=0.01):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.lr = lr

    def forward(self, X):
        return X @ self.W + self.b

    def train_step(self, X, y):
        logits = self.forward(X)
        probs = softmax(logits, axis=-1)
        batch_size = X.shape[0]

        # One-hot encode y
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1.0

        # Gradient of cross-entropy loss w.r.t. logits
        dlogits = (probs - y_onehot) / batch_size

        # Gradients
        dW = X.T @ dlogits
        db = dlogits.sum(axis=0)

        # Update
        self.W -= self.lr * dW
        self.b -= self.lr * db

        loss = -np.mean(np.log(probs[np.arange(batch_size), y] + 1e-10))
        return loss
```

### When Feature Extraction Works Well

- Very small labeled datasets (< 500 examples) — too few samples to update 110M params
- Fast iteration: extract features once (store to disk), train head in seconds
- Resource constraints: no GPU available for BERT fine-tuning
- Multiple tasks: one set of features, train many cheap heads

### When Feature Extraction Falls Short

- The task domain is very different from pre-training data (medical text, legal text, code)
- The task requires nuanced understanding that benefits from end-to-end optimization
- Generally: full fine-tuning (or LoRA) will outperform feature extraction when resources allow

**Key takeaway:** Feature extraction freezes the pre-trained backbone and trains only a small head. It is fast, cheap, and works well with very small datasets but typically underperforms full fine-tuning when enough data and compute are available.

---

## Section 4: LoRA — Low-Rank Adaptation

LoRA (Hu et al., 2021) is the most widely adopted parameter-efficient fine-tuning (PEFT) method. It achieves near full fine-tuning quality while updating fewer than 1% of model parameters.

### The Intuition

Pre-trained weight matrices $W_0 \in \mathbb{R}^{d \times k}$ encode general knowledge. During fine-tuning, the change $\Delta W$ needed to adapt to a specific task is hypothesized to have **low intrinsic rank** — the effective number of "degrees of freedom" needed is much smaller than $d \times k$.

If $\Delta W$ has rank $r \ll \min(d, k)$, we can decompose it as:

$$\Delta W = B A$$

where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$.

The forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

During training, $W_0$ is frozen. Only $A$ and $B$ are updated.

### Parameter Count

Original: $d \times k$ parameters.
LoRA: $r \times k + d \times r = r(d + k)$ parameters.

For BERT-Base with $d = k = 768$ and rank $r = 8$:
- Original per-layer: $768 \times 768 = 589{,}824$ parameters
- LoRA per-layer: $8 \times (768 + 768) = 12{,}288$ parameters
- **Ratio: 2% of original**

Applied to all attention projection matrices in all 12 layers:
- Original trainable parameters: $4 \times 12 \times 589{,}824 \approx 28M$
- LoRA trainable parameters: $4 \times 12 \times 12{,}288 \approx 590K$ — about 50× fewer.

### Initialization

- $A$ is initialized with random Gaussian values
- $B$ is initialized to **zeros**

This ensures $\Delta W = BA = 0$ at the start of training, so the model begins from the pre-trained state (the LoRA modification starts as a no-op).

### The Scaling Factor

In practice, LoRA scales the output by $\alpha / r$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $\alpha$ is a hyperparameter (commonly set to the same value as $r$). This makes the learning rate insensitive to the choice of $r$.

```python
import numpy as np

class LoRALinear:
    """
    A linear layer with LoRA adaptation.
    W_0 is frozen; A and B are trainable.
    """
    def __init__(self, d_in, d_out, rank=8, alpha=8, lr=1e-4, seed=42):
        rng = np.random.RandomState(seed)

        # Frozen pre-trained weight (simulated)
        self.W_0 = rng.randn(d_in, d_out) / np.sqrt(d_in)

        # LoRA matrices
        self.A = rng.randn(d_in, rank) * 0.01   # (d_in, r)
        self.B = np.zeros((rank, d_out))          # (r, d_out)

        self.rank = rank
        self.scale = alpha / rank
        self.lr = lr

    def forward(self, x):
        """
        x: (batch, d_in)
        Returns: (batch, d_out)
        """
        # Frozen pretrained path
        base_out = x @ self.W_0

        # LoRA adaptation (B @ A applied to x)
        # x @ A: (batch, rank), then @ B: (batch, d_out)
        lora_out = (x @ self.A) @ self.B * self.scale

        return base_out + lora_out

    def lora_params(self):
        """Return only the trainable LoRA parameters."""
        return self.A, self.B

    def merge_weights(self):
        """
        Merge LoRA into W_0 for efficient inference.
        After merging, LoRA overhead is zero at inference time.
        """
        delta_W = self.A @ self.B * self.scale
        merged = self.W_0 + delta_W
        return merged

# Example
np.random.seed(0)
layer = LoRALinear(d_in=768, d_out=768, rank=8, alpha=16)
x = np.random.randn(4, 768)  # batch of 4
out = layer.forward(x)
print("Output shape:", out.shape)  # (4, 768)

# Count parameters
total_params = 768 * 768
lora_params = 768 * 8 + 8 * 768
print(f"Original params: {total_params:,}")
print(f"LoRA params:     {lora_params:,}")
print(f"Ratio:           {lora_params/total_params:.2%}")
```

### Where to Apply LoRA

LoRA is typically applied to the query and value projection matrices in each attention layer ($W^Q$ and $W^V$). The original paper found that applying LoRA to more matrices (also $W^K$ and $W^O$) with the same total parameter budget is often better than using a higher rank on just $Q$ and $V$.

### Merging for Inference

After training, $B$ and $A$ can be merged into $W_0$: $W_{\text{merged}} = W_0 + BA \cdot \alpha/r$. The merged model has the same architecture as the original — no overhead at inference time. This is a key advantage over methods that add new modules.

### Choosing the Rank $r$

- $r = 4$ to $r = 16$: works well for most NLP tasks
- $r = 64$ or higher: rarely needed, may not outperform lower ranks
- Sweeping $r$ is a reasonable hyperparameter search

**Key takeaway:** LoRA trains low-rank matrices $A$ and $B$ representing $\Delta W = BA$ while freezing $W_0$. With rank $r \approx 8$, LoRA trains <1% of the parameters while achieving performance close to full fine-tuning. Weights can be merged for zero-overhead inference.

---

## Section 5: Prompt Tuning and Instruction Tuning

### Soft Prompt Tuning

Prompt tuning (Lester et al., 2021) keeps **all model weights frozen** and learns a small set of "soft prompt" token embeddings prepended to the input.

```
[P1][P2][P3][P4] The movie was great. → Positive
     ^learned^
```

The $P_i$ are continuous vectors (not real words) trained by gradient descent. The model's weights are completely frozen — only these prompt vectors are updated.

- **Parameters**: $p \times d_{\text{model}}$ where $p$ is the number of prompt tokens (typically 10-100)
- For BERT-Base: $100 \times 768 = 76{,}800$ parameters — about 0.07% of the model

At large scales (T5-XXL, 11B parameters), prompt tuning nearly matches full fine-tuning. At smaller scales, the gap is larger.

### Prefix Tuning

A more expressive variant: prepend learned vectors to the **keys and values** in every attention layer, not just the input. This allows the soft prompt to influence every layer's attention.

### Instruction Tuning

Instruction tuning is different from soft prompt tuning: it is a form of supervised fine-tuning where the examples are formatted as natural language instructions.

```
Instruction: Classify the sentiment of the following text.
Text: The movie was absolutely fantastic.
Answer: Positive
```

By training on thousands of such instruction-following examples across many tasks, the model learns to follow instructions, greatly improving zero-shot performance on new tasks. FLAN (Google, 2021), InstructGPT (OpenAI, 2022), and Alpaca (Stanford, 2023) all use variants of instruction tuning.

```python
# Instruction format for fine-tuning
def format_classification_example(text, label, label_names):
    """Format a single example for instruction tuning."""
    instruction = "Classify the sentiment of the following movie review."
    options = " | ".join([f"{i+1}. {name}" for i, name in enumerate(label_names)])
    return {
        "input": f"Instruction: {instruction}\nOptions: {options}\nText: {text}\nAnswer:",
        "output": label_names[label]
    }

examples = [
    ("The film was breathtaking.", 1, ["Negative", "Positive"]),
    ("A complete waste of time.", 0, ["Negative", "Positive"]),
]

for text, label, names in examples:
    ex = format_classification_example(text, label, names)
    print(ex["input"])
    print("→", ex["output"])
    print()
```

**Key takeaway:** Soft prompt tuning learns only a small set of continuous prompt vectors; all model weights are frozen. Instruction tuning fine-tunes on natural language instruction-following examples across many tasks, dramatically improving zero-shot generalization.

---

## Section 6: Catastrophic Forgetting and How to Mitigate It

### What Is Catastrophic Forgetting?

When you fine-tune a pre-trained model on a new task, the gradient updates push the weights toward that task. This can **overwrite** the knowledge from pre-training or from previously learned tasks. The model "forgets" what it knew before.

For example: fine-tune BERT for sentiment analysis on movie reviews. Then fine-tune the same checkpoint on medical text classification. The resulting model might perform well on medical text but badly on movie reviews. The medical training overwrote representations useful for movies.

```
Initial State → Fine-tune Task A → Fine-tune Task B → Evaluate Task A
Pre-trained    (sentiment)         (medical NER)       → Catastrophic drop!
```

### Symptoms

- Performance on task A drops dramatically after fine-tuning on task B
- Performance on general language understanding (GLUE benchmark) drops after strong task-specific fine-tuning
- The model "unlearns" syntactic or factual knowledge encoded during pre-training

### Mitigation Strategy 1: Regularization (L2 toward Pre-Trained Weights)

Instead of unconstrained optimization, add a term penalizing deviation from the pre-trained weights $\theta_0$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \|\theta - \theta_0\|_2^2$$

This encourages the model to stay close to the pre-trained weights while still fitting the task. A higher $\lambda$ preserves more pre-training knowledge.

```python
def fine_tuning_loss_with_l2(task_loss, params, pretrained_params, lambda_reg=0.01):
    """
    Task loss + L2 regularization toward pre-trained weights.
    task_loss: scalar loss from task-specific data
    params: current model parameters (list of arrays)
    pretrained_params: original pre-trained parameters (frozen)
    lambda_reg: regularization strength
    """
    reg_loss = 0.0
    for p, p0 in zip(params, pretrained_params):
        reg_loss += np.sum((p - p0) ** 2)
    return task_loss + lambda_reg * reg_loss
```

### Mitigation Strategy 2: Small Learning Rate

Using a very small learning rate (1e-5 rather than 1e-3) limits how far weights move from the pre-trained initialization. This is the single most important practical mitigation.

### Mitigation Strategy 3: Linear Learning Rate Warmup + Decay

A typical schedule:
1. Warmup: learning rate increases linearly from 0 to max LR over the first 5-10% of training steps
2. Decay: learning rate decreases (linearly or with cosine schedule) to near 0

```python
def get_lr_schedule(step, total_steps, max_lr=2e-5, warmup_fraction=0.06):
    """
    Warmup + linear decay schedule.
    Returns learning rate for given step.
    """
    warmup_steps = int(total_steps * warmup_fraction)

    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Linear decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * (1.0 - progress)

# Plot the schedule
total_steps = 1000
lrs = [get_lr_schedule(s, total_steps) for s in range(total_steps)]
print(f"Max LR at step {int(total_steps*0.06)}: {max(lrs):.2e}")
print(f"LR at step 0: {lrs[0]:.2e}")
print(f"LR at step {total_steps-1}: {lrs[-1]:.2e}")
```

### Mitigation Strategy 4: Multi-Task Fine-Tuning

Instead of sequentially fine-tuning on each task, fine-tune on all tasks simultaneously. This forces the model to maintain representations useful for all tasks and prevents any single task from dominating.

This is essentially how instruction tuning works — a single fine-tuning stage over many diverse tasks.

### Mitigation Strategy 5: Parameter-Efficient Methods (LoRA, Adapters)

If you use LoRA or adapters, the original weights are never modified. You can train many separate LoRA adapters for different tasks and swap them at inference time. Catastrophic forgetting becomes irrelevant because the base model is always intact.

**Key takeaway:** Catastrophic forgetting occurs when fine-tuning overwrites pre-trained knowledge. Mitigations include: small learning rates, L2 regularization toward pre-trained weights, warmup schedules, multi-task training, and parameter-efficient methods that leave base weights intact.

---

## Section 7: Fine-Tuning Datasets and Task Formats

### Classification Tasks

Input: a text (or pair of texts). Output: a discrete label.

Examples:
- **SST-2** (Stanford Sentiment Treebank): 67K movie review sentences, 2-class sentiment
- **MNLI** (Multi-Genre NLI): 393K sentence pairs for natural language inference (entailment/contradiction/neutral)
- **CoLA** (Corpus of Linguistic Acceptability): grammaticality judgments

Format: `[CLS] text [SEP]` → head on `[CLS]` → cross-entropy loss

### Named Entity Recognition (NER)

Input: a sentence. Output: a label per token (e.g., PER, ORG, LOC, O for no entity).

```
Barack Obama visited Paris on Monday.
B-PER I-PER O       B-LOC O  O
```

Format: `[CLS] Barack Obama visited Paris on Monday [SEP]`
         →  labels for each token (using BIO or BIOES tagging)

Fine-tuning: linear head on each token representation, predict class per token.

Challenge: WordPiece/BPE may split one word into multiple tokens. Standard approach: take the first subword token of each word and ignore the rest for labeling.

```python
def align_labels_with_tokens(words, labels, tokenizer, label2id):
    """
    Align word-level NER labels with subword tokens.
    Returns: input_ids, label_ids (with -100 for non-first subwords)
    """
    input_ids = []
    label_ids = []

    for word, label in zip(words, labels):
        # Tokenize the word into subwords
        word_tokens = tokenizer(word)  # e.g., "Washington" → ["wash", "##ington"]

        if not word_tokens:
            continue

        input_ids.extend(word_tokens)
        # First subword gets the real label
        label_ids.append(label2id[label])
        # Subsequent subwords get -100 (ignored in loss)
        label_ids.extend([-100] * (len(word_tokens) - 1))

    return input_ids, label_ids

# Example (conceptual — using integer token IDs as proxies)
words = ["Paris", "is", "beautiful"]
labels = ["B-LOC", "O", "O"]
label2id = {"B-LOC": 1, "I-LOC": 2, "O": 0}

# Simulate tokenizer output (Paris → [par, ##is])
def mock_tokenizer(word):
    splits = {"Paris": [1001, 1002], "is": [1003], "beautiful": [1004, 1005, 1006]}
    return splits.get(word, [9999])

ids, lab_ids = align_labels_with_tokens(words, labels, mock_tokenizer, label2id)
print("Token ids:", ids)
print("Label ids:", lab_ids)
# Token ids: [1001, 1002, 1003, 1004, 1005, 1006]
# Label ids: [1, -100, 0, 0, -100, -100]
```

### Extractive Question Answering

Input: a passage and a question. Output: start and end positions of the answer span within the passage.

Dataset: **SQuAD** (Stanford Question Answering Dataset) — 100K question-passage-answer triples.

Format:
```
[CLS] question [SEP] passage tokens [SEP]
```
Two separate linear heads predict the start and end token position.

$$\text{score\_start}_i = h_i \cdot w_{\text{start}}, \quad \text{score\_end}_j = h_j \cdot w_{\text{end}}$$
$$P(\text{start}=i) = \text{softmax}(\text{score\_start})_i$$

At inference: find the span $[i, j]$ ($i \leq j$) maximizing $P(\text{start}=i) \cdot P(\text{end}=j)$.

**Key takeaway:** Different tasks require different output heads and label formats. Classification uses a head on `[CLS]`; NER uses per-token heads; QA uses span-prediction heads. The backbone is the same; only the head and data format change.

---

## Section 8: Practical Tips and the Hugging Face Trainer API

### Critical Hyperparameters

| Hyperparameter | Recommended Range | Notes |
|---|---|---|
| Learning rate | 1e-5 to 5e-5 | Much lower than training from scratch |
| Batch size | 16 to 32 | Larger batches need LR scaling |
| Epochs | 2 to 5 | Too many epochs → overfitting |
| Warmup steps | 6% of total steps | Typical |
| Weight decay | 0.01 | Applied to non-bias, non-LN params |
| Gradient clipping | 1.0 | Prevents gradient explosion |

**The most important rule**: Use a **small learning rate**. The default LR for training from scratch (1e-3, 1e-4) will destroy the pre-trained representations. For BERT fine-tuning, 2e-5 is a solid starting point.

### Overfitting on Small Datasets

With fewer than ~1,000 training examples, full fine-tuning often overfits:
- Use feature extraction instead
- Or use LoRA with rank 4-8
- Or use heavy regularization (large weight decay, dropout)
- Or do data augmentation (back-translation, synonym replacement)

### The PEFT Library

Hugging Face's `peft` library provides LoRA, prefix tuning, and other PEFT methods for any transformer model in 5 lines:

```python
# Pseudocode (requires peft library)
# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2
# )

# lora_config = LoraConfig(
#     r=8,           # rank
#     lora_alpha=16, # scaling factor
#     target_modules=["query", "value"],  # which layers to adapt
#     lora_dropout=0.1,
#     bias="none",
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
# Trainable params: 296,448 || All params: 109,782,276 || Trainable%: 0.27%
```

### The Hugging Face Trainer (Conceptual Overview)

```python
# Pseudocode for the Hugging Face Trainer API
# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_ratio=0.06,
#     weight_decay=0.01,
#     learning_rate=2e-5,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics,  # your evaluation function
# )

# trainer.train()
# trainer.evaluate()
```

Key things the Trainer handles automatically:
- Device management (CPU/GPU/multi-GPU)
- Gradient accumulation for large effective batch sizes
- Gradient clipping
- LR scheduling with warmup
- Mixed precision training (fp16/bf16)
- Checkpoint saving and loading the best model

### A Minimal Training Loop (Pure NumPy, Conceptual)

```python
import numpy as np
from scipy.special import softmax

def simple_finetune_loop(X_train, y_train, X_val, y_val,
                          d_model=768, num_classes=2,
                          lr=2e-5, epochs=3, batch_size=16):
    """
    Simplified fine-tuning loop for a linear head over frozen BERT features.
    X_train: (N, d_model) pre-extracted features
    y_train: (N,) integer labels
    """
    np.random.seed(42)
    W = np.random.randn(d_model, num_classes) * 0.01
    b = np.zeros(num_classes)

    N = X_train.shape[0]
    history = []

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(N)
        X_train, y_train = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        steps = 0

        for start in range(0, N, batch_size):
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            bs = X_batch.shape[0]

            # Forward
            logits = X_batch @ W + b  # (bs, C)
            probs = softmax(logits, axis=-1)

            # Loss
            loss = -np.mean(np.log(probs[np.arange(bs), y_batch] + 1e-10))
            epoch_loss += loss
            steps += 1

            # Gradient (cross-entropy w.r.t. logits)
            dlogits = (probs.copy())
            dlogits[np.arange(bs), y_batch] -= 1
            dlogits /= bs

            dW = X_batch.T @ dlogits
            db = dlogits.sum(axis=0)

            W -= lr * dW
            b -= lr * db

        # Validation accuracy
        val_logits = X_val @ W + b
        val_preds = val_logits.argmax(axis=-1)
        val_acc = (val_preds == y_val).mean()

        avg_loss = epoch_loss / steps
        history.append((avg_loss, val_acc))
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    return W, b, history

# Simulate some data
np.random.seed(0)
N_train, N_val, d = 200, 50, 768
X_tr = np.random.randn(N_train, d)
y_tr = (X_tr[:, 0] > 0).astype(int)  # artificial linear task
X_v = np.random.randn(N_val, d)
y_v = (X_v[:, 0] > 0).astype(int)

W_final, b_final, hist = simple_finetune_loop(X_tr, y_tr, X_v, y_v,
                                               d_model=d, lr=0.01, epochs=3)
```

### The PEFT Family

Parameter-efficient fine-tuning methods can be categorized:

| Method | What is trained | Parameters | Speed |
|---|---|---|---|
| Full fine-tuning | All weights | 100% | Slow |
| Feature extraction | Head only | <0.1% | Very fast |
| Adapters | Small bottleneck modules | 0.5-3% | Moderate |
| LoRA | Low-rank $\Delta W$ matrices | 0.1-1% | Moderate |
| Prefix tuning | Learned key-value prepends | 0.1% | Moderate |
| Prompt tuning | Soft input tokens | <0.01% | Very fast |

**Key takeaway:** Use a learning rate of 1e-5 to 5e-5 for fine-tuning. The Hugging Face Trainer and PEFT libraries abstract away most boilerplate. For resource-constrained settings, LoRA with rank 8 is an excellent default that approaches full fine-tuning quality.

---

## Section 9: Adapter Modules and QLoRA

### Adapter Modules

Adapters (Houlsby et al., 2019) insert small bottleneck modules **inside** each Transformer layer. Unlike LoRA, which modifies the weight matrices themselves, adapters add new layers.

An adapter module is a two-layer residual MLP:
$$\text{Adapter}(x) = x + W_{\text{up}} \cdot f(W_{\text{down}} \cdot x)$$

where:
- $W_{\text{down}} \in \mathbb{R}^{d \times r}$ projects down to a bottleneck dimension $r$
- $f$ is a nonlinear activation (typically ReLU or GELU)
- $W_{\text{up}} \in \mathbb{R}^{r \times d}$ projects back up
- The residual $+x$ means the adapter starts as an identity (if $W_{\text{up}}$ is near zero at init)

Adapters are inserted in two places per Transformer block: after the self-attention sub-layer and after the FFN sub-layer.

```python
import numpy as np

class AdapterModule:
    """
    Bottleneck adapter for a single position.
    Frozen weights: the surrounding Transformer.
    Trainable: W_down, b_down, W_up, b_up.
    """
    def __init__(self, d_model, bottleneck_dim, seed=42):
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(d_model)
        # Down-project: d_model → bottleneck
        self.W_down = rng.randn(d_model, bottleneck_dim) * scale
        self.b_down = np.zeros(bottleneck_dim)
        # Up-project: bottleneck → d_model (init near zero for identity start)
        self.W_up = rng.randn(bottleneck_dim, d_model) * 0.01
        self.b_up = np.zeros(d_model)

    def forward(self, x):
        """
        x: (..., d_model)
        Returns: (..., d_model) — residual adapter output
        """
        # Down-project + activate
        h = np.maximum(0, x @ self.W_down + self.b_down)   # ReLU
        # Up-project
        delta = h @ self.W_up + self.b_up
        # Residual connection
        return x + delta

    def num_params(self):
        return (self.W_down.size + self.b_down.size +
                self.W_up.size + self.b_up.size)

# Example
d_model, r = 768, 64
adapter = AdapterModule(d_model, r)
x = np.random.randn(5, d_model)
out = adapter.forward(x)
print(f"Adapter output shape: {out.shape}")
print(f"Adapter params: {adapter.num_params():,}")
print(f"Reduction vs full layer ({d_model}²): {adapter.num_params()/(d_model**2):.2%}")
```

### Adapter vs LoRA: Practical Comparison

| Property | Adapters | LoRA |
|---|---|---|
| Where applied | Between sub-layers (added depth) | Within attention weight matrices |
| Inference cost | Extra forward pass through adapter | Zero (can merge into W₀) |
| Expressiveness | High (nonlinear activation) | Moderate (linear, but applied everywhere) |
| Parameters | $2 \times (d \times r + r \times d)$ per block | $2 \times (d \times r)$ per matrix |
| Merge for inference | No | Yes |
| Popularity (2024) | Less common | Dominant |

LoRA has largely displaced adapters in practice because:
1. Weights can be merged at inference — zero overhead
2. Simpler to implement
3. Strong empirical results

### QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) enables fine-tuning of very large models (65B+ parameters) on a single GPU by combining:

1. **4-bit NormalFloat quantization (NF4)** of the frozen base model: instead of 16-bit floats, use 4-bit integers. The base model uses ~4 GB for 7B params instead of ~14 GB.

2. **Double quantization**: quantize the quantization constants themselves, saving another ~0.4 bits per parameter.

3. **Paged optimizers**: use CPU memory as an overflow for GPU optimizer states, preventing OOM on long sequences.

4. **LoRA in bf16**: the LoRA matrices (A and B) are kept in full 16-bit precision — they are small, so the cost is negligible.

The result: fine-tune a 65B-parameter LLaMA model on a single NVIDIA A100 80GB GPU — something that would normally require 8× A100s.

```python
import numpy as np

def quantize_nf4(weights, block_size=64):
    """
    Simplified 4-bit NormalFloat quantization.
    NF4 uses a quantile-based grid for normally-distributed weights.
    Returns quantized indices and dequantization info.
    """
    # NF4 quantization levels (values from the paper)
    # These are the 16 levels that minimize quantization error
    # for weights drawn from N(0, 1)
    nf4_levels = np.array([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0923, 0.0,
         0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0
    ])

    n = len(weights)
    num_blocks = (n + block_size - 1) // block_size
    quantized = np.zeros(n, dtype=np.uint8)
    scales = np.zeros(num_blocks)

    for b in range(num_blocks):
        block = weights[b*block_size : (b+1)*block_size]
        scale = np.abs(block).max() + 1e-8
        scales[b] = scale
        normalized = block / scale

        # Find nearest NF4 level for each weight
        for i, w in enumerate(normalized):
            idx = np.abs(nf4_levels - w).argmin()
            quantized[b*block_size + i] = idx

    return quantized, scales, nf4_levels

def dequantize_nf4(quantized, scales, nf4_levels, block_size=64):
    """Reconstruct approximate float weights from 4-bit indices."""
    n = len(quantized)
    weights = np.zeros(n)
    for b in range((n + block_size - 1) // block_size):
        block_idx = quantized[b*block_size : (b+1)*block_size]
        weights[b*block_size : (b+1)*block_size] = nf4_levels[block_idx] * scales[b]
    return weights

# Simulate quantizing a small weight matrix
np.random.seed(42)
w = np.random.randn(256)  # 256 weights from N(0,1)
q, scales, levels = quantize_nf4(w)
w_dequant = dequantize_nf4(q, scales, levels)

mse = np.mean((w - w_dequant)**2)
bits_per_weight = 4  # 16 levels = 4 bits
print(f"Original size: {w.nbytes} bytes ({w.dtype})")
print(f"Quantized size: {q.nbytes // 2} bytes (4-bit, ~{len(w)/2} bytes)")
print(f"Compression: {w.nbytes / (len(w)/2):.1f}x")
print(f"Reconstruction MSE: {mse:.6f}")
```

### The PEFT Ecosystem

The Hugging Face PEFT library implements all major parameter-efficient fine-tuning methods in a unified API:

```python
# Pseudocode: unified PEFT API
# from peft import (
#     LoraConfig, AdaLoraConfig, AdapterConfig,
#     PrefixTuningConfig, PromptTuningConfig,
#     get_peft_model, TaskType
# )

# All methods use the same training loop:
# 1. Define config
# 2. Wrap model with get_peft_model()
# 3. Train normally
# 4. Save adapter weights (very small files!)

# LoRA config example
lora_config_example = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"  # or SEQ_CLS, TOKEN_CLS, SEQ_2_SEQ_LM
}

# AdaLoRA: adaptive rank allocation per layer
adalora_config_example = {
    # Like LoRA, but rank r is learned per layer during training
    # Important layers get higher rank, unimportant layers get rank 0
    "init_r": 12,
    "target_r": 8,
    "deltaT": 1,
    "beta1": 0.85,
    "beta2": 0.85,
}

# Practical sizes for a 7B parameter model
method_comparison = {
    "Full fine-tuning": "7B params (28 GB fp16)",
    "LoRA r=8 (q,v only)": "~8M params (32 MB)",
    "LoRA r=64 (all attn)": "~320M params (1.3 GB)",
    "Prompt tuning (100 tokens)": "~0.3M params (1.2 MB)",
}
for method, size in method_comparison.items():
    print(f"{method}: {size}")
```

### When to Use Which Method

A practical decision guide:

```
Do you have GPU with enough memory for full FT?
├── YES, and >1000 labeled examples? → Full fine-tuning (best quality)
├── YES, but small dataset (<500)? → LoRA r=4-8, heavy dropout
└── NO (limited VRAM)?
    ├── Model fits in fp16? → LoRA r=8-16
    ├── Model too large? → QLoRA (4-bit base + LoRA)
    └── No GPU at all? → Feature extraction or prompt tuning
```

**Key takeaway:** Adapter modules add residual bottleneck MLPs inside Transformer layers. QLoRA extends LoRA by quantizing the frozen base model to 4-bit, enabling fine-tuning of 65B+ models on a single GPU. The PEFT library provides a unified API for all these methods.

---

## Quick Reference: Fine-Tuning Cheat Sheet

**Task-to-Head mapping:**

| Task | Input Format | Output Head | Loss |
|---|---|---|---|
| Binary/Multi-class classification | `[CLS] text [SEP]` | Linear on `[CLS]` → $C$ logits | Cross-entropy |
| NER (token labeling) | `[CLS] w1 w2 ... [SEP]` | Linear per token → label logits | Cross-entropy (per token) |
| Extractive QA | `[CLS] Q [SEP] Passage [SEP]` | Two linear heads (start/end scores) | Cross-entropy over positions |
| Sentence pair (NLI, STS) | `[CLS] sent_A [SEP] sent_B [SEP]` | Linear on `[CLS]` → labels | Cross-entropy / MSE |
| Masked LM (pre-training) | `[CLS] tok1 [MASK] tok3 [SEP]` | Linear → vocab size | Cross-entropy at `[MASK]` |

**Critical hyperparameters for fine-tuning BERT/RoBERTa:**

| Hyperparameter | Value | Why |
|---|---|---|
| Learning rate | 2e-5 (range: 1e-5 – 5e-5) | Pre-trained weights are fragile; large LR destroys them |
| Batch size | 16 or 32 | Larger batches need proportionally higher LR |
| Epochs | 2–5 | Overfitting risk on small datasets |
| Warmup steps | ~6% of total | Smooth LR ramp avoids early instability |
| Weight decay | 0.01 | L2 regularization on non-bias parameters |
| Grad clip | 1.0 | Prevents gradient explosions from large logits |
| Dropout | 0.1 (existing) | Don't change from model defaults |

**LoRA rank guide:**

| Task difficulty | Dataset size | Recommended rank |
|---|---|---|
| Simple classification | > 10K examples | r = 4 |
| Moderate (NER, QA) | 1K–10K | r = 8 |
| Complex (instruction tuning) | Any | r = 16–64 |
| Domain adaptation | > 100K | r = 64–128 |

**Memory comparison for 7B model training:**

| Method | GPU Memory | Params Updated |
|---|---|---|
| Full fine-tuning (fp16) | ~80 GB | 7B (100%) |
| LoRA r=8, q/v only (fp16) | ~16 GB | ~8M (0.1%) |
| QLoRA r=8 (4-bit base) | ~6 GB | ~8M (0.1%) |
| Feature extraction | ~14 GB inference only | ~2M (head only) |

**Key takeaway:** Use this quick reference during the IOAI competition. For any NLP fine-tuning problem: identify the task → choose the head → set LR to 2e-5 → train 3 epochs. If memory is constrained: switch to LoRA (r=8) or QLoRA.

---

## Summary

This lesson covered the spectrum of techniques for adapting pre-trained language models to specific tasks:

1. **Why fine-tune**: pre-trained models encode general language knowledge; fine-tuning specializes it with minimal labeled data. Transfer learning reduces labeled data requirements by 10-100×.

2. **Full fine-tuning**: update all weights. Best performance but high compute and memory cost. Needs 1000+ examples to avoid overfitting.

3. **Feature extraction**: freeze backbone, train only the head. Fast and cheap; suitable for tiny datasets or no GPU.

4. **LoRA**: train low-rank matrices $\Delta W = BA$ ($r \approx 8$); freeze $W_0$. Updates <1% of parameters; matches full fine-tuning; merges for zero-cost inference.

5. **Prompt tuning / instruction tuning**: soft prompt tuning freezes everything and learns continuous prompt vectors; instruction tuning fine-tunes on diverse natural language tasks.

6. **Catastrophic forgetting**: mitigate with small LR, L2 regularization toward pre-trained weights, warmup schedules, or PEFT methods that leave base weights intact.

7. **Task formats**: classification → head on `[CLS]`; NER → per-token head; QA → span-prediction head.

8. **Practical tips**: learning rate 2e-5, 2-5 epochs, warmup 6%, gradient clipping 1.0. Use the Hugging Face PEFT library to implement LoRA in 5 lines.

The next lesson moves from adapting models to generating text with them — including temperature, sampling strategies, beam search, and the RLHF pipeline that turns raw LLMs into helpful assistants.
