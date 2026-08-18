---
title: Transformers and BERT
track: nlp
order: 5
estimatedTime: 40
difficulty: advanced
---

# Transformers and BERT

BERT (Bidirectional Encoder Representations from Transformers) is the model behind modern NLP. In competition, loading a pretrained BERT and fine-tuning it is usually the fastest route to high accuracy on text tasks.

---

## 1. The Transformer Architecture

A transformer processes all tokens in parallel (unlike RNNs which are sequential). It has two main parts:

**Encoder:** reads the input and builds a rich representation of each token. This is what BERT uses.

**Decoder:** generates output tokens one at a time. This is what GPT uses.

For classification tasks (IOAI), you almost always use only the encoder.

Each encoder layer has:
1. Multi-head self-attention (lets each token look at all others)
2. Feed-forward network (two linear layers with ReLU)
3. Layer normalization and residual connections

```
Input
  |
[Token Embeddings + Positional Encoding]
  |
[Encoder Layer 1: Self-Attn -> Add+Norm -> FFN -> Add+Norm]
  |
[Encoder Layer 2: ...]
  |
 ...
  |
[Encoder Layer 12: ...]   <- BERT base has 12 layers
  |
Output: 768-dim vector for each token
```

---

## 2. BERT's Pretraining

BERT was trained on 3.3 billion words with two tasks:

**Masked Language Modeling:** Randomly hide 15% of words and predict them from context. This forces the model to understand context in both directions.

**Next Sentence Prediction:** Given two sentences, predict if sentence B actually follows sentence A. This teaches sentence-level relationships.

After pretraining, BERT "knows" a lot about English language. You just need to fine-tune it for your specific task.

---

## 3. Using BERT for Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained BERT for classification (adds a linear layer on top)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2    # adjust for your task
)

# Tokenize
texts = ["I love this movie", "This film was terrible"]
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Forward pass
outputs = model(**inputs)
logits = outputs.logits
print(logits.shape)          # (2, 2)

preds = logits.argmax(dim=-1)
print(preds)                 # predicted class for each input
```

The model adds a linear layer on top of the `[CLS]` token embedding and outputs one logit per class.

**Quick check:** Why is the `[CLS]` token used for classification instead of averaging all tokens?
> BERT was trained to put sentence-level information into the `[CLS]` token. Using it for classification leverages what the model already learned during pretraining.

---

## 4. Fine-Tuning BERT

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# Training loop
def train_bert(model, train_loader, epochs=3, lr=2e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}: loss = {total_loss / len(train_loader):.4f}')
```

Key settings for fine-tuning BERT:
- Learning rate: `2e-5` to `5e-5` (much lower than training from scratch)
- Epochs: 3-5 (too many = overfitting on your small dataset)
- Batch size: 16 or 32

---

## 5. Popular Model Variants

| Model | Size | Notes |
|---|---|---|
| `bert-base-uncased` | 110M params | Standard choice, good balance |
| `bert-large-uncased` | 340M params | More accurate, slower |
| `distilbert-base-uncased` | 66M params | 60% faster, 97% of BERT's performance |
| `roberta-base` | 125M params | Usually better than BERT |
| `xlm-roberta-base` | 270M params | Multilingual (for non-English tasks) |

For competition: try `roberta-base` first. If speed matters, use `distilbert-base-uncased`.

---

## 6. Getting Sentence Embeddings

Sometimes you need a fixed-size embedding for each sentence (for retrieval or similarity tasks):

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch, seq, 768)
    mask = attention_mask.unsqueeze(-1).float()         # (batch, seq, 1)
    return (token_embeddings * mask).sum(1) / mask.sum(1)

texts = ["First sentence", "Second sentence"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

embeddings = mean_pooling(outputs, inputs['attention_mask'])
print(embeddings.shape)   # (2, 768) - one 768-dim vector per sentence
```

Mean pooling (average all token embeddings) often works better than just the `[CLS]` token for sentence similarity.

---

## Sort the Transformer Concepts

```widget
{
  "type": "concept-sort",
  "title": "BERT or GPT? Which model type for each task?",
  "categories": [
    { "name": "BERT (Encoder)", "color": "#5B5BD6" },
    { "name": "GPT (Decoder)", "color": "#F97316" }
  ],
  "items": [
    { "text": "Text classification (spam detection)", "category": "BERT (Encoder)" },
    { "text": "Open-ended text generation", "category": "GPT (Decoder)" },
    { "text": "Masked language modeling pretraining", "category": "BERT (Encoder)" },
    { "text": "Causal language modeling (predict next token)", "category": "GPT (Decoder)" },
    { "text": "Named entity recognition", "category": "BERT (Encoder)" },
    { "text": "ChatGPT-style conversation", "category": "GPT (Decoder)" }
  ]
}
```

---

## Practice Questions

**Quick check:** How many parameters does `bert-base-uncased` have, and what does "base" vs "large" mean?
> BERT-base has ~110 million parameters (12 layers, 768 hidden, 12 heads). BERT-large has ~340M (24 layers, 1024 hidden, 16 heads). "Large" is slower to fine-tune but generally scores higher.

**Quick check:** What learning rate should you use to fine-tune BERT, and why is it so small?
> 2e-5 to 5e-5. Much smaller than training from scratch because BERT's weights are already good — you're nudging them, not relearning. A larger rate destroys the pretrained features (called "catastrophic forgetting").

**Quick check:** Your BERT fine-tune gets 0.95 train F1 but 0.72 validation F1 after epoch 5. What should you do?
> It's overfitting. Try: fewer epochs (stop at epoch 2-3), stronger weight decay, smaller learning rate, or a larger dropout. For small datasets, freeze the first 6 layers and only fine-tune the last 6.

---

## Summary

| Step | Code |
|---|---|
| Load tokenizer | `AutoTokenizer.from_pretrained('bert-base-uncased')` |
| Load model | `AutoModelForSequenceClassification.from_pretrained(...)` |
| Tokenize | `tokenizer(texts, padding=True, truncation=True, max_length=128)` |
| Fine-tune LR | `2e-5` to `5e-5` |
| Epochs | 3-5 |
| Batch size | 16 or 32 |

In competition: grab `roberta-base`, tokenize your data, fine-tune for 3-5 epochs. You can often get 85-95% accuracy on text classification tasks this way with very little custom code.
