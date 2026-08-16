---
title: Fine-Tuning for NLP Tasks
track: nlp
order: 6
estimatedTime: 40
difficulty: advanced
---

# Fine-Tuning for NLP Tasks

Fine-tuning is adapting a pretrained model to your specific task. In competition, this is usually more effective than training from scratch. This lesson covers the full pipeline for the most common NLP competition tasks.

---

## 1. Common NLP Competition Tasks

**Text classification:** Given a text, predict a label. (sentiment, topic, toxicity)

**Named Entity Recognition (NER):** Label each word as a person, organization, location, or other.

**Question Answering:** Given a question and a passage, find the answer span in the passage.

**Text regression:** Given a text, predict a number (score, rating, similarity).

This lesson focuses on classification and regression since they are most common in IOAI.

---

## 2. Full Fine-Tuning Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Hyperparameters ---
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 3e-5

# --- Dataset ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors=None
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- Load data (replace with your data loading code) ---
# texts = [...], labels = [...]

# --- Setup ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# --- DataLoaders ---
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
val_dataset   = SentimentDataset(X_val,   y_val,   tokenizer)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# --- Optimizer and scheduler ---
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader),   # warm up for 1 epoch
    num_training_steps=len(train_loader) * EPOCHS
)

# --- Training loop ---
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f'Epoch {epoch+1}: loss={train_loss/len(train_loader):.3f}, val_acc={acc:.3f}')
```

**Quick check:** What does `clip_grad_norm_` do and why is it used?
> It limits the maximum gradient magnitude to 1.0. This prevents "gradient explosions" where very large gradients cause the weights to jump to bad values. It is standard practice when fine-tuning transformers.

---

## 3. Tips for Better Results

**Use a warmup schedule:** The scheduler above warms up the learning rate for the first epoch, then decreases it linearly. This prevents the pretrained weights from being disrupted too quickly in early training.

**Gradient clipping:** Already in the example above (`clip_grad_norm_`). Use `max_norm=1.0`.

**Freeze layers:** If you have very little data, freeze the first few BERT layers and only train the top layers:

```python
# Freeze all except the last 2 transformer layers
for name, param in model.named_parameters():
    if 'encoder.layer.10' in name or 'encoder.layer.11' in name or 'classifier' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**Class imbalance:** If one class has far more examples, use a weighted loss:

```python
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.FloatTensor(weights).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Then in the loop:
loss = criterion(outputs.logits, batch['labels'])
```

---

## 4. Text Regression

For predicting a score (not a class), use regression:

```python
from transformers import AutoModelForSequenceClassification

# 1 output = regression
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

# In the loss computation:
outputs = model(**batch)
predictions = outputs.logits.squeeze()
loss = torch.nn.MSELoss()(predictions, batch['labels'].float())
```

The model outputs a single number per input. Use MSE or MAE as the loss.

---

## 5. Saving and Loading

```python
# Save
model.save_pretrained('saved_model/')
tokenizer.save_pretrained('saved_model/')

# Load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('saved_model/')
model = AutoModelForSequenceClassification.from_pretrained('saved_model/')
```

---

## Summary

| Component | Recommendation |
|---|---|
| Model | `distilbert-base-uncased` (fast) or `roberta-base` (accurate) |
| Learning rate | `2e-5` to `5e-5` |
| Epochs | 3-5 |
| Batch size | 16-32 |
| Warmup | 1 epoch worth of steps |
| Gradient clip | `1.0` |
| Imbalanced data | Weighted cross-entropy loss |

The pipeline above is reusable. Change the model name, the number of labels, and the data loading code, and it works for almost any text classification task.
