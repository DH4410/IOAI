---
title: Building a Strong Baseline
track: ioai
order: 3
estimatedTime: 40
difficulty: advanced
---

# Building a Strong Baseline

In competitions, the teams that win are rarely the ones with the most complex models. They are the ones with the best baselines. A strong baseline gives you a valid submission immediately and a reference point for every improvement you try.

---

## 1. Why Baselines Matter

The most common mistake: skip the baseline and jump straight to complex models. Then you run out of time with a half-finished neural network, and the person with a well-tuned LightGBM baseline beats you.

Rule: get a working baseline in the first 20 minutes. Everything after that is improvement.

---

## 2. The Baseline Template (Tabular Data)

Copy this pattern for any tabular task:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# 1. Load data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# 2. Identify target and features
TARGET = 'label'
FEATURES = [c for c in train.columns if c not in [TARGET, 'id']]

X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]

# 3. Handle categoricals
for col in X.select_dtypes('object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# 4. Fill missing values
X = X.fillna(-999)
X_test = X_test.fillna(-999)

# 5. Train and evaluate
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

# 6. Train on all data and predict
model.fit(X, y)
preds = model.predict(X_test)

# 7. Save submission
submission = pd.DataFrame({'id': test['id'], TARGET: preds})
submission.to_csv('submission.csv', index=False)
```

**Quick check:** Why do we fill missing values with `-999` instead of the mean?
> For tree-based models (LightGBM, XGBoost), `-999` tells the model "this was missing." The model can learn that missingness itself is a signal. Mean imputation hides the information that a value was missing.

---

## 3. The Baseline Template (Image Data)

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
train_dataset = ImageFolder('data/train', transform=transform_train)
val_dataset   = ImageFolder('data/val',   transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

# Model: pretrained ResNet, replace head
num_classes = len(train_dataset.classes)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    print(f'Epoch {epoch+1}: val acc = {correct/total:.2%}')
```

---

## 4. The Baseline Template (Text Data)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train['text'])
X_test  = vectorizer.transform(test['text'])
y_train = train['label']

# Model
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f'CV F1: {scores.mean():.4f}')

model.fit(X_train, y_train)
preds = model.predict(X_test)
pd.DataFrame({'id': test['id'], 'label': preds}).to_csv('submission.csv', index=False)
```

**Quick check:** For a text classification task with 5 classes and class imbalance, which metric should you use?
> `f1_macro` (or `f1_weighted`). Accuracy is misleading when classes are imbalanced. F1 macro averages F1 across classes, giving equal weight to rare classes.

---

## 5. Cross-Validation

Never submit based only on training accuracy. Use cross-validation to estimate test performance:

```python
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))   # out-of-fold predictions

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model.fit(X_fold_train, y_fold_train)
    oof_preds[val_idx] = model.predict(X_fold_val)

    fold_score = accuracy_score(y_fold_val, oof_preds[val_idx])
    print(f'Fold {fold+1}: {fold_score:.4f}')

print(f'Overall OOF: {accuracy_score(y, oof_preds):.4f}')
```

OOF (out-of-fold) predictions give you an unbiased estimate of how well your model generalizes.

---

## Which Model First?

```widget
{
  "type": "concept-sort",
  "title": "Match the Task to its Competition Baseline",
  "categories": [
    { "name": "LightGBM / XGBoost", "color": "#5B5BD6" },
    { "name": "Pretrained CNN / ViT", "color": "#22C55E" },
    { "name": "TF-IDF + LogReg → BERT", "color": "#F97316" }
  ],
  "items": [
    { "text": "Tabular data with categorical and numerical features", "category": "LightGBM / XGBoost" },
    { "text": "Image classification from raw photos", "category": "Pretrained CNN / ViT" },
    { "text": "Text sentiment analysis", "category": "TF-IDF + LogReg → BERT" },
    { "text": "CSV with 50 numerical features, predict survival", "category": "LightGBM / XGBoost" },
    { "text": "Counting objects in satellite images", "category": "Pretrained CNN / ViT" },
    { "text": "Classify customer reviews as positive/negative/neutral", "category": "TF-IDF + LogReg → BERT" }
  ]
}
```

---

## Summary

| Task | Baseline model | Time to baseline |
|---|---|---|
| Tabular classification | LightGBM + 5-fold CV | 15-20 min |
| Tabular regression | LightGBM (same) | 15-20 min |
| Image classification | Pretrained ResNet18 | 20-30 min |
| Text classification | TF-IDF + LogReg | 10-15 min |

Get the baseline working first. Then improve one thing at a time and measure. If a change does not improve CV score by at least 0.3-0.5%, it might just be noise.
