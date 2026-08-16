---
title: Matplotlib - Visualizing Data
track: python
order: 11
estimatedTime: 25
difficulty: intermediate
---

# Matplotlib - Visualizing Data

You use plots to understand your data and to check if your model is working. This lesson covers the plots you will actually use in an ML competition.

---

## 1. The Basics

```python
import matplotlib.pyplot as plt
import numpy as np
```

The simplest possible plot:

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.plot(x, y)
plt.title('My Plot')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
```

`plt.show()` opens the window. In a Jupyter notebook, plots appear inline without it.

---

## 2. The Four Plots You Need Most

### Line plot - trends over time or training

```python
epochs = range(1, 11)
train_loss = [0.9, 0.7, 0.55, 0.4, 0.32, 0.27, 0.22, 0.19, 0.17, 0.15]
val_loss   = [0.92, 0.74, 0.6, 0.52, 0.5, 0.51, 0.53, 0.55, 0.57, 0.60]

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--')
plt.legend()
plt.title('Training Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

**Quick check:** Why would `val_loss` going up while `train_loss` keeps going down be a problem?
> That is overfitting. The model is memorizing training data but not generalizing.

### Scatter plot - relationship between two features

```python
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

plt.scatter(x, y, alpha=0.5, color='steelblue')
plt.title('Feature X vs Feature Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

`alpha` controls transparency (0 = invisible, 1 = solid). Useful when points overlap.

### Histogram - distribution of a feature

```python
data = np.random.normal(loc=50, scale=10, size=500)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()
```

Look at histograms to spot outliers, skewed data, or multimodal distributions.

### Bar chart - comparing categories

```python
categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
values = [23, 45, 12, 67]

plt.bar(categories, values, color='steelblue')
plt.title('Count per Category')
plt.ylabel('Count')
plt.show()
```

---

## 3. Multiple Plots Side by Side

Use `plt.subplots()` when you want to compare things:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left plot
axes[0].plot(epochs, train_loss)
axes[0].set_title('Train Loss')

# Right plot
axes[1].scatter(x, y, alpha=0.5)
axes[1].set_title('Scatter')

plt.tight_layout()   # prevents labels from overlapping
plt.show()
```

The `figsize=(width, height)` is in inches. `(12, 4)` is a common size for two side-by-side plots.

**Quick check:** What does `axes[0]` refer to?
> The first (left) plot. `axes[1]` is the right one.

For a 2x2 grid, use `subplots(2, 2)` and access with `axes[row][col]`.

---

## 4. Useful ML Patterns

### Confusion matrix heatmap

```python
import sklearn.metrics as sm

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]
cm = sm.confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

### Feature importance bar chart

```python
# After training a RandomForestClassifier
importances = model.feature_importances_
feature_names = X.columns

idx = np.argsort(importances)[::-1]  # sort descending

plt.bar(range(len(importances)), importances[idx])
plt.xticks(range(len(importances)), feature_names[idx], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

---

## Summary

| What to plot | Use |
|---|---|
| Training / validation loss over epochs | Line plot |
| Two features against each other | Scatter plot |
| Distribution of one column | Histogram |
| Compare category counts | Bar chart |
| Multiple plots together | `plt.subplots(rows, cols)` |

You do not need to memorize every option. When you need something specific (colors, labels, tick rotation), look it up. The patterns above cover 90% of competition visualization.
