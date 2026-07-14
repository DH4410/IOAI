---
title: Decision Trees
track: ml
order: 5
estimatedTime: 40
difficulty: intermediate
---

# Decision Trees

Decision trees are one of the most intuitive and interpretable machine learning models. They make predictions by asking a series of yes/no questions about the features. If you have ever played "20 Questions," you understand the basic idea.

---

## The 20 Questions Intuition

Suppose you want to predict whether a person likes classical music. You might ask:
1. Age > 30? → If yes, go to question 2. If no, go to question 3.
2. Has college degree? → If yes: likely likes classical. If no: uncertain.
3. Parents listen to classical? → If yes: might like it. If no: probably not.

This is exactly a decision tree. Each internal node is a question (a split on a feature), each branch is an answer (yes/no, or threshold comparison), and each leaf node is a prediction.

---

## Tree Terminology

```
                    [Is Age > 30?]          ← root node (first split)
                   /              \
          [Has Degree?]        [Parents Listen?]   ← internal nodes
          /          \          /            \
     [Likes]    [Uncertain]  [Might Like]  [Doesn't Like]  ← leaf nodes
```

- **Root node**: the first split — the most informative question
- **Internal node**: subsequent splits
- **Leaf node**: a terminal node with a prediction
- **Depth**: the length of the longest path from root to leaf
- **Branch**: an edge connecting two nodes

---

## How Does the Tree Choose Which Feature to Split On?

The key question: given a node with a set of training examples, which feature and threshold gives the best split?

A good split should separate the classes as cleanly as possible. We measure "cleanliness" with **impurity measures**.

### Gini Impurity

Gini impurity measures how often a randomly chosen element would be misclassified if it were randomly labeled according to the class distribution.

For a node with K classes:

```
Gini = 1 - Σ pₖ²
```

Where `pₖ` is the proportion of class k examples in the node.

- Gini = 0: perfectly pure (all examples are the same class) — best possible
- Gini = 0.5 (for binary): maximally impure (50/50 split) — worst

```python
import numpy as np

def gini_impurity(y):
    """Compute Gini impurity for an array of labels."""
    n = len(y)
    if n == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)

# Examples
print("Gini impurity examples:")
print(f"  All class 0:         {gini_impurity(np.array([0,0,0,0,0])):.4f}  (pure)")
print(f"  50/50 split:         {gini_impurity(np.array([0,0,0,1,1])):.4f}")
print(f"  All class 1:         {gini_impurity(np.array([1,1,1,1,1])):.4f}  (pure)")
print(f"  3-class equal split: {gini_impurity(np.array([0,0,1,1,2,2])):.4f}")
```

### Information Gain and Entropy

**Entropy** comes from information theory. It measures the average amount of information (surprise) in a distribution.

```
H = -Σ pₖ log₂(pₖ)
```

- H = 0: perfectly pure (no surprise — we know the class)
- H = 1: maximally impure for binary (completely uncertain)

**Information Gain** is the reduction in entropy achieved by splitting:

```
IG = H(parent) - [weighted average entropy of children]
```

```python
import numpy as np

def entropy(y):
    """Compute entropy for an array of labels."""
    n = len(y)
    if n == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n
    # Avoid log(0) by filtering out zero probabilities
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def information_gain(y_parent, y_left, y_right):
    """
    Information gain from splitting y_parent into y_left and y_right.
    """
    n = len(y_parent)
    n_l = len(y_left)
    n_r = len(y_right)

    h_parent = entropy(y_parent)
    h_left   = entropy(y_left)
    h_right  = entropy(y_right)

    weighted_children = (n_l / n) * h_left + (n_r / n) * h_right
    return h_parent - weighted_children

# Example: which split is better?
y_parent = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 3 class-0, 5 class-1
print(f"Parent entropy: {entropy(y_parent):.4f}")

# Split A: [0,0,0] vs [1,1,1,1,1]
y_left_a  = np.array([0, 0, 0])
y_right_a = np.array([1, 1, 1, 1, 1])
ig_a = information_gain(y_parent, y_left_a, y_right_a)
print(f"\nSplit A: [0,0,0] | [1,1,1,1,1]")
print(f"  Left entropy:   {entropy(y_left_a):.4f}")
print(f"  Right entropy:  {entropy(y_right_a):.4f}")
print(f"  Information Gain: {ig_a:.4f}  ← higher is better")

# Split B: [0,0,1] vs [0,1,1,1,1]
y_left_b  = np.array([0, 0, 1])
y_right_b = np.array([0, 1, 1, 1, 1])
ig_b = information_gain(y_parent, y_left_b, y_right_b)
print(f"\nSplit B: [0,0,1] | [0,1,1,1,1]")
print(f"  Left entropy:   {entropy(y_left_b):.4f}")
print(f"  Right entropy:  {entropy(y_right_b):.4f}")
print(f"  Information Gain: {ig_b:.4f}")

print(f"\nBetter split: {'A' if ig_a > ig_b else 'B'}")
```

---

## Finding the Best Split

For each feature and threshold, we compute the impurity reduction. The tree picks the (feature, threshold) pair that gives the greatest gain.

```python
import numpy as np

def best_split(X, y, criterion='gini'):
    """
    Find the best feature and threshold to split on.
    Returns (best_feature_idx, best_threshold, best_gain).
    """
    n, p = X.shape
    best_gain  = -np.inf
    best_feat  = None
    best_thresh = None

    if criterion == 'gini':
        impurity_fn = gini_impurity
    else:
        impurity_fn = entropy

    parent_impurity = impurity_fn(y)

    for feat in range(p):
        thresholds = np.unique(X[:, feat])

        for thresh in thresholds:
            left_mask  = X[:, feat] <= thresh
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            y_left  = y[left_mask]
            y_right = y[right_mask]

            n_l = len(y_left)
            n_r = len(y_right)

            child_impurity = (n_l/n) * impurity_fn(y_left) + (n_r/n) * impurity_fn(y_right)
            gain = parent_impurity - child_impurity

            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


# Test on a small dataset
X = np.array([
    [2.5, 1.0],
    [1.5, 2.0],
    [3.5, 0.5],
    [0.5, 3.0],
    [4.0, 1.5],
    [1.0, 1.0],
])
y = np.array([1, 0, 1, 0, 1, 0])  # alternating classes

feat, thresh, gain = best_split(X, y)
print(f"Best split: feature {feat}, threshold {thresh:.1f}, gain {gain:.4f}")
print(f"Left  (x{feat} <= {thresh}): {y[X[:, feat] <= thresh]}")
print(f"Right (x{feat} >  {thresh}): {y[X[:, feat] > thresh]}")
```

---

## Depth and Overfitting

The depth of a tree controls how complex it is.

- **Shallow tree (max_depth=1)**: Makes only one split. High bias, low variance. Underfits.
- **Deep tree (unlimited depth)**: Grows until each leaf has one example. Perfectly memorizes training data. Overfits badly.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(
    n_samples=300, n_features=5, n_informative=3, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{'max_depth':>10} {'Train Acc':>10} {'Test Acc':>10} {'n_leaves':>10}")
print("-" * 44)
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    tr_acc  = tree.score(X_train, y_train)
    te_acc  = tree.score(X_test, y_test)
    n_leaves = tree.get_n_leaves()
    depth_label = str(depth) if depth is not None else "None (full)"
    print(f"{depth_label:>10} {tr_acc:>10.4f} {te_acc:>10.4f} {n_leaves:>10}")
```

> **Notice:** At `max_depth=None`, the tree achieves 100% training accuracy but may have lower test accuracy. This is overfitting in action.

---

## Pruning

Pruning is the process of removing branches that provide little information gain, making the tree simpler and more generalizable.

**Pre-pruning** (sklearn): Set `max_depth`, `min_samples_split`, `min_samples_leaf` to prevent the tree from growing too deep.

**Post-pruning**: Grow the full tree, then remove branches. Sklearn supports **cost-complexity pruning** via the `ccp_alpha` parameter.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find the range of ccp_alpha values
full_tree = DecisionTreeClassifier(random_state=42)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # exclude the last (trivial tree)

print("Cost-Complexity Pruning:")
print(f"{'ccp_alpha':>12} {'Train Acc':>10} {'Test Acc':>10} {'n_leaves':>10}")
print("-" * 46)
for alpha in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree.fit(X_train, y_train)
    print(f"{alpha:>12.3f} {tree.score(X_train, y_train):>10.4f} "
          f"{tree.score(X_test, y_test):>10.4f} {tree.get_n_leaves():>10}")
```

---

## Building a Simple Decision Tree from Scratch

```python
import numpy as np

class SimpleDecisionTree:
    """A basic decision tree for binary classification."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts/n)**2)

    def _best_split(self, X, y):
        n, p = X.shape
        best = {'gain': -1, 'feature': None, 'threshold': None}
        parent_gini = self._gini(y)

        for feat in range(p):
            for thresh in np.unique(X[:, feat]):
                l = y[X[:, feat] <= thresh]
                r = y[X[:, feat] >  thresh]
                if len(l) == 0 or len(r) == 0:
                    continue
                gain = parent_gini - (len(l)/n)*self._gini(l) - (len(r)/n)*self._gini(r)
                if gain > best['gain']:
                    best = {'gain': gain, 'feature': feat, 'threshold': thresh}

        return best

    def _build(self, X, y, depth):
        # Stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'prediction': classes[np.argmax(counts)],
                    'prob': counts.max() / len(y)}

        split = self._best_split(X, y)
        if split['gain'] <= 0 or split['feature'] is None:
            classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'prediction': classes[np.argmax(counts)],
                    'prob': counts.max() / len(y)}

        feat, thresh = split['feature'], split['threshold']
        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'gain': split['gain'],
            'left':  self._build(X[left_mask],  y[left_mask],  depth+1),
            'right': self._build(X[right_mask], y[right_mask], depth+1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)
        return self

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree, x) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=4, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_tree = SimpleDecisionTree(max_depth=4, min_samples_split=5)
my_tree.fit(X_train, y_train)

sk_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
sk_tree.fit(X_train, y_train)

print(f"My tree  — Test accuracy: {my_tree.score(X_test, y_test):.4f}")
print(f"Sklearn  — Test accuracy: {sk_tree.score(X_test, y_test):.4f}")
```

---

## Sklearn: DecisionTreeClassifier and DecisionTreeRegressor

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ── Classification ─────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

print("Decision Tree Classifier (Iris):")
print(f"  Train accuracy: {clf.score(X_train, y_train):.4f}")
print(f"  Test accuracy:  {clf.score(X_test, y_test):.4f}")
print(f"  Tree depth:     {clf.get_depth()}")
print(f"  Leaves:         {clf.get_n_leaves()}")

# Print the tree as text
print("\nTree structure:")
print(export_text(clf, feature_names=iris.feature_names))

# ── Regression ─────────────────────────────────────────────────────
np.random.seed(0)
X_r, y_r = make_regression(n_samples=300, n_features=5, noise=10, random_state=0)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=0)

reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=0)
reg.fit(X_train_r, y_train_r)

print(f"\nDecision Tree Regressor:")
print(f"  Train R²: {reg.score(X_train_r, y_train_r):.4f}")
print(f"  Test R²:  {reg.score(X_test_r, y_test_r):.4f}")
```

---

## Visualizing a Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(iris.data, iris.target)

# Text representation
print(export_text(clf, feature_names=list(iris.feature_names)))

# Check one path manually
x = iris.data[0]  # first flower
print(f"\nExample flower features:")
print(f"  sepal length: {x[0]}")
print(f"  sepal width:  {x[1]}")
print(f"  petal length: {x[2]}")
print(f"  petal width:  {x[3]}")
print(f"\nTree predicts class: {clf.predict([x])[0]} "
      f"= '{iris.target_names[clf.predict([x])[0]]}'")
print(f"True class: {iris.target[0]} = '{iris.target_names[iris.target[0]]}'")

# Show what happens at each node
print("\nDecision path (1 = node visited, 0 = not visited):")
node_indicator = clf.decision_path([x])
print(f"  Nodes visited: {node_indicator.toarray()[0]}")
```

---

## Feature Importance

Decision trees can tell us how much each feature contributed to the splits. Features used higher up in the tree (with more data flowing through them) are more important.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

print(f"Test accuracy: {tree.score(X_test, y_test):.4f}\n")

importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"{'Rank':>4} {'Feature':>30} {'Importance':>12}")
print("-" * 50)
for rank, i in enumerate(indices[:10]):  # top 10
    bar = "#" * int(importances[i] * 50)
    print(f"{rank+1:>4} {data.feature_names[i]:>30} {importances[i]:>12.4f}  {bar}")
```

---

## Handling Missing Values and Categorical Features

Decision trees handle these more naturally than linear models:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Missing values: sklearn trees can use surrogate splits or you can impute
# Categorical features: need encoding before sklearn

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Example: data with missing values
X = np.array([
    [25, 50000, 1],
    [35, np.nan, 0],  # missing income
    [np.nan, 75000, 0],  # missing age
    [45, 90000, 1],
    [28, 55000, 1],
    [52, np.nan, 0],
])
y = np.array([0, 1, 0, 1, 0, 1])

# Pipeline: impute then tree
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('tree', DecisionTreeClassifier(max_depth=2, random_state=42))
])

pipeline.fit(X, y)
predictions = pipeline.predict(X)
print("Predictions with missing value handling:", predictions)
print("True labels:                             ", y)
print(f"Accuracy: {(predictions == y).mean():.4f}")
```

---

## When Trees Work Well vs. Fail

| Works Well | Struggles |
|---|---|
| Mixed feature types (numeric + categorical) | Very smooth, continuous relationships |
| Non-linear relationships | Extrapolating beyond training range |
| Outliers (splits are threshold-based) | High-dimensional sparse data (text) |
| Feature interactions | Small datasets (overfits easily) |
| Interpretability is required | Needs very precise decision boundaries |

---

## Summary

| Concept | Key Idea |
|---|---|
| Gini impurity | 1 - Σpₖ² — measures class mixture |
| Entropy | -Σpₖ log₂(pₖ) — information measure |
| Information Gain | Reduction in impurity from a split |
| Best split | Feature + threshold with highest gain |
| Overfitting in trees | Deep trees memorize training data |
| Pruning | Remove branches with little gain |
| Feature importance | Gini gain weighted by samples at each node |

Decision trees are powerful on their own, but they shine when combined into **ensembles**. In the next lesson, we will learn how random forests and gradient boosting — the reigning champions of classical ML — amplify the power of individual trees.
