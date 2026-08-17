---
title: Decision Trees
track: ml
order: 5
estimatedTime: 40
difficulty: intermediate
---

# Decision Trees

A decision tree is one of the most intuitive ML models — it works exactly like a flowchart of yes/no questions. No feature scaling required, handles mixed data types, and is completely interpretable. It's also the building block of Random Forest and XGBoost, which dominate IOAI-style competitions.

---

## 1. How a Tree Makes Predictions

The tree splits the data by asking questions about features:

```
                   [Age > 30?]
                  /            \
             YES /              \ NO
           [Income > 50k?]    [Student?]
           /          \         /       \
      YES /            \ NO  YES /       \ NO
       BUYS          SKIP   BUYS        SKIP
    (prob: 0.80)  (0.70)  (0.65)       (0.75)
```

Each leaf node stores the **majority class** (or the probability distribution). To predict a new point: start at the root, follow the questions down, reach a leaf, return its class.

---

## 2. How Splits Are Chosen: Gini Impurity

At each node, the tree tries **every feature × every threshold** and picks the split that makes the resulting groups as *pure* as possible. "Pure" means one class dominates.

**Gini impurity** of a node with $K$ classes:

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

- Pure node (all one class): $G = 1 - 1^2 = 0$ ✓
- Perfectly mixed (50/50 binary): $G = 1 - (0.5^2 + 0.5^2) = 0.5$ (worst)

**Information Gain** (splitting criterion):

$$\text{IG} = G(\text{parent}) - \frac{n_\text{left}}{n} G(\text{left}) - \frac{n_\text{right}}{n} G(\text{right})$$

The tree picks the split with **maximum information gain** — biggest reduction in impurity.

```python
import numpy as np

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_gain(y, y_left, y_right):
    n = len(y)
    ig = gini(y) - (len(y_left)/n * gini(y_left)) - (len(y_right)/n * gini(y_right))
    return ig

# Example: split on age > 30
y        = np.array([1,1,1,0,0,1,0,0])   # 4 positive, 4 negative
y_left   = np.array([1,1,0,0])            # age <= 30
y_right  = np.array([1,0,1,0])            # age > 30

print(f"Parent Gini:  {gini(y):.3f}")
print(f"Left Gini:    {gini(y_left):.3f}")
print(f"Right Gini:   {gini(y_right):.3f}")
print(f"Info gain:    {information_gain(y, y_left, y_right):.3f}")
```

---

## 3. Building from Scratch

```python
class SimpleDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._grow(X, y, depth=0)

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n, n_feats = X.shape

        for feat in range(n_feats):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_gini - (len(left)/n)*self._gini(left) - (len(right)/n)*self._gini(right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        mask = X[:, feat] <= thresh
        return {
            'leaf': False, 'feat': feat, 'thresh': thresh,
            'left':  self._grow(X[mask], y[mask], depth+1),
            'right': self._grow(X[~mask], y[~mask], depth+1)
        }

    def _traverse(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feat']] <= node['thresh']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])
```

---

## 4. Using scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(
    max_depth=4,          # maximum tree depth — key hyperparameter
    min_samples_leaf=4,   # minimum samples at each leaf
    criterion='gini',     # 'gini' or 'entropy'
    random_state=42
)
tree.fit(X_train, y_train)

print(f"Train accuracy: {accuracy_score(y_train, tree.predict(X_train)):.2%}")
print(f"Test accuracy:  {accuracy_score(y_test,  tree.predict(X_test)):.2%}")

# See the actual decision rules (great for debugging!)
feature_names = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
print(export_text(tree, feature_names=feature_names))
```

---

## 5. Feature Importance

Decision trees give you a built-in importance score: how often each feature was used to split, weighted by how much information gain it gave.

```python
for name, importance in zip(feature_names, tree.feature_importances_):
    bar = '█' * int(importance * 30)
    print(f"{name:12s} {importance:.3f} {bar}")
```

This is one of the most useful tools in competitions — it tells you which features actually matter.

---

## 6. The Overfitting Problem

Trees with no constraints will perfectly fit the training data. Each leaf becomes a single training example.

```
max_depth=None:  Train = 100%  Test = 78%   (memorized training data)
max_depth=4:     Train = 96%   Test = 93%   (good generalization)
max_depth=1:     Train = 77%   Test = 75%   (underfits, too simple)
```

```python
depths = range(1, 20)
train_scores, test_scores = [], []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_scores.append(t.score(X_train, y_train))
    test_scores.append(t.score(X_test, y_test))

# Plot to see the bias-variance tradeoff visually
import matplotlib.pyplot as plt
plt.plot(depths, train_scores, label='Train', marker='o')
plt.plot(depths, test_scores, label='Test', marker='s')
plt.xlabel('Max depth'), plt.ylabel('Accuracy')
plt.legend(), plt.title('Depth vs Accuracy')
plt.show()
```

---

## 7. Overfit Curve: Watch the Split

```widget
{
  "type": "overfit-curve",
  "title": "Tree complexity vs performance",
  "subtitle": "Move the slider to see how increasing model complexity affects train and validation accuracy."
}
```

Notice: training accuracy increases monotonically as you add complexity. Validation accuracy peaks at the sweet spot and then falls — that's overfitting.

---

## 8. Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `max_depth` | None | Primary control for complexity. Start with 3-8. |
| `min_samples_leaf` | 1 | Larger → smaller tree → less overfit |
| `min_samples_split` | 2 | Min samples to consider splitting a node |
| `max_features` | None | Randomly restrict which features to consider (used in Random Forest) |
| `criterion` | 'gini' | 'gini' or 'entropy' — rarely makes a big difference |

**Cross-validate** these hyperparameters rather than guessing:

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [2, 4, 6, 8, None],
    'min_samples_leaf': [1, 4, 10, 20],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("CV score:   ", grid.best_score_:.2%")
```

---

## 9. Sort These Tree Concepts

```widget
{
  "type": "concept-sort",
  "title": "Tree Properties: What Does Each Reduce?",
  "categories": [
    { "name": "Reduces Overfitting", "color": "#22C55E" },
    { "name": "Increases Complexity", "color": "#EF4444" }
  ],
  "items": [
    { "text": "Larger max_depth", "category": "Increases Complexity" },
    { "text": "min_samples_leaf = 20", "category": "Reduces Overfitting" },
    { "text": "No depth limit", "category": "Increases Complexity" },
    { "text": "Pruning", "category": "Reduces Overfitting" },
    { "text": "More features", "category": "Increases Complexity" },
    { "text": "min_samples_split = 50", "category": "Reduces Overfitting" }
  ]
}
```

---

## 10. Decision Trees vs Other Models

| | Decision Tree | Logistic Reg. | SVM |
|---|---|---|---|
| Feature scaling needed | **No** | Yes | Yes |
| Handles non-linearity | Yes (splits) | No | With kernel |
| Interpretable | **Very** | Moderate | No |
| Overfits easily | **Yes** | Less | Less |
| Best use case | Tree ensembles | Fast baseline | High-dim data |

Trees rarely win competitions alone — but Random Forest (many trees) and XGBoost (boosted trees) are competition powerhouses.

---

## Summary

| Concept | Key point |
|---|---|
| Split criterion | Maximize information gain / minimize Gini impurity |
| `max_depth` | Most important hyperparameter — controls overfitting |
| Feature importance | Built-in: which features were used most for splitting |
| Trees overfit | Unconstrained tree → 100% train accuracy, poor test |
| No scaling needed | Trees split on thresholds, not distances |

The next lesson: **Random Forest** — take 100 trees, each trained on a random subset of data and features, then average their predictions. This dramatically reduces overfitting.
