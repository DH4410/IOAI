---
title: Matplotlib — Visualizing Data
track: python
order: 11
estimatedTime: 45
difficulty: intermediate
---

# Matplotlib — Visualizing Data

Data visualization is not decoration — it is one of the most important skills in machine learning. A single chart can reveal patterns that would take hours to find by staring at numbers. In ML, you use visualization to understand your data before training, to monitor training progress, to diagnose problems, and to communicate results.

Matplotlib is Python's foundational plotting library. Nearly every other plotting library (Pandas plots, Seaborn, Plotly, Bokeh) is either built on Matplotlib or inspired by it. Learning it deeply means you can create and customize any plot you encounter in your ML career.

---

## 1. Why Visualization Matters in ML

Here are four situations where a chart tells you something numbers alone cannot:

1. **Understanding distributions:** Are your features normally distributed? Are there outliers? A histogram answers this instantly.
2. **Spotting correlations:** A scatter plot shows whether two variables move together.
3. **Monitoring training:** Plotting training loss vs validation loss across epochs reveals overfitting.
4. **Evaluating models:** An ROC curve or confusion matrix heatmap summarizes model performance visually.

> **Tip:** Before training any model, always visualize your data. You will catch problems — class imbalance, outliers, mislabeled data — that would silently hurt your model if you skipped this step.

---

## 2. Basic Plotting: `plt.plot()` and `plt.show()`

The simplest way to create a plot:

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Display it
plt.show()
```

When you run this, a window appears with a sine wave. In Jupyter notebooks, `plt.show()` may not be needed if you have `%matplotlib inline` set.

### Saving to a File

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.exp(-x * 0.5) * np.sin(x)

plt.plot(x, y)
plt.savefig("my_plot.png", dpi=150, bbox_inches="tight")
# dpi=150 sets resolution (dots per inch)
# bbox_inches='tight' trims whitespace around the figure
plt.show()
```

> **Tip:** Always call `plt.savefig()` BEFORE `plt.show()`. After `show()`, Matplotlib clears the figure, so saving after showing gives you a blank file.

---

## 3. The Figure and Axes Object Model

The most important concept to understand in Matplotlib is the difference between **Figure** and **Axes**.

- A **Figure** is the entire window or canvas — the outer container.
- **Axes** (not to be confused with axis) is a single plot within the figure — it has its own x-axis, y-axis, title, and data.

You can have one Axes per Figure, or many Axes (a grid of subplots).

```python
import matplotlib.pyplot as plt
import numpy as np

# === Approach 1: Implicit (pyplot interface) ===
# Easy for quick single plots
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Quick Plot")
plt.show()

# === Approach 2: Explicit (Object-Oriented interface) ===
# Better for complex figures, subplots, and professional work
fig, ax = plt.subplots()

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))
ax.set_title("Sine Wave")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")

plt.show()
```

The explicit `fig, ax = plt.subplots()` approach is strongly preferred in professional code because:
- You can have multiple axes (subplots) easily
- Methods are called on the `ax` object — it is clear what you are modifying
- It avoids confusing interactions with Matplotlib's global state

> **Convention:** In professional code you almost always see `fig, ax = plt.subplots()`. Get comfortable with this pattern.

### Figure Properties

```python
import matplotlib.pyplot as plt

# Control figure size (width, height) in inches
fig, ax = plt.subplots(figsize=(10, 4))

# figsize=(width, height): wider figures for timeseries, taller for bar charts
fig, ax = plt.subplots(figsize=(6, 8))

# Figure with a title
fig.suptitle("Overall Figure Title", fontsize=16, fontweight="bold")
```

---

## 4. Line Plots

Line plots are ideal for showing trends over time (loss curves, price history, etc.):

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(9, 5))

# Multiple lines on the same axes
ax.plot(x, np.sin(x), label="sin(x)", color="blue",  linewidth=2)
ax.plot(x, np.cos(x), label="cos(x)", color="red",   linewidth=2, linestyle="--")
ax.plot(x, np.sin(x) * np.exp(-x * 0.2),
        label="damped sin", color="green", linewidth=1.5, linestyle="-.")

# Labels and title
ax.set_title("Trigonometric Functions", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)

# Legend
ax.legend(fontsize=11)

# Grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Line Style Options

| `linestyle` | Appearance |
|-------------|------------|
| `"-"` | Solid line (default) |
| `"--"` | Dashed |
| `"-."` | Dash-dot |
| `":"` | Dotted |
| `"none"` | No line (just markers) |

### Marker Options

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 5, 4]

fig, ax = plt.subplots()
ax.plot(x, y, marker="o", markersize=8, markerfacecolor="red",
        linewidth=1.5, color="navy")

# Common markers: "o" circle, "s" square, "^" triangle, "x" cross, "*" star, "." dot
plt.show()
```

---

## 5. Scatter Plots

Scatter plots show the relationship between two continuous variables. They are essential for visualizing correlations and cluster structure:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Three groups of data (like three classes in a classification problem)
n = 50

group_a = np.random.randn(n, 2) + [1, 1]
group_b = np.random.randn(n, 2) + [-1, 2]
group_c = np.random.randn(n, 2) + [0, -2]

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(group_a[:, 0], group_a[:, 1], label="Class A", color="royalblue",
           alpha=0.7, s=50)
ax.scatter(group_b[:, 0], group_b[:, 1], label="Class B", color="tomato",
           alpha=0.7, s=50)
ax.scatter(group_c[:, 0], group_c[:, 1], label="Class C", color="forestgreen",
           alpha=0.7, s=50)

ax.set_title("Scatter Plot — Three Classes", fontsize=14)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Using Color and Size to Encode Extra Dimensions

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
n = 100

x       = np.random.rand(n)
y       = np.random.rand(n)
color   = np.random.rand(n)    # A third variable encoded as color
sizes   = np.random.rand(n) * 300  # A fourth variable encoded as size

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(x, y, c=color, s=sizes, cmap="viridis", alpha=0.8)

# Colorbar to explain the color scale
fig.colorbar(scatter, ax=ax, label="Third Variable")

ax.set_title("Scatter with Color and Size Encoding")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")

plt.tight_layout()
plt.show()
```

> **Key parameters for `scatter()`:**
> - `c=` — color (single color, array of values, or array of colors)
> - `s=` — marker size in points² (a single value or array for variable sizes)
> - `cmap=` — colormap name (used when `c=` is an array of numbers)
> - `alpha=` — transparency, 0 (invisible) to 1 (opaque)

---

## 6. Bar Charts

Bar charts compare discrete categories:

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ["Engineering", "Marketing", "Design", "Sales", "HR"]
values     = [42, 28, 15, 35, 20]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(categories, values, color=["steelblue", "salmon", "mediumseagreen",
                                          "goldenrod", "orchid"],
              edgecolor="white", linewidth=0.8)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
            str(int(height)), ha="center", va="bottom", fontsize=10)

ax.set_title("Employees by Department", fontsize=14)
ax.set_xlabel("Department")
ax.set_ylabel("Number of Employees")
ax.set_ylim(0, 50)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

### Horizontal Bar Chart

Great for long category names:

```python
import matplotlib.pyplot as plt

labels   = ["Natural Language Processing", "Computer Vision",
            "Reinforcement Learning", "Time Series", "Tabular Data"]
accuracy = [0.87, 0.93, 0.72, 0.81, 0.90]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(labels, accuracy, color="steelblue")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Accuracy")
ax.set_title("Model Accuracy by Task Type")
ax.axvline(x=0.8, color="red", linestyle="--", label="Threshold (0.8)")
ax.legend()
plt.tight_layout()
plt.show()
```

### Grouped Bar Chart

```python
import matplotlib.pyplot as pd
import matplotlib.pyplot as plt
import numpy as np

models   = ["Model A", "Model B", "Model C"]
train_acc = [0.95, 0.92, 0.98]
val_acc   = [0.87, 0.90, 0.82]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, train_acc, width, label="Train Accuracy", color="steelblue")
bars2 = ax.bar(x + width/2, val_acc,   width, label="Val Accuracy",   color="salmon")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7. Histograms

Histograms show the **distribution** of a numerical variable — how many values fall into each range:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.randn(1000)   # Standard normal distribution

fig, ax = plt.subplots(figsize=(8, 5))

# bins: number of bars (or a list of bin edges)
ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)

ax.set_title("Distribution of Values (Standard Normal)")
ax.set_xlabel("Value")
ax.set_ylabel("Count")

# Add mean and std lines
ax.axvline(data.mean(), color="red",    linestyle="--", linewidth=2,
           label=f"Mean = {data.mean():.2f}")
ax.axvline(data.mean() + data.std(), color="orange", linestyle="--",
           linewidth=1.5, label=f"+1 SD = {data.mean() + data.std():.2f}")
ax.axvline(data.mean() - data.std(), color="orange", linestyle="--",
           linewidth=1.5, label=f"−1 SD = {data.mean() - data.std():.2f}")

ax.legend()
plt.tight_layout()
plt.show()
```

### Overlapping Histograms — Comparing Two Distributions

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
class_0_scores = np.random.normal(60, 10, 200)
class_1_scores = np.random.normal(80, 8, 200)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(class_0_scores, bins=25, alpha=0.6, color="tomato",
        label="Class 0", density=True)
ax.hist(class_1_scores, bins=25, alpha=0.6, color="steelblue",
        label="Class 1", density=True)
# density=True normalizes so the area under each histogram sums to 1

ax.set_title("Score Distributions by Class")
ax.set_xlabel("Score")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
```

> **Reading histograms:** Look for the shape. A bell curve suggests normal distribution. A long tail to the right means right-skewed data. Two peaks (bimodal) may indicate two sub-groups in your data.

---

## 8. Box Plots

A box plot summarizes a distribution with five statistics: minimum, Q1, median, Q3, maximum. It is excellent for comparing distributions across groups:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Five groups with different distributions
group_data = [
    np.random.normal(70, 8, 100),
    np.random.normal(75, 12, 100),
    np.random.normal(65, 5, 100),
    np.random.normal(80, 15, 100),
    np.random.normal(72, 6, 100)
]
group_labels = ["Dept A", "Dept B", "Dept C", "Dept D", "Dept E"]

fig, ax = plt.subplots(figsize=(9, 6))

bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))

# Color each box differently
colors = ["steelblue", "salmon", "mediumseagreen", "goldenrod", "orchid"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Score Distributions by Department")
ax.set_ylabel("Score")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

### Reading a Box Plot

```
          |
     ─────┤  ← Maximum (or 1.5 × IQR whisker)
          │
    ┌─────┐
    │     │  ← Q3 (75th percentile)
    │─────│  ← Median (50th percentile)
    │     │  ← Q1 (25th percentile)
    └─────┘
          │
     ─────┤  ← Minimum (or 1.5 × IQR whisker)

     ●    ← Outliers (points beyond 1.5 × IQR)
```

- **Box height (IQR)** = Q3 - Q1 — wider box means more spread
- **Median line position** — if close to Q3, the distribution skews left
- **Dots beyond whiskers** = outliers

---

## 9. Subplots — Multiple Plots in One Figure

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 100)

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: line plot
axes[0, 0].plot(x, np.sin(x), color="steelblue")
axes[0, 0].set_title("Sine Wave")
axes[0, 0].grid(True, alpha=0.3)

# Top-right: scatter
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50),
                   alpha=0.6, color="tomato", s=40)
axes[0, 1].set_title("Random Scatter")

# Bottom-left: histogram
axes[1, 0].hist(np.random.randn(500), bins=25, color="mediumseagreen",
                edgecolor="white")
axes[1, 0].set_title("Normal Distribution")

# Bottom-right: bar chart
categories = ["A", "B", "C", "D"]
vals = [3, 7, 2, 9]
axes[1, 1].bar(categories, vals, color="goldenrod")
axes[1, 1].set_title("Bar Chart")

plt.suptitle("Four Plot Types", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

### Subplots with Shared Axes

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# sharex means all plots share the same x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

ax1.plot(x, np.sin(x), color="steelblue")
ax1.set_ylabel("sin(x)")
ax1.grid(True, alpha=0.3)

ax2.plot(x, np.cos(x), color="tomato")
ax2.set_ylabel("cos(x)")
ax2.grid(True, alpha=0.3)

ax3.plot(x, np.sin(x) + np.cos(x), color="mediumseagreen")
ax3.set_ylabel("sin + cos")
ax3.set_xlabel("x")
ax3.grid(True, alpha=0.3)

fig.suptitle("Trigonometric Functions", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 10. Styling: Labels, Limits, Grid, and `tight_layout`

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = np.linspace(0, 5, 50)
y = x ** 2 + np.random.randn(50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.6, color="steelblue", s=40)
ax.plot(x, x**2, color="red", linewidth=2, label="y = x²")

# Title and axis labels
ax.set_title("Noisy Quadratic Data", fontsize=16, fontweight="bold")
ax.set_xlabel("x value", fontsize=13)
ax.set_ylabel("y value", fontsize=13)

# Axis limits
ax.set_xlim(-0.2, 5.5)
ax.set_ylim(-2, 30)

# Tick labels
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=11)

# Grid
ax.grid(True, linestyle="--", alpha=0.4)

# Legend
ax.legend(fontsize=12, loc="upper left")

# tight_layout adjusts subplot params to avoid overlapping elements
plt.tight_layout()
plt.show()
```

### Common `legend` Locations

| `loc=` | Position |
|--------|----------|
| `"upper left"` | Top-left corner |
| `"upper right"` | Top-right corner (default) |
| `"lower left"` | Bottom-left |
| `"lower right"` | Bottom-right |
| `"center"` | Middle |
| `"best"` | Matplotlib picks least-cluttered spot |

---

## 11. Colors and Styles

### Named Colors and Hex Codes

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Named colors
axes[0].plot([1,2,3], [1,4,2], color="tomato",        linewidth=3, label="tomato")
axes[0].plot([1,2,3], [2,3,4], color="steelblue",     linewidth=3, label="steelblue")
axes[0].plot([1,2,3], [3,2,1], color="mediumseagreen", linewidth=3, label="mediumseagreen")
axes[0].legend(fontsize=9)
axes[0].set_title("Named Colors")

# Hex codes (like web colors)
axes[1].plot([1,2,3], [1,4,2], color="#E74C3C", linewidth=3, label="#E74C3C")
axes[1].plot([1,2,3], [2,3,4], color="#3498DB", linewidth=3, label="#3498DB")
axes[1].plot([1,2,3], [3,2,1], color="#2ECC71", linewidth=3, label="#2ECC71")
axes[1].legend(fontsize=9)
axes[1].set_title("Hex Colors")

# Short codes: 'r' 'g' 'b' 'c' 'm' 'y' 'k' 'w'
axes[2].plot([1,2,3], [1,4,2], color="r", linewidth=3, label="red 'r'")
axes[2].plot([1,2,3], [2,3,4], color="b", linewidth=3, label="blue 'b'")
axes[2].plot([1,2,3], [3,2,1], color="g", linewidth=3, label="green 'g'")
axes[2].legend(fontsize=9)
axes[2].set_title("Short Codes")

plt.tight_layout()
plt.show()
```

### Plot Styles

Matplotlib comes with built-in style sheets that change the overall look:

```python
import matplotlib.pyplot as plt
import numpy as np

# See all available styles
print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic',
#  'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
#  'grayscale', 'seaborn', 'seaborn-bright', ...]

# Apply a style before creating plots
plt.style.use("seaborn-v0_8-whitegrid")

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Seaborn Style")
plt.show()

# Reset to default after
plt.style.use("default")
```

> **Tip:** `"seaborn-v0_8-whitegrid"` and `"ggplot"` are popular choices for clean, professional-looking plots. For presentations, `"dark_background"` can look striking.

---

## 12. Annotations

Annotations add text and arrows to call out specific points in your plot:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, color="steelblue", linewidth=2)

# Simple text at a coordinate
ax.text(np.pi/2, 1.05, "Maximum\n(π/2, 1)", ha="center", fontsize=11,
        color="darkred")

# Arrow annotation pointing to a feature
ax.annotate("Zero crossing",
            xy=(np.pi, 0),               # point to annotate
            xytext=(np.pi + 0.5, 0.4),  # where to put the text
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=11, color="black")

# Horizontal and vertical reference lines
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.axvline(x=np.pi, color="orange", linestyle=":", linewidth=1.5,
           label=f"x = π")

ax.set_title("Annotated Sine Wave", fontsize=14)
ax.set_xlabel("x (radians)")
ax.set_ylabel("sin(x)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Shading a Region

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x) * np.exp(-x * 0.1)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, y, color="steelblue", linewidth=2)

# Shade the area between y and 0
ax.fill_between(x, y, 0, where=(y > 0), alpha=0.3, color="steelblue",
                label="Positive area")
ax.fill_between(x, y, 0, where=(y < 0), alpha=0.3, color="tomato",
                label="Negative area")

ax.axhline(0, color="black", linewidth=1)
ax.set_title("Damped Sine with Shaded Areas")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 13. Useful Patterns for ML

### Plotting Training Loss Curves

The most important plot in deep learning:

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate training history
epochs = np.arange(1, 51)
np.random.seed(42)

train_loss = 2.0 * np.exp(-epochs * 0.08) + np.random.randn(50) * 0.02
val_loss   = 2.0 * np.exp(-epochs * 0.06) + np.random.randn(50) * 0.03 + 0.1

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, train_loss, label="Training Loss",   color="steelblue", linewidth=2)
ax.plot(epochs, val_loss,   label="Validation Loss", color="tomato",    linewidth=2,
        linestyle="--")

# Mark where overfitting begins (where val loss starts going up)
best_epoch = int(np.argmin(val_loss)) + 1
ax.axvline(x=best_epoch, color="green", linestyle=":", linewidth=2,
           label=f"Best epoch ({best_epoch})")

ax.set_title("Training and Validation Loss", fontsize=14)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
```

> **Reading the loss curve:** If training loss keeps going down but validation loss starts going up, your model is **overfitting** — it is memorizing the training data instead of generalizing.

### Confusion Matrix Heatmap with `plt.imshow()`

```python
import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix (rows = actual, cols = predicted)
cm = np.array([
    [50,  2,  1],
    [ 3, 45,  4],
    [ 1,  5, 48]
])
class_names = ["Cat", "Dog", "Bird"]

fig, ax = plt.subplots(figsize=(7, 6))

# imshow displays a 2D array as a colored grid
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)

# Labels
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title("Confusion Matrix", fontsize=14)

# Annotate each cell with the count
for i in range(len(class_names)):
    for j in range(len(class_names)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, color=color, fontweight="bold")

plt.tight_layout()
plt.show()
```

> **Reading a confusion matrix:** Diagonal cells (top-left to bottom-right) show correct predictions. Off-diagonal cells show errors. A bright diagonal with dark off-diagonal means a good model.

### Plotting Feature Importance

```python
import matplotlib.pyplot as plt
import numpy as np

features   = ["Age", "Income", "Education", "Hours/Week", "Capital Gain",
              "Occupation", "Marital Status"]
importance = [0.12, 0.25, 0.08, 0.15, 0.22, 0.10, 0.08]

# Sort by importance
sorted_idx = np.argsort(importance)
sorted_features   = [features[i] for i in sorted_idx]
sorted_importance = [importance[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(sorted_features, sorted_importance, color="steelblue", edgecolor="white")

ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance")
ax.axvline(x=np.mean(importance), color="red", linestyle="--",
           label=f"Mean = {np.mean(importance):.2f}")
ax.legend()
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.show()
```

### ROC Curve Concept

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated ROC curve data
# In practice, compute this from sklearn.metrics.roc_curve()
fpr = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
tpr = np.array([0.0, 0.40, 0.65, 0.80, 0.88, 0.92, 0.95, 0.97, 1.0])
auc = 0.89

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(fpr, tpr, color="steelblue", linewidth=2,
        label=f"ROC Curve (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
        linewidth=1.5, label="Random Classifier (AUC = 0.5)")

ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 14. Seaborn — Built on Matplotlib

Seaborn is a statistical visualization library built on top of Matplotlib. It produces beautiful plots with less code and adds useful statistical visualizations that Matplotlib does not have natively.

### Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

### `sns.heatmap()` — Correlation Matrices

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)

# Create a DataFrame with correlated features
n = 100
df = pd.DataFrame({
    "age":     np.random.randint(20, 60, n),
    "income":  np.random.randint(30000, 120000, n),
    "score":   np.random.randn(n) * 10 + 70,
    "hours":   np.random.randint(20, 60, n),
})
df["income"] = df["income"] + df["age"] * 500   # income correlated with age

# Compute correlation matrix
corr = df.corr()

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    corr,
    annot=True,        # show values in each cell
    fmt=".2f",         # format as 2 decimal places
    cmap="coolwarm",   # diverging colormap: blue=negative, red=positive
    vmin=-1, vmax=1,   # anchor the colorscale to [-1, 1]
    linewidths=0.5,
    ax=ax
)

ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()
```

> **Reading a heatmap:** Values close to +1 (dark red) mean two features move together. Values close to -1 (dark blue) mean they move in opposite directions. Values near 0 mean no linear relationship. Highly correlated features can hurt some models (multicollinearity).

### `sns.pairplot()` — Pairwise Relationships

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50

df = pd.DataFrame({
    "feature_1": np.random.randn(n),
    "feature_2": np.random.randn(n),
    "feature_3": np.random.randn(n),
    "class":     np.random.choice(["A", "B"], n)
})
# Make feature_2 correlated with feature_1
df["feature_2"] = df["feature_1"] * 0.7 + df["feature_2"] * 0.3

# pairplot creates a grid of scatter plots (off-diagonal) and histograms (diagonal)
g = sns.pairplot(df, hue="class", palette={"A": "steelblue", "B": "tomato"},
                 diag_kind="hist", plot_kws={"alpha": 0.6})
g.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14)
plt.show()
```

### Other Useful Seaborn Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)
df = pd.DataFrame({
    "dept":  ["Eng"]*30 + ["Mkt"]*30 + ["Design"]*30,
    "salary": np.concatenate([
        np.random.normal(80000, 10000, 30),
        np.random.normal(65000, 8000, 30),
        np.random.normal(60000, 7000, 30)
    ])
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Violin plot — like a box plot with the distribution shape shown
sns.violinplot(data=df, x="dept", y="salary", palette="Set2", ax=axes[0])
axes[0].set_title("Salary by Department (Violin)")

# Box plot
sns.boxplot(data=df, x="dept", y="salary", palette="Set2", ax=axes[1])
axes[1].set_title("Salary by Department (Box)")

plt.tight_layout()
plt.show()
```

---

## 15. Saving and Exporting Figures

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.random.randn(100).cumsum(), color="steelblue", linewidth=2)
ax.set_title("Random Walk")
ax.grid(True, alpha=0.3)

# PNG — raster format, good for documents/slides
fig.savefig("plot.png", dpi=200, bbox_inches="tight")

# PDF — vector format, perfect for academic papers (scales infinitely)
fig.savefig("plot.pdf", bbox_inches="tight")

# SVG — vector format, good for web
fig.savefig("plot.svg", bbox_inches="tight")

plt.show()
```

---

## 16. A Complete ML Visualization Workflow

Here is everything together: load data, visualize distributions, correlation, and a model result:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# Simulate a dataset
df = pd.DataFrame({
    "feature_1": np.random.randn(n),
    "feature_2": np.random.randn(n),
    "label":     np.random.randint(0, 2, n)
})
# Make the classes separable
df.loc[df["label"] == 1, "feature_1"] += 1.5
df.loc[df["label"] == 1, "feature_2"] += 1.5

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Feature distributions by class
for label_val, color, name in [(0, "steelblue", "Class 0"),
                                 (1, "tomato",    "Class 1")]:
    subset = df[df["label"] == label_val]
    axes[0].hist(subset["feature_1"], bins=20, alpha=0.6,
                 color=color, label=name, density=True)

axes[0].set_title("Feature 1 Distribution by Class")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Density")
axes[0].legend()

# Plot 2: Scatter of both features
for label_val, color, name in [(0, "steelblue", "Class 0"),
                                 (1, "tomato",    "Class 1")]:
    subset = df[df["label"] == label_val]
    axes[1].scatter(subset["feature_1"], subset["feature_2"],
                    color=color, label=name, alpha=0.6, s=25)

axes[1].set_title("Feature Scatter Plot")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Simulated confusion matrix
cm = np.array([[82, 18], [12, 88]])
im = axes[2].imshow(cm, cmap="Blues")
fig.colorbar(im, ax=axes[2])
for i in range(2):
    for j in range(2):
        color = "white" if cm[i,j] > 50 else "black"
        axes[2].text(j, i, cm[i,j], ha="center", va="center",
                     fontsize=16, color=color, fontweight="bold")
axes[2].set_xticks([0,1]); axes[2].set_xticklabels(["Pred 0", "Pred 1"])
axes[2].set_yticks([0,1]); axes[2].set_yticklabels(["True 0", "True 1"])
axes[2].set_title("Confusion Matrix")

fig.suptitle("ML Visualization Workflow", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## Summary

| Chart type | Best for | Key function |
|------------|----------|-------------|
| Line plot | Trends, time series, loss curves | `ax.plot()` |
| Scatter plot | Relationships between two variables, class visualization | `ax.scatter()` |
| Bar chart | Comparing categories | `ax.bar()`, `ax.barh()` |
| Histogram | Distribution of one variable | `ax.hist()` |
| Box plot | Distribution comparison across groups | `ax.boxplot()` |
| Heatmap | Correlation matrix, confusion matrix | `ax.imshow()`, `sns.heatmap()` |
| Pairplot | All pairwise relationships | `sns.pairplot()` |

**Key Matplotlib concepts:**
- `fig, ax = plt.subplots()` — always use the object-oriented API
- `plt.tight_layout()` — prevents overlapping labels
- `fig.savefig()` before `plt.show()` — saves before clearing
- `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()` — always label your axes
- `ax.legend()` — always add a legend when you have multiple series

> **Next steps:** You now have the three core data science tools: NumPy (arrays and math), Pandas (data cleaning and manipulation), and Matplotlib (visualization). The next lessons cover machine learning algorithms — starting with the math you will need, then building actual models.
