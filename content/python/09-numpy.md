---
title: NumPy — Arrays and Math
track: python
order: 9
estimatedTime: 50
difficulty: intermediate
---

# NumPy — Arrays and Math

NumPy is the foundation of nearly every machine learning and data science library in Python. Before you can use scikit-learn, TensorFlow, PyTorch, or Pandas, you need to understand NumPy. This lesson teaches you everything you need to get started.

---

## 1. What Is NumPy and Why Does It Matter?

NumPy stands for **Numerical Python**. It gives Python a powerful array object and a large collection of mathematical functions that operate on those arrays.

### The Speed Problem with Python Lists

Python lists are flexible — they can hold any mix of types — but that flexibility comes at a cost: **slowness**. When you do math on a Python list, Python has to inspect each element individually, figure out its type, and then apply the operation.

Compare the two approaches:

```python
import time

# Python list approach
data = list(range(1_000_000))

start = time.time()
result = [x * 2 for x in data]
end = time.time()
print(f"Python list: {end - start:.4f} seconds")

# NumPy approach
import numpy as np
np_data = np.arange(1_000_000)

start = time.time()
result = np_data * 2
end = time.time()
print(f"NumPy array: {end - start:.4f} seconds")
```

On a typical laptop, NumPy is **10 to 100 times faster** than pure Python for numerical operations. For a dataset with a million points, that difference is the gap between a program that feels instant and one that feels slow.

### Why Is NumPy Faster?

NumPy arrays store data in **contiguous blocks of memory**, all of the same type. This means:

1. **No type checking per element** — NumPy already knows every element is, say, a 64-bit float.
2. **Vectorized operations** — Instead of looping in Python, NumPy calls highly optimized C (and sometimes Fortran) code under the hood.
3. **Cache efficiency** — Contiguous memory means the CPU cache can load many elements at once.

### Vectorization: The Key Idea

"Vectorization" means applying an operation to an entire array at once, rather than looping element by element. Here is the mental shift:

```python
# Non-vectorized (slow, Python loop)
scores = [72, 85, 91, 60, 78]
normalized = []
for s in scores:
    normalized.append((s - 60) / (100 - 60))
print(normalized)

# Vectorized (fast, NumPy)
import numpy as np
scores = np.array([72, 85, 91, 60, 78])
normalized = (scores - 60) / (100 - 60)
print(normalized)
# Output: [0.3   0.625 0.775 0.    0.45 ]
```

The NumPy version is not just faster — it is also shorter and easier to read. This is what professional ML code looks like.

> **Note:** The concept of vectorization carries over to PyTorch and TensorFlow. Learning NumPy first makes those libraries much easier to understand.

---

## 2. Creating Arrays

### `np.array()` — From Python Data

The most basic way to create a NumPy array is from a Python list:

```python
import numpy as np

# 1D array
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(type(a))    # <class 'numpy.ndarray'>

# 2D array (matrix)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# You can specify the data type
floats = np.array([1, 2, 3], dtype=float)
print(floats)   # [1. 2. 3.]

bools = np.array([True, False, True])
print(bools)    # [ True False  True]
```

### `np.zeros()` and `np.ones()` — Initialized Arrays

When you know the shape of an array but want to start with all zeros or ones:

```python
import numpy as np

# Zeros
z = np.zeros(5)
print(z)   # [0. 0. 0. 0. 0.]

# Zeros matrix (3 rows, 4 columns)
z2d = np.zeros((3, 4))
print(z2d)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Ones
o = np.ones((2, 3))
print(o)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# A constant value other than 0 or 1: use np.full()
fives = np.full((2, 4), 5)
print(fives)
# [[5 5 5 5]
#  [5 5 5 5]]

# Identity matrix (1s on diagonal, 0s elsewhere)
identity = np.eye(4)
print(identity)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

> **Tip:** `np.zeros` and `np.ones` are commonly used to pre-allocate arrays before filling them with computed values. This is much faster than building a list and converting it.

### `np.arange()` — Like Python's `range()`

```python
import numpy as np

# np.arange(stop)
a = np.arange(10)
print(a)   # [0 1 2 3 4 5 6 7 8 9]

# np.arange(start, stop)
b = np.arange(2, 8)
print(b)   # [2 3 4 5 6 7]

# np.arange(start, stop, step)
c = np.arange(0, 1, 0.1)
print(c)   # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

# Counting down
d = np.arange(10, 0, -2)
print(d)   # [10  8  6  4  2]
```

### `np.linspace()` — Evenly Spaced Values

`linspace` (linear space) creates a specified number of evenly spaced values between a start and end:

```python
import numpy as np

# 5 values between 0 and 1, inclusive
a = np.linspace(0, 1, 5)
print(a)   # [0.   0.25 0.5  0.75 1.  ]

# 10 values between -3 and 3
b = np.linspace(-3, 3, 10)
print(b)
# [-3.         -2.33333333 -1.66666667 -1.         -0.33333333
#   0.33333333  1.          1.66666667  2.33333333  3.        ]
```

> **Tip:** Use `np.linspace` when you need a fixed number of points (like for plotting a smooth curve). Use `np.arange` when you care about the step size.

### `np.random` — Random Arrays

Random arrays are essential for initializing weights in neural networks and for generating test data:

```python
import numpy as np

# Random floats between 0 and 1 (uniform distribution)
np.random.seed(42)   # Set seed for reproducibility
a = np.random.rand(3, 4)
print(a)
# [[0.374 0.951 0.732 0.599]
#  [0.156 0.058 0.866 0.709]
#  [0.020 0.970 0.832 0.212]]

# Random integers
b = np.random.randint(low=1, high=10, size=(2, 5))
print(b)
# [[6 1 4 4 8]
#  [1 9 9 8 9]]

# Standard normal distribution (mean=0, std=1)
c = np.random.randn(3, 3)
print(c)
# [[ 0.497 -0.138  0.648]
#  [ 1.523  1.469 -0.234]
#  [-0.234 -0.234  0.234]]

# Normal distribution with custom mean and std
# mean=100, std=15 (like IQ scores)
iq_scores = np.random.normal(loc=100, scale=15, size=1000)
print(iq_scores.mean())  # ~100
print(iq_scores.std())   # ~15

# Shuffle an existing array (in-place)
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)  # e.g. [3 7 0 8 2 6 1 9 4 5]

# Random choice from an array
choices = np.random.choice([10, 20, 30, 40], size=6, replace=True)
print(choices)  # e.g. [30 10 40 10 20 30]
```

> **Important:** Always set `np.random.seed()` at the beginning of experiments so that your results are reproducible. Without a seed, every run gives different numbers.

---

## 3. Array Properties

Every NumPy array has attributes that tell you about its structure:

```python
import numpy as np

a = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print(a.shape)    # (2, 3) — 2 rows, 3 columns
print(a.ndim)     # 2 — two-dimensional
print(a.size)     # 6 — total number of elements
print(a.dtype)    # float64 — 64-bit floating point

b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(b.shape)    # (2, 2, 2) — 3D array: 2 "sheets", 2 rows, 2 cols
print(b.ndim)     # 3
print(b.size)     # 8
```

### Common dtypes

| dtype | Meaning | Typical use |
|-------|---------|-------------|
| `int32` | 32-bit integer | Integer labels |
| `int64` | 64-bit integer | Default integer |
| `float32` | 32-bit float | GPU training (saves memory) |
| `float64` | 64-bit float | Default float, high precision |
| `bool` | True/False | Boolean masks |
| `complex128` | Complex number | Signal processing |

```python
import numpy as np

# Change dtype
a = np.array([1, 2, 3])
print(a.dtype)   # int64 (or int32 on Windows)

b = a.astype(float)
print(b.dtype)   # float64
print(b)         # [1. 2. 3.]

c = a.astype(np.float32)
print(c.dtype)   # float32 — uses half the memory of float64
```

---

## 4. Indexing and Slicing

### 1D Array Indexing

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50])

print(a[0])     # 10 — first element
print(a[-1])    # 50 — last element
print(a[1:4])   # [20 30 40] — slice from index 1 to 3
print(a[::2])   # [10 30 50] — every other element
print(a[::-1])  # [50 40 30 20 10] — reversed
```

### 2D Array Indexing

For 2D arrays, you use `[row, col]` notation:

```python
import numpy as np

m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(m[0, 0])    # 1 — top-left
print(m[1, 2])    # 6 — row 1, column 2
print(m[-1, -1])  # 9 — bottom-right

# Slicing rows and columns
print(m[0, :])    # [1 2 3] — entire first row
print(m[:, 1])    # [2 5 8] — entire second column
print(m[0:2, 1:3])
# [[2 3]
#  [5 6]]

# All rows, every other column
print(m[:, ::2])
# [[1 3]
#  [4 6]
#  [7 9]]
```

> **Note:** NumPy slices return **views**, not copies. Modifying a slice modifies the original array. To get an independent copy, use `.copy()`.

```python
import numpy as np

original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
view[0] = 99
print(original)   # [ 1 99  3  4  5] — original was changed!

# To avoid this:
safe_copy = original[1:4].copy()
safe_copy[0] = 0
print(original)   # [ 1 99  3  4  5] — unchanged
```

### Boolean Indexing (Masking)

This is one of the most powerful features in NumPy — and it is used constantly in ML for filtering data:

```python
import numpy as np

scores = np.array([72, 85, 91, 60, 78, 95, 55, 88])

# Create a boolean mask
mask = scores >= 80
print(mask)   # [False  True  True False False  True False  True]

# Apply the mask to select elements
high_scores = scores[mask]
print(high_scores)   # [85 91 95 88]

# You can do this in one line
print(scores[scores >= 80])   # [85 91 95 88]

# Multiple conditions
print(scores[(scores >= 70) & (scores < 90)])   # [72 85 78 88]

# Set all failing scores to 0
scores[scores < 70] = 0
print(scores)   # [ 72  85  91   0  78  95   0  88]
```

### Fancy Indexing

Fancy indexing means using an array of indices to select elements:

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])

# Select elements at indices 0, 2, 4
indices = [0, 2, 4]
print(a[indices])   # [10 30 50]

# Works for 2D too
m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Select rows 0 and 2
print(m[[0, 2]])
# [[1 2 3]
#  [7 8 9]]

# Select specific (row, col) pairs
rows = [0, 1, 2]
cols = [0, 1, 2]
print(m[rows, cols])   # [1 5 9] — diagonal elements
```

---

## 5. Array Operations and Broadcasting

### Element-Wise Operations

When two arrays have the **same shape**, operations happen element by element:

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)    # [11 22 33 44]
print(a * b)    # [ 10  40  90 160]
print(b / a)    # [10. 10. 10. 10.]
print(a ** 2)   # [ 1  4  9 16]
print(np.sqrt(a))  # [1.    1.414 1.732 2.   ]
```

### Broadcasting — The Most Important Concept

Broadcasting lets NumPy perform operations on arrays of **different shapes** by "stretching" the smaller array to match the larger one. Understanding this is crucial for ML.

**Rule 1: Scalar and array**

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# The scalar 10 is "broadcast" to match the shape of a
print(a + 10)    # [11 12 13 14 15]
print(a * 3)     # [ 3  6  9 12 15]
print(a > 3)     # [False False False  True  True]
```

Think of it as if the scalar is copied 5 times: `[10, 10, 10, 10, 10]`.

**Rule 2: 1D array and 2D array**

```python
import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

row = np.array([10, 20, 30])

# row is broadcast across every row of matrix
result = matrix + row
print(result)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

Visually, `row` gets repeated 3 times (once per row):
```
[10 20 30]          [10 20 30]
           becomes  [10 20 30]
                    [10 20 30]
```

**Rule 3: Column vector and 2D array**

```python
import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Column vector — shape (3, 1)
col = np.array([[100], [200], [300]])

result = matrix + col
print(result)
# [[101 102 103]
#  [204 205 206]
#  [307 308 309]]
```

`col` gets broadcast across every column. Each row of the result gets a different value added.

**Broadcasting Rules (formal)**

Two shapes are compatible if, reading from the right, each dimension either:
- Matches exactly, or
- One of them is 1 (that dimension gets stretched)

```python
# Examples of compatible shapes:
# (3, 4) + (4,)     → OK: (4,) treated as (1, 4), broadcast to (3, 4)
# (3, 4) + (3, 1)   → OK: (3, 1) broadcast to (3, 4)
# (3, 4) + (1, 1)   → OK: scalar-like, broadcast to (3, 4)

# Incompatible:
# (3, 4) + (3,)     → ERROR: 4 ≠ 3 and neither is 1

import numpy as np
a = np.ones((3, 4))
b = np.ones((3,))
try:
    a + b
except ValueError as e:
    print(e)
# operands could not be broadcast together with shapes (3,4) (3,)
```

**Real-world broadcasting example: normalizing features**

```python
import numpy as np

# Feature matrix: 5 samples, 3 features
X = np.array([
    [2.0, 80.0, 1.5],
    [3.0, 90.0, 1.7],
    [1.5, 70.0, 1.6],
    [2.5, 85.0, 1.8],
    [4.0, 95.0, 1.9]
])

# Compute mean and std for each feature (axis=0 = along rows)
mean = X.mean(axis=0)   # shape (3,)
std  = X.std(axis=0)    # shape (3,)

print("Means:", mean)   # [2.6  84.  1.7]
print("Stds:", std)     # [0.83 8.54 0.14]

# Normalize: (X - mean) / std
# Broadcasting: both mean and std have shape (3,), broadcast to (5, 3)
X_normalized = (X - mean) / std
print(X_normalized)
```

---

## 6. Universal Functions (ufuncs)

NumPy's universal functions operate element-wise on arrays and are implemented in C, making them very fast.

### Aggregation Functions

```python
import numpy as np

a = np.array([4, 2, 7, 1, 9, 3, 6, 8, 5])

print(np.sum(a))      # 45
print(np.mean(a))     # 5.0
print(np.median(a))   # 5.0
print(np.std(a))      # 2.581...  (standard deviation)
print(np.var(a))      # 6.666...  (variance)
print(np.min(a))      # 1
print(np.max(a))      # 9
print(np.argmin(a))   # 3 — index of minimum value
print(np.argmax(a))   # 4 — index of maximum value
print(np.cumsum(a))   # [ 4  6 13 14 23 26 32 40 45]
print(np.cumprod(np.array([1, 2, 3, 4])))  # [ 1  2  6 24]
```

### Math Functions

```python
import numpy as np

a = np.array([0, 1, 2, 3])

print(np.sqrt(a))     # [0.    1.    1.414 1.732]
print(np.exp(a))      # [1.    2.718 7.389 20.086]
print(np.log(np.array([1, np.e, np.e**2])))  # [0. 1. 2.]
print(np.log2(np.array([1, 2, 4, 8])))       # [0. 1. 2. 3.]
print(np.log10(np.array([1, 10, 100])))       # [0. 1. 2.]
print(np.abs(np.array([-3, 1, -4, 1, -5])))  # [3 1 4 1 5]

# Trigonometry
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(angles))  # [0.    0.5   0.707 0.866 1.   ]
print(np.cos(angles))  # [1.    0.866 0.707 0.5   0.   ]
```

---

## 7. The Axis Parameter

The `axis` parameter is one of the most confusing parts of NumPy for beginners, but it is used constantly. Let's explain it visually.

Imagine a 2D array like a spreadsheet:

```
       col0  col1  col2
row0 [  1,    2,    3  ]
row1 [  4,    5,    6  ]
row2 [  7,    8,    9  ]
```

- **`axis=0`** means "collapse along the row dimension" — you reduce the rows into one, giving you one value **per column**.
- **`axis=1`** means "collapse along the column dimension" — you reduce the columns into one, giving you one value **per row**.

```python
import numpy as np

m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Sum with no axis — adds everything
print(np.sum(m))          # 45

# axis=0: sum each column (collapse rows)
print(np.sum(m, axis=0))  # [12 15 18]
# 1+4+7=12, 2+5+8=15, 3+6+9=18

# axis=1: sum each row (collapse columns)
print(np.sum(m, axis=1))  # [ 6 15 24]
# 1+2+3=6, 4+5+6=15, 7+8+9=24
```

> **Memory trick:** `axis=0` collapses rows → result has fewer rows. `axis=1` collapses columns → result has fewer columns.

More examples:

```python
import numpy as np

data = np.array([
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0],
    [70.0, 80.0, 90.0],
    [100.0, 110.0, 120.0]
])
# shape (4, 3): 4 students, 3 test scores each

# Average score per test (across all students)
print(data.mean(axis=0))   # [ 55.  65.  75.]

# Average score per student (across all tests)
print(data.mean(axis=1))   # [ 20.  50.  80. 110.]

# Which student got the highest average?
print(np.argmax(data.mean(axis=1)))   # 3 (last student)

# Standard deviation of each test (how spread out are the scores?)
print(data.std(axis=0))   # [33.54 33.54 33.54]
```

---

## 8. Reshaping Arrays

Reshaping lets you change the shape of an array without changing its data. This is critical for feeding data into ML models.

### `reshape()`

```python
import numpy as np

a = np.arange(12)
print(a)       # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(a.shape) # (12,)

# Reshape to 3x4 matrix
b = a.reshape(3, 4)
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Use -1 to let NumPy figure out one dimension
c = a.reshape(2, -1)   # 2 rows, NumPy computes columns (= 6)
print(c.shape)   # (2, 6)

d = a.reshape(-1, 3)   # NumPy computes rows (= 4), 3 columns
print(d.shape)   # (4, 3)

# 3D reshape
e = a.reshape(2, 2, 3)
print(e)
# [[[ 0  1  2]
#   [ 3  4  5]]
#  [[ 6  7  8]
#   [ 9 10 11]]]
```

### `flatten()` and `ravel()`

Both turn a multi-dimensional array into a 1D array:

```python
import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6]])

# flatten() returns a copy
flat = m.flatten()
flat[0] = 99
print(m[0, 0])   # 1 — original not changed

# ravel() returns a view (faster, but careful!)
rav = m.ravel()
rav[0] = 99
print(m[0, 0])   # 99 — original WAS changed
```

> **Tip:** In ML, you often need to flatten images. A 28×28 image becomes a 784-element vector using `.flatten()` or `.reshape(-1)`.

### `np.newaxis` — Adding a Dimension

`np.newaxis` inserts a new axis of size 1, which is useful for broadcasting:

```python
import numpy as np

a = np.array([1, 2, 3])
print(a.shape)   # (3,)

# Add row dimension: (3,) → (1, 3)
row = a[np.newaxis, :]
print(row.shape)   # (1, 3)

# Add column dimension: (3,) → (3, 1)
col = a[:, np.newaxis]
print(col.shape)   # (3, 1)
print(col)
# [[1]
#  [2]
#  [3]]

# Now col * row broadcasts to a 3x3 outer product
outer = col * row
print(outer)
# [[1 2 3]
#  [2 4 6]
#  [3 6 9]]
```

---

## 9. Stacking and Combining Arrays

### `np.concatenate()`

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate 1D
print(np.concatenate([a, b]))   # [1 2 3 4 5 6]

# Concatenate 2D along rows (axis=0)
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

print(np.concatenate([m1, m2], axis=0))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Concatenate along columns (axis=1)
print(np.concatenate([m1, m2], axis=1))
# [[1 2 5 6]
#  [3 4 7 8]]
```

### `np.vstack()` and `np.hstack()`

These are shortcuts for vertical (row-wise) and horizontal (column-wise) stacking:

```python
import numpy as np

a = np.array([[1, 2, 3]])   # shape (1, 3)
b = np.array([[4, 5, 6]])   # shape (1, 3)

# vstack: stack vertically (add rows)
print(np.vstack([a, b]))
# [[1 2 3]
#  [4 5 6]]

# hstack: stack horizontally (add columns)
print(np.hstack([a, b]))
# [[1 2 3 4 5 6]]

# Common ML use: add new samples to dataset
X = np.random.rand(100, 5)    # 100 samples, 5 features
X_new = np.random.rand(20, 5)  # 20 new samples
X_combined = np.vstack([X, X_new])
print(X_combined.shape)   # (120, 5)
```

### `np.stack()`

`np.stack` creates a **new axis** and stacks arrays along it:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Default: stack along new axis 0
result = np.stack([a, b])
print(result)
# [[1 2 3]
#  [4 5 6]]
print(result.shape)   # (2, 3)

# Stack along axis 1
result2 = np.stack([a, b], axis=1)
print(result2)
# [[1 4]
#  [2 5]
#  [3 6]]
print(result2.shape)   # (3, 2)
```

---

## 10. Matrix Operations

Linear algebra is the backbone of machine learning. NumPy has a full suite of matrix operations.

### Dot Product and Matrix Multiplication

```python
import numpy as np

# 1D vectors: dot product (sum of element-wise products)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32

# 2D matrices: matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# np.dot for 2D performs matrix multiplication
print(np.dot(A, B))
# [[19 22]
#  [43 50]]

# The @ operator does the same thing (Python 3.5+)
print(A @ B)
# [[19 22]
#  [43 50]]
```

> **Critical:** In ML, the core operation is `X @ W` where `X` is the feature matrix and `W` is the weight matrix. Understanding matrix multiplication is essential.

### Shape rules for matrix multiplication

For `A @ B` to work:
- A has shape `(m, k)`
- B has shape `(k, n)`
- Result has shape `(m, n)`

The inner dimensions must match!

```python
import numpy as np

# X: 100 samples, 10 features → shape (100, 10)
X = np.random.rand(100, 10)

# W: weights from 10 features to 5 outputs → shape (10, 5)
W = np.random.rand(10, 5)

# Output: 100 samples, 5 outputs → shape (100, 5)
output = X @ W
print(output.shape)   # (100, 5)
```

### Transpose

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)   # (2, 3)

# Transpose: swap rows and columns
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(A.T.shape)  # (3, 2)

# Equivalent
print(np.transpose(A).shape)  # (3, 2)
```

### Norms and Linear Algebra

```python
import numpy as np

v = np.array([3.0, 4.0])

# L2 norm (Euclidean length): sqrt(3² + 4²) = 5
print(np.linalg.norm(v))   # 5.0

# L1 norm: |3| + |4| = 7
print(np.linalg.norm(v, ord=1))   # 7.0

# Normalize a vector to unit length
v_normalized = v / np.linalg.norm(v)
print(v_normalized)         # [0.6 0.8]
print(np.linalg.norm(v_normalized))  # 1.0

# Matrix inverse
A = np.array([[2.0, 1.0], [1.0, 3.0]])
A_inv = np.linalg.inv(A)
print(A_inv)
# [[ 0.6 -0.2]
#  [-0.2  0.4]]

# Verify: A @ A_inv should be identity
print(np.round(A @ A_inv))
# [[1. 0.]
#  [0. 1.]]

# Solving linear systems: Ax = b
b = np.array([4.0, 5.0])
x = np.linalg.solve(A, b)
print(x)       # [1.4 1.2]
print(A @ x)   # [4. 5.] — verified!

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)   # [2.56 2.44] approx
print("Eigenvectors:\n", eigenvectors)

# Determinant
print(np.linalg.det(A))   # 5.0
```

---

## 11. Boolean Operations

### `np.where()`

`np.where` is like a vectorized `if-else`:

```python
import numpy as np

scores = np.array([72, 85, 91, 60, 78, 95, 55, 88])

# np.where(condition, value_if_true, value_if_false)
grades = np.where(scores >= 75, "Pass", "Fail")
print(grades)
# ['Pass' 'Pass' 'Pass' 'Fail' 'Pass' 'Pass' 'Fail' 'Pass']

# Clamp values between 0 and 100
raw = np.array([-5, 50, 105, 80, -10, 95])
clamped = np.where(raw < 0, 0, np.where(raw > 100, 100, raw))
print(clamped)   # [  0  50 100  80   0  95]

# Replace negative values with their absolute value (equivalent to np.abs)
a = np.array([-3, 1, -4, 1, -5, 9, 2, -6])
abs_a = np.where(a < 0, -a, a)
print(abs_a)   # [3 1 4 1 5 9 2 6]
```

### `np.any()` and `np.all()`

```python
import numpy as np

a = np.array([1, 0, 3, 0, 5])

# any: is at least one element True (or nonzero)?
print(np.any(a > 4))    # True  (5 > 4)
print(np.any(a > 10))   # False

# all: are ALL elements True (or nonzero)?
print(np.all(a > 0))    # False (0 values present)
print(np.all(a >= 0))   # True

# With axis:
m = np.array([[1, 0], [3, 4]])
print(np.any(m == 0, axis=0))   # [ True False] — col 0 has no zeros
print(np.any(m == 0, axis=1))   # [ True False] — row 0 has a zero
```

---

## 12. Practical ML Uses

### Building a Feature Matrix

In ML, your data is always a 2D array: **rows are samples, columns are features**.

```python
import numpy as np

# 5 houses: [area_m2, num_rooms, age_years, distance_to_center_km]
X = np.array([
    [85.0, 3, 10, 2.5],
    [120.0, 4, 5, 8.0],
    [60.0, 2, 25, 1.0],
    [200.0, 5, 2, 15.0],
    [95.0, 3, 8, 4.0]
])

# Labels (prices in thousands)
y = np.array([250.0, 380.0, 160.0, 550.0, 280.0])

print("Features shape:", X.shape)   # (5, 4)
print("Labels shape:", y.shape)     # (5,)
print("Number of samples:", X.shape[0])   # 5
print("Number of features:", X.shape[1])  # 4
```

### One-Hot Encoding

ML models often need categorical variables as binary vectors. For example, "cat", "dog", "bird" become `[1,0,0]`, `[0,1,0]`, `[0,0,1]`:

```python
import numpy as np

# Class labels: 0=cat, 1=dog, 2=bird
labels = np.array([0, 1, 2, 1, 0, 2, 0])

num_classes = 3
n_samples = len(labels)

# Create the one-hot matrix
one_hot = np.zeros((n_samples, num_classes))
one_hot[np.arange(n_samples), labels] = 1

print(one_hot)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
```

### Normalizing Data

Before training a model, it is usually essential to normalize (standardize) the features:

```python
import numpy as np

np.random.seed(0)
X = np.random.randn(100, 4) * np.array([100, 1, 0.01, 50])  # Different scales

# Min-max normalization: scale to [0, 1]
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_minmax = (X - X_min) / (X_max - X_min)
print("Min-max range:", X_minmax.min(axis=0), "to", X_minmax.max(axis=0))

# Z-score normalization (standardization): mean=0, std=1
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_standard = (X - X_mean) / X_std

print("Mean after standardization:", X_standard.mean(axis=0).round(2))
print("Std after standardization:", X_standard.std(axis=0).round(2))
```

### Splitting Data into Train/Test Sets

```python
import numpy as np

np.random.seed(42)

# 200 samples, 5 features
X = np.random.randn(200, 5)
y = np.random.randint(0, 2, 200)

# Shuffle indices
indices = np.arange(200)
np.random.shuffle(indices)

# 80% train, 20% test
split = int(0.8 * 200)   # = 160
train_indices = indices[:split]
test_indices  = indices[split:]

X_train = X[train_indices]
X_test  = X[test_indices]
y_train = y[train_indices]
y_test  = y[test_indices]

print("Training set:", X_train.shape, y_train.shape)   # (160, 5) (160,)
print("Test set:    ", X_test.shape, y_test.shape)      # (40, 5)  (40,)
```

### Implementing a Simple Linear Regression with NumPy

To see how all these pieces fit together, here is a complete linear regression from scratch:

```python
import numpy as np

# Generate synthetic data: y = 2x + 1 + noise
np.random.seed(0)
n = 100
X = np.random.rand(n, 1)          # shape (100, 1)
y = 2 * X.squeeze() + 1 + np.random.randn(n) * 0.3   # shape (100,)

# Add bias column (column of ones)
X_b = np.hstack([np.ones((n, 1)), X])   # shape (100, 2)

# Solve normal equation: w = (X^T X)^{-1} X^T y
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Intercept: {w[0]:.3f}")   # ~1.0
print(f"Slope:     {w[1]:.3f}")   # ~2.0

# Make predictions
y_pred = X_b @ w

# Mean squared error
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")
```

---

## Summary

Here is what you have learned:

| Topic | Key functions |
|-------|--------------|
| Creating arrays | `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.random.rand()` |
| Properties | `.shape`, `.dtype`, `.ndim`, `.size` |
| Indexing | `a[i]`, `a[i,j]`, boolean mask, fancy indexing |
| Operations | `+`, `-`, `*`, `/`, `**`, broadcasting |
| Aggregation | `np.sum()`, `np.mean()`, `np.std()`, `np.argmax()` |
| Reshaping | `.reshape()`, `.flatten()`, `.ravel()`, `np.newaxis` |
| Stacking | `np.concatenate()`, `np.vstack()`, `np.hstack()` |
| Linear algebra | `np.dot()`, `@`, `.T`, `np.linalg.inv()`, `np.linalg.norm()` |
| Logic | `np.where()`, `np.any()`, `np.all()` |

NumPy is the language of ML data. Every time you work with a dataset, a feature matrix, or model weights, you will use NumPy. Master these operations and the rest of the ML stack will feel natural.

> **Next steps:** In the next lesson you will learn Pandas, which is built on top of NumPy and adds powerful tools for loading, cleaning, and exploring tabular data.
