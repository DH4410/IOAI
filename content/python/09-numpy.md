---
title: NumPy - Arrays and Math
track: python
order: 9
estimatedTime: 35
difficulty: intermediate
---

# NumPy - Arrays and Math

NumPy gives Python a fast array object. Almost every ML library (scikit-learn, PyTorch, Pandas) uses NumPy under the hood, so you need to know the basics before moving on.

---

## 1. Creating Arrays

Import NumPy first:

```python
import numpy as np
```

The three ways you will use most:

```python
# From a Python list
a = np.array([1, 2, 3, 4])

# Range of numbers
b = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]

# Grid of evenly spaced points
c = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# All zeros or ones (useful for initializing)
zeros = np.zeros((3, 4))   # 3 rows, 4 columns, all 0.0
ones  = np.ones((2, 3))
```

**Quick check:** What shape does `np.zeros((3, 4))` produce?
> 3 rows, 4 columns. The argument is always `(rows, columns)`.

NumPy arrays are faster than Python lists for math because they store all values as the same type (usually 64-bit floats) in a continuous block of memory.

---

## 2. Array Properties

Every array has three things worth checking:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)   # (2, 3)  - 2 rows, 3 columns
print(a.ndim)    # 2       - number of dimensions
print(a.dtype)   # int64   - type of each element
print(a.size)    # 6       - total number of elements
```

The `.shape` attribute is the one you will check constantly.

---

## 3. Indexing and Slicing

NumPy indexing works just like Python lists, but you can index multiple dimensions at once.

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Single element: [row, column]
print(a[0, 2])    # 30

# Entire row
print(a[1, :])    # [40, 50, 60]

# Entire column
print(a[:, 0])    # [10, 40, 70]

# Slice: first two rows, last two columns
print(a[:2, 1:])  # [[20, 30], [50, 60]]
```

**Boolean indexing** - pick elements that meet a condition:

```python
scores = np.array([45, 82, 67, 91, 55])
print(scores[scores > 60])   # [82, 67, 91]
```

This is extremely useful for filtering data.

---

## 4. Math and Broadcasting

You can do math on the whole array without a loop:

```python
a = np.array([1, 2, 3, 4])
print(a * 2)        # [2, 4, 6, 8]
print(a + 10)       # [11, 12, 13, 14]
print(a ** 2)       # [1, 4, 9, 16]
print(np.sqrt(a))   # [1.0, 1.41, 1.73, 2.0]
```

**Broadcasting** means NumPy automatically applies a scalar (single number) to every element in the array. It also works between arrays of different but compatible shapes:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

# b is broadcast across every row of a
print(a + b)
# [[11, 22, 33],
#  [14, 25, 36]]
```

**Quick check:** If `a.shape` is `(3, 4)` and `b.shape` is `(4,)`, will `a + b` work?
> Yes. NumPy adds `b` to each of the 3 rows of `a`.

Useful aggregation functions:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.sum())        # 21 - sum of everything
print(a.sum(axis=0))  # [5, 7, 9] - sum each column
print(a.sum(axis=1))  # [6, 15]   - sum each row
print(a.mean())       # 3.5
print(a.max())        # 6
print(a.argmax())     # 5 (flat index of the maximum)
```

`axis=0` goes down the rows (across columns). `axis=1` goes across the columns (across rows). This trips people up at first - just try both and see which direction makes sense for your problem.

---

## 5. Reshaping Arrays

Reshaping changes the shape without changing the data:

```python
a = np.arange(12)       # [0, 1, 2, ..., 11]
b = a.reshape(3, 4)     # 3 rows, 4 columns
c = a.reshape(2, 2, 3)  # 3D array
```

The rule: the total number of elements must stay the same. `12 = 3x4 = 2x2x3`.

Use `-1` to let NumPy figure out one dimension:

```python
a = np.arange(12)
b = a.reshape(-1, 4)    # NumPy calculates 3 rows
c = a.reshape(3, -1)    # NumPy calculates 4 columns
```

Flatten back to 1D:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.flatten())   # [1, 2, 3, 4, 5, 6]
print(a.ravel())     # same, but shares memory when possible (faster)
```

---

## 6. Matrix Math

For ML you often need matrix operations:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication (NOT element-wise)
C = A @ B           # or np.matmul(A, B)
print(C)
# [[19, 22],
#  [43, 50]]

# Transpose
print(A.T)
# [[1, 3],
#  [2, 4]]

# Dot product of two 1D vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(v1 @ v2)   # 1*4 + 2*5 + 3*6 = 32
```

**Quick check:** What is the difference between `A * B` and `A @ B`?
> `A * B` multiplies element by element. `A @ B` does true matrix multiplication (rows of A dotted with columns of B).

---

## Sort NumPy Operations

```widget
{
  "type": "concept-sort",
  "title": "Element-wise or Reduction? Which Operation?",
  "categories": [
    { "name": "Element-wise (same shape output)", "color": "#5B5BD6" },
    { "name": "Reduction (collapses dimension)", "color": "#F97316" }
  ],
  "items": [
    { "text": "a * 2 (multiply all values by 2)", "category": "Element-wise (same shape output)" },
    { "text": "a.sum(axis=0) (sum down columns)", "category": "Reduction (collapses dimension)" },
    { "text": "np.sqrt(a) (square root each value)", "category": "Element-wise (same shape output)" },
    { "text": "a.mean() (one average number)", "category": "Reduction (collapses dimension)" },
    { "text": "a + b (broadcast-add two arrays)", "category": "Element-wise (same shape output)" },
    { "text": "a.max(axis=1) (max per row)", "category": "Reduction (collapses dimension)" }
  ]
}
```

---

## Summary

| What you need | How to do it |
|---|---|
| Create array from list | `np.array([1, 2, 3])` |
| Create zeros/ones | `np.zeros((rows, cols))` |
| Check dimensions | `a.shape`, `a.ndim` |
| Single element | `a[row, col]` |
| Whole row | `a[i, :]` |
| Filter by condition | `a[a > 5]` |
| Math on whole array | `a * 2`, `a + 10`, `np.sqrt(a)` |
| Sum by column | `a.sum(axis=0)` |
| Sum by row | `a.sum(axis=1)` |
| Reshape | `a.reshape(rows, cols)` |
| Matrix multiply | `A @ B` |
| Transpose | `A.T` |

The next lessons build directly on this. When you see `X.shape`, `axis=`, and `@` in ML code, you will know exactly what is happening.
