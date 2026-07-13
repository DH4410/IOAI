---
title: Vectors and Matrices
track: math
order: 1
estimatedTime: 50
difficulty: intermediate
---

# Vectors and Matrices

Almost everything in AI is built out of two objects: **vectors** and **matrices**. If you understand these two things really well, a huge amount of AI stops feeling like magic and starts feeling like arithmetic.

That's not an exaggeration. When a neural network "thinks," what it's actually doing under the hood is multiplying matrices together, over and over, billions of times. When an image gets turned into something a model can understand, it becomes a grid of numbers — a matrix. When a word gets turned into meaning, it becomes a list of numbers — a vector.

So let's slow down and really get comfortable with these. No rushing. By the end of this lesson you'll be able to add them, scale them, multiply them, and — most importantly — *picture* what they're doing.

---

## What is a vector?

Start with the simplest possible idea. A **vector** is just a list of numbers.

That's it. Here's one:

$$\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

This vector has two numbers in it: a 3 and a 4. We say it's a **2-dimensional** vector, or "2D," because it has 2 numbers.

Here's a bigger one:

$$\vec{w} = \begin{bmatrix} 1 \\ 5 \\ 2 \\ 9 \end{bmatrix}$$

This one is 4-dimensional. Four numbers stacked up.

In Python with numpy, a vector is just an array:

```python
import numpy as np

v = np.array([3, 4])
w = np.array([1, 5, 2, 9])

print(v)          # [3 4]
print(w)          # [1 5 2 9]
print(v.shape)    # (2,)  -> 2 numbers
print(w.shape)    # (4,)  -> 4 numbers
```

Run that. The `.shape` tells you how many numbers are inside.

### Two ways to think about a vector

A vector is a list of numbers, yes. But there are two *pictures* in your head that make vectors click.

**Picture 1: a vector is an arrow.**

Take $\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$. Imagine a flat piece of graph paper. Start at the origin (the point $(0,0)$, the center) and draw an arrow that goes 3 steps to the right and 4 steps up. That arrow *is* the vector.

So a vector has two things: a **direction** (which way the arrow points) and a **length** (how long the arrow is). The vector $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$ points up-and-to-the-right, and it happens to have length 5 (we'll see why later — it's the Pythagorean theorem).

**Picture 2: a vector is a point.**

Instead of an arrow, you can just think of $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$ as *the location* you end up at: the point 3 across, 4 up. Arrow or destination — same information.

Both pictures are useful. Sometimes "arrow" helps (adding vectors), sometimes "point" helps (a data point in a dataset).

### Why vectors matter for AI

Here's where it gets real. In AI, almost every piece of data becomes a vector.

- A **word** like "dog" gets turned into a vector of maybe 768 numbers. This is called an *embedding*. Words with similar meanings end up as vectors that point in similar directions.
- An **image** can be flattened into a vector — a 28×28 grayscale image becomes a vector of 784 numbers.
- A **user** on a website might be a vector of their preferences.
- A **row in a spreadsheet** — one house, with its price, size, number of rooms — is a vector.

When people say a model works in "high-dimensional space," they mean the vectors have a *lot* of numbers in them. A 768-dimensional vector is just a list of 768 numbers. You can't picture 768 dimensions (nobody can), but the math works exactly the same as the 2D arrow you *can* picture. That's the trick: learn it in 2D, trust it in 768D.

---

## What is a matrix?

A **matrix** is a grid of numbers. Rows and columns.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

This matrix has **2 rows** and **3 columns**. We say it's a "2×3 matrix" (read "two by three"). Rows first, then columns — always in that order. Remember it as "RC," like "Roman Column" or "Row then Column."

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A)
print(A.shape)   # (2, 3)  -> 2 rows, 3 columns
```

The `.shape` is `(2, 3)`: 2 rows, 3 columns.

### A vector is just a skinny matrix

Notice: a vector is really just a matrix with one column (or one row). The vector $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$ is a 2×1 matrix. So everything you learn about matrices also applies to vectors. They're the same family of thing.

### Reading a matrix: rows and columns

We use subscripts to point at a single number inside a matrix. $A_{ij}$ means "the number in row $i$, column $j$." Rows and columns are counted starting from 1 in math notation.

For our matrix $A$ above:
- $A_{11} = 1$ (row 1, column 1)
- $A_{23} = 6$ (row 2, column 3)
- $A_{12} = 2$ (row 1, column 2)

> **Careful:** Math counts from 1, but Python counts from 0. In numpy, `A[0, 0]` is the top-left (the 1), and `A[1, 2]` is the bottom-right (the 6). This off-by-one is the single most common source of bugs when you move between math on paper and code. Keep it in mind.

```python
print(A[0, 0])   # 1  (math would call this A_11)
print(A[1, 2])   # 6  (math would call this A_23)
print(A[0])      # [1 2 3]  -> the whole first row
print(A[:, 1])   # [2 5]    -> the whole second column
```

### Why matrices matter for AI

Two huge reasons.

**Reason 1: data comes in tables, and a table is a matrix.** Imagine a spreadsheet of houses. Each row is a house. Each column is a feature (price, size, bedrooms). That whole spreadsheet is one big matrix. If you have 1000 houses and 5 features, that's a 1000×5 matrix. This is *the* standard way data enters an ML model.

**Reason 2: every neural network layer is a matrix.** We'll build to this at the end of the lesson. The "weights" of a neural network — the numbers it learns during training — are stored in matrices. When the network processes your data, it multiplies your data (a vector) by these weight matrices. That's the core computation. Get matrix multiplication and you understand the beating heart of deep learning.

---

## Adding vectors

Adding two vectors is beautifully simple: **add them position by position.**

$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 1+3 \\ 2+4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

First number plus first number, second plus second. Done.

```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])

print(a + b)   # [4 6]
```

### The one rule: shapes must match

You can only add two vectors if they're the same length. Adding a 2D vector to a 3D vector makes no sense — there's no fourth position to pair the third number with.

```python
a = np.array([1, 2])
b = np.array([3, 4, 5])
# a + b  -> ERROR: shapes (2,) and (3,) don't match
```

numpy will complain loudly. Good. That error is protecting you from nonsense.

### The picture: tip to tail

Remember vectors are arrows. Adding them has a lovely visual meaning: **put them tip to tail.**

Draw arrow $\vec{a}$ starting at the origin. Now draw arrow $\vec{b}$ starting *from the tip of* $\vec{a}$. Where you end up is $\vec{a} + \vec{b}$. Walking 3-east-2-north and then 1-east-4-north gets you to the same place as walking 4-east-6-north directly.

This is exactly how movement combines in physics and games — and it's how a model combines different pieces of evidence, each nudging the result in some direction.

### Adding matrices

Same idea, just a grid instead of a list. Add matching positions:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix} = \begin{bmatrix} 11 & 22 \\ 33 & 44 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20], [30, 40]])

print(A + B)
# [[11 22]
#  [33 44]]
```

Again, shapes must match: you can only add a 2×2 to another 2×2.

---

## Scalar multiplication

A **scalar** is just a single number (as opposed to a vector or matrix). The word "scalar" comes from "scale" — because multiplying a vector by a scalar *scales* it.

To multiply a vector by a scalar, multiply *every* number inside by that scalar:

$$3 \times \begin{bmatrix} 1 \\ 2 \\ 5 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \\ 15 \end{bmatrix}$$

```python
v = np.array([1, 2, 5])
print(3 * v)     # [ 3  6 15]
print(0.5 * v)   # [0.5 1.  2.5]
print(-1 * v)    # [-1 -2 -5]
```

### The picture: stretching and flipping

If a vector is an arrow, scaling by 3 makes the arrow **3 times longer** but keeps it pointing the exact same way. Scaling by 0.5 makes it **half as long**. Scaling by $-1$ **flips it around** to point in the opposite direction. Scaling by a negative number bigger than 1, like $-2$, both flips *and* stretches.

The key insight: **scalar multiplication never changes the direction** (except flipping it exactly backwards for negatives). It only changes the length.

This matters in AI because "how long" a vector is often means "how strong" a signal is, while "which direction" often means "what the signal *is* about." Scaling changes strength, not meaning.

### Scaling matrices

Exactly the same — multiply every entry:

```python
A = np.array([[1, 2], [3, 4]])
print(2 * A)
# [[2 4]
#  [6 8]]
```

---

## Matrix multiplication — the big one

Here we go. This is the single most important operation in this entire lesson, maybe in all of AI math. It's also the one that trips people up, because it does *not* work the way you'd guess. So we'll go slowly.

### It's NOT position-by-position

Your instinct after learning addition is: "multiply matching positions." **That is wrong for matrix multiplication.** (There is an operation that does that — it's called the *elementwise* or *Hadamard* product, and numpy writes it as `A * B`. But that is a different, less important operation. Matrix multiplication is something else.)

Matrix multiplication is built out of the **dot product** — taking a row from the first matrix, a column from the second, multiplying them pairwise, and adding up.

### Start with the simplest case: matrix times vector

Let's multiply a matrix by a vector. This is *the* operation a neural network layer does, so if you only master one thing, master this.

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \ ?$$

Here's the recipe. To get each number in the answer, take a **row** of the matrix, line it up against the **vector**, multiply pairwise, and sum.

**First answer number** — use the first row $[1, 2]$ against the vector $[5, 6]$:
$$1 \times 5 + 2 \times 6 = 5 + 12 = 17$$

**Second answer number** — use the second row $[3, 4]$ against the vector $[5, 6]$:
$$3 \times 5 + 4 \times 6 = 15 + 24 = 39$$

So:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

The matrix ate a 2D vector and spat out a new 2D vector. It **transformed** the input into an output. Hold that thought — it's the whole point.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
x = np.array([5, 6])

print(A @ x)     # [17 39]
```

That `@` symbol is Python's matrix-multiply operator. Use `@`, not `*`. (`*` would do the wrong, elementwise thing.) You can also write `np.dot(A, x)` or `A.dot(x)` — all three are the same.

### Now matrix times matrix

Same recipe, just repeat it for every column of the second matrix. To multiply $A \times B$:

> Each entry of the result = (a **row** of $A$) dotted with (a **column** of $B$).

The entry in row $i$, column $j$ of the answer comes from row $i$ of $A$ and column $j$ of $B$.

Let's do it:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \ ?$$

**Top-left** = row 1 of $A$ ⋅ column 1 of $B$ = $[1,2]\cdot[5,7] = 1(5)+2(7) = 5+14 = 19$

**Top-right** = row 1 of $A$ ⋅ column 2 of $B$ = $[1,2]\cdot[6,8] = 1(6)+2(8) = 6+16 = 22$

**Bottom-left** = row 2 of $A$ ⋅ column 1 of $B$ = $[3,4]\cdot[5,7] = 3(5)+4(7) = 15+28 = 43$

**Bottom-right** = row 2 of $A$ ⋅ column 2 of $B$ = $[3,4]\cdot[6,8] = 3(6)+4(8) = 18+32 = 50$

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)
# [[19 22]
#  [43 50]]
```

Check every number by hand at least once. Do it now. The "muscle memory" of *row dot column* is worth building.

### The shape rule (the thing everyone forgets)

Matrix multiplication has a strict rule about shapes, and 90% of the errors you'll ever hit are violations of it. Here it is:

> To multiply an $(m \times n)$ matrix by a $(p \times q)$ matrix, you **need $n = p$** — the inner numbers must match. The result is $(m \times q)$ — the outer numbers.

Line the shapes up side by side:

$$(m \times \underbrace{n) \quad (p}_{\text{must match}} \times q) \longrightarrow (m \times q)$$

Example: a $(2 \times 3)$ times a $(3 \times 4)$. Inner numbers are 3 and 3 — they match, good. Result is $(2 \times 4)$. 

A $(2 \times 3)$ times a $(2 \times 3)$? Inner numbers are 3 and 2 — they don't match. **Illegal.** numpy throws an error.

Why this rule? Because you're dotting a row of the first with a column of the second. A row of the first matrix has $n$ numbers in it. A column of the second has $p$ numbers. To dot them (multiply pairwise) they must be the same length, so $n = p$. That's the entire reason. It's not arbitrary.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])          # (2, 3)
B = np.array([[1, 0], [0, 1], [1, 1]])        # (3, 2)

print((A @ B).shape)    # (2, 2)  -> outer numbers

C = np.array([[1, 2], [3, 4]])                # (2, 2)
# A @ C  -> ERROR: (2,3) @ (2,2), inner numbers 3 and 2 don't match
```

### Order matters! $AB \neq BA$

Here's something that surprises everyone coming from regular number multiplication, where $3 \times 5 = 5 \times 3$. **For matrices, the order usually changes the answer.** $AB$ and $BA$ are generally *different matrices* — and sometimes one is even legal while the other is illegal (because of the shape rule).

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)
# [[19 22]
#  [43 50]]

print(B @ A)
# [[23 34]
#  [31 46]]
```

Totally different. So when you see $AB$ in AI, the order is meaningful. "Apply $B$, then apply $A$" is not the same as "apply $A$, then apply $B$" — just like putting on socks then shoes is not the same as shoes then socks.

---

## The big idea: a matrix transforms space

This is the intuition that makes matrices *click*, so let's give it real attention.

We saw that a matrix times a vector gives a new vector. So a matrix is like a **machine**: you feed a vector in, you get a vector out. Mathematicians call this a **transformation** or a **linear map**. A matrix *transforms* one vector into another.

But here's the beautiful part. Don't think of the matrix as transforming *one* vector. Think of it as transforming **all of space at once** — every possible arrow, all at the same time, in a consistent way.

### The identity matrix: the "do nothing" machine

The simplest matrix is the **identity matrix**, written $I$. It has 1s down the diagonal and 0s everywhere else:

$$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

Multiplying any vector by $I$ gives you *the same vector back*. It's the "do nothing" transformation — the matrix equivalent of multiplying a number by 1.

```python
I = np.eye(2)          # np.eye makes an identity matrix
x = np.array([7, 3])

print(I @ x)           # [7. 3.]  -> unchanged
```

### A stretching machine

Now change the diagonal:

$$S = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$$

Feed in $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$:

$$\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

This matrix **stretches** the world: everything gets twice as wide and three times as tall. Every arrow in the plane gets pulled outward. This is literally what happens when you resize an image non-uniformly.

```python
S = np.array([[2, 0], [0, 3]])
print(S @ np.array([1, 1]))   # [2 3]
print(S @ np.array([4, 5]))   # [8 15]  -> width x2, height x3
```

### A rotation machine

Certain matrices **rotate** everything around the origin. Here's one that rotates by 90 degrees counter-clockwise:

$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

Feed in the arrow $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (pointing east):

$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

The east-pointing arrow became a north-pointing arrow. It rotated 90°. And *every* arrow in the plane rotates 90° when you hit it with $R$.

```python
R = np.array([[0, -1], [1, 0]])
print(R @ np.array([1, 0]))    # [0 1]  east -> north
print(R @ np.array([0, 1]))    # [-1 0] north -> west
```

Stretch, rotate, flip, squish, shear — these are all just different matrices. **A matrix is a way of moving space around.** That single idea — "matrix = transformation of space" — is worth more than any formula. Hold onto it.

### Chaining transformations = multiplying matrices

Here's why matrix multiplication was *defined* the weird way it was. If matrix $A$ is one transformation and matrix $B$ is another, then doing $B$ first and then $A$ is the *same* as applying the single combined matrix $A B$.

$$A(Bx) = (AB)x$$

That's the whole reason matrix multiplication uses "row dot column" — it's precisely the rule that makes "apply one transformation after another" work out. So multiplying matrices = **stacking transformations**. And that's exactly what a deep neural network is: a stack of transformations, one per layer.

---

## Bringing it home: a neural network layer

Let's connect everything to real AI. A single layer of a neural network does this:

$$\vec{y} = W\vec{x} + \vec{b}$$

Let's decode every symbol:

- $\vec{x}$ is the **input** vector — your data (an image, a word embedding, a row of features).
- $W$ is the **weight matrix** — the numbers the network *learned* during training.
- $\vec{b}$ is the **bias** vector — a little offset added on.
- $\vec{y}$ is the **output** vector — the layer's result, passed to the next layer.

That $W\vec{x}$ is a matrix-vector multiplication — *exactly* the operation you just learned. The matrix transforms the input into a new space, the bias shifts it, and that's one layer.

```python
import numpy as np

# A tiny neural network layer: 3 inputs -> 2 outputs
np.random.seed(0)
W = np.random.randn(2, 3)   # weight matrix: 2 rows (outputs), 3 cols (inputs)
b = np.array([0.5, -0.5])   # bias: one per output
x = np.array([1.0, 2.0, 3.0])  # input: 3 features

y = W @ x + b
print("Input :", x)
print("Output:", y)          # a new 2-dimensional vector
```

Run it. You just executed a neural network layer by hand. A real network like GPT does this same thing, just with matrices that have *billions* of numbers, stacked dozens of layers deep. But the operation in each layer? It's the matrix-vector multiply you learned today. Nothing more exotic.

### Why do it with matrices at all?

Two reasons, both important.

1. **It's compact.** One matrix multiply computes *all* the outputs at once. Without matrices you'd write a tangle of nested loops.
2. **It's blazingly fast.** GPUs are hardware built specifically to multiply matrices ridiculously fast. The entire deep-learning revolution rode on the fact that "the core operation is matrix multiplication, and we have chips that do that a trillion times a second." If neural nets needed some other operation, the hardware might not exist. This is not a small point — it's *why* modern AI happened when it did.

---

## Elementwise multiply vs matrix multiply (don't mix them up)

One more warning, because it bites everyone. numpy has two multiply operations and they're totally different:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)    # ELEMENTWISE: multiply matching positions
# [[ 5 12]
#  [21 32]]

print(A @ B)    # MATRIX MULTIPLY: row dot column
# [[19 22]
#  [43 50]]
```

- `*` = elementwise (Hadamard) product. Multiply position by position. Shapes must match exactly.
- `@` = matrix multiplication. Row dot column. Inner shapes must match.

When people say "matrix multiplication" in AI, they almost always mean `@`. But `*` shows up too (for example, in attention masks and gating). Read carefully and know which one you want.

---

## Summary

- A **vector** is a list of numbers. Picture it as an arrow with a direction and a length, or as a point in space. In AI, data becomes vectors (word embeddings, flattened images, feature rows).
- A **matrix** is a grid of numbers with rows and columns. Shape is (rows × columns). A data table is a matrix; a neural network's weights are matrices.
- **Add** vectors/matrices position by position; shapes must match. Picture it as arrows tip-to-tail.
- **Scalar multiply** scales every entry — it stretches or flips an arrow but never changes its direction.
- **Matrix multiplication** is *row dot column*, not position-by-position. Use `@` in numpy. The inner shapes must match; the result takes the outer shapes. Order matters: $AB \neq BA$.
- The big intuition: **a matrix transforms space** — it stretches, rotates, flips, and squishes every vector at once. Multiplying matrices chains transformations together.
- Every neural network layer is $\vec{y} = W\vec{x} + \vec{b}$ — a matrix-vector multiply plus a bias. This is why matrices are the foundation of all of deep learning, and why GPUs (matrix-multiply machines) power modern AI.
