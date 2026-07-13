---
title: Dot Product and Norms
track: math
order: 2
estimatedTime: 40
difficulty: intermediate
---

# Dot Product and Norms

In the last lesson you met the dot product as the little "row dot column" step hiding inside matrix multiplication. Now we're going to pull it out and stare at it directly, because the dot product is one of the most important single operations in all of AI.

Why such a big deal? Because the dot product answers a question that machines ask constantly: **how similar are these two things?** How similar is this image to that piece of text? How similar is this search query to that document? How aligned are these two ideas? Under the hood, the answer is almost always a dot product.

We'll also learn about **norms** — a fancy word for "the length of a vector" — because you can't talk about similarity without talking about size. Let's go.

---

## The dot product: definition

The dot product takes **two vectors** and gives back **one number** (a scalar). Here's the recipe, and it's simple: multiply matching positions, then add everything up.

$$\vec{a} \cdot \vec{b} = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

That funny $\sum$ symbol (Greek capital sigma) just means "add up all of these." So $\sum_i a_i b_i$ reads as "for each position $i$, multiply $a_i$ times $b_i$, and add all those products together."

Let's do one by hand. Take $\vec{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\vec{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$:

$$\vec{a} \cdot \vec{b} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$$

Two vectors went in, one number (32) came out. That's the dot product.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Three ways to compute the dot product, all identical:
print(np.dot(a, b))    # 32
print(a @ b)           # 32
print(np.sum(a * b))   # 32  (elementwise multiply, then sum — the definition!)
```

Notice the third one: `a * b` multiplies position-by-position giving `[4, 10, 18]`, and `np.sum` adds them to get 32. That *is* the definition, written out. The `@` and `np.dot` are just shortcuts for the same thing.

### The rule: same length

Just like addition, the dot product needs both vectors to be the **same length**. You can't dot a 3D vector with a 5D one — there'd be positions left over with nothing to pair. numpy will error out, and that's a feature, not a bug.

---

## What does the dot product *mean*?

The formula is easy. But *why* would multiply-and-add tell you anything useful? Here's the geometric truth, and it's gorgeous.

There's a second formula for the exact same dot product:

$$\vec{a} \cdot \vec{b} = |\vec{a}|\,|\vec{b}|\cos\theta$$

where $|\vec{a}|$ and $|\vec{b}|$ are the **lengths** of the two vectors, and $\theta$ (theta) is the **angle between them**. Both formulas give the same number — one uses coordinates, one uses lengths and angle.

This second formula is where the intuition lives. Let's break down what it tells us about *direction*:

- If the two vectors point in the **same direction**, $\theta = 0°$, and $\cos 0° = 1$. The dot product is big and positive — as large as it can be.
- If they're **perpendicular** (at a right angle), $\theta = 90°$, and $\cos 90° = 0$. The dot product is **exactly zero**.
- If they point in **opposite directions**, $\theta = 180°$, and $\cos 180° = -1$. The dot product is big and *negative*.

So the sign and size of the dot product tell you about *alignment*:

> **The dot product measures how much two vectors point the same way.** Positive = aligned. Zero = perpendicular. Negative = opposed.

This is the single most important intuition in the lesson. When a machine computes $\vec{a} \cdot \vec{b}$ and gets a big positive number, it's saying "these two things are aligned — they agree — they're similar." When it gets a negative number, "these disagree." When it gets zero, "these are unrelated."

### A quick sanity check

Let's verify perpendicular vectors give zero. Take $\vec{a} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (pointing east) and $\vec{b} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (pointing north). They're at a perfect right angle.

$$\vec{a} \cdot \vec{b} = (1)(0) + (0)(1) = 0$$

Zero, exactly as promised.

```python
east = np.array([1, 0])
north = np.array([0, 1])
print(np.dot(east, north))   # 0  -> perpendicular
```

---

## Norms: the length of a vector

Before we can build a proper "similarity" measure, we need to nail down **length**. The length of a vector is called its **norm**, written with double bars: $\|\vec{v}\|$. (Single bars $|\cdot|$ are also common for vectors; they mean the same thing here.)

### The L2 norm (ordinary length)

The most common norm is the **L2 norm**, which is just plain old distance — the Pythagorean theorem stretched to any number of dimensions.

$$\|\vec{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_i v_i^2}$$

Square every component, add them up, take the square root. In 2D, this is exactly the hypotenuse of a right triangle — the straight-line distance from the origin to the point.

Take $\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$:

$$\|\vec{v}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

That's the famous 3-4-5 triangle. The arrow "3 right, 4 up" has length exactly 5.

```python
import numpy as np

v = np.array([3, 4])
print(np.linalg.norm(v))          # 5.0
print(np.sqrt(np.sum(v**2)))      # 5.0  -> the definition spelled out
```

Notice the connection to the dot product: $\|\vec{v}\|_2^2 = \vec{v} \cdot \vec{v}$. A vector dotted *with itself* gives its length squared. (Makes sense: $\sum v_i v_i = \sum v_i^2$.) That's a handy fact you'll see everywhere.

```python
print(np.dot(v, v))               # 25  = 5 squared
```

### The L1 norm (taxicab distance)

The **L1 norm** adds up the *absolute values* of the components — no squaring, no square root:

$$\|\vec{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n| = \sum_i |v_i|$$

For $\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$: $\|\vec{v}\|_1 = |3| + |4| = 7$.

Why "taxicab"? Because it's how far a taxi drives in a city grid. To get from the corner of a building to a point 3 blocks east and 4 blocks north, a taxi can't cut diagonally through buildings — it drives 3 blocks, turns, drives 4 blocks: total 7 blocks. The L2 norm (5) is the crow-flies distance; the L1 norm (7) is the streets-you-actually-drive distance.

```python
v = np.array([3, 4])
print(np.linalg.norm(v, ord=1))   # 7.0
print(np.sum(np.abs(v)))          # 7  -> the definition
```

### When to use each

Both measure "size," but they behave differently, and that difference matters in ML:

| | L1 norm | L2 norm |
|---|---|---|
| Formula | sum of absolute values | square root of sum of squares |
| Nickname | taxicab / Manhattan | Euclidean / straight-line |
| Big values | treats them linearly | punishes them *hard* (squared) |
| In ML | Lasso, encourages sparsity (many zeros) | Ridge, encourages small smooth weights |

The key behavioral difference: **L2 squares things, so it punishes large components much more harshly than small ones.** A component of 10 contributes 100 to L2 but only 10 to L1. This is why L2 tends to spread "blame" out evenly and keep everything small, while L1 is happy to zero some things out entirely and keep others large. You'll meet both again in the regularization lesson — L1 (Lasso) and L2 (Ridge) are named directly after these norms.

---

## Cosine similarity: the star of the show

Now we combine everything. Here's a problem: the raw dot product depends on how *long* the vectors are, not just their direction. Two documents about the same topic might have very different dot products just because one is longer. We usually care about *direction* (the meaning/topic), not magnitude (the length). How do we measure just the direction agreement?

Rearrange the geometric formula $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$ to solve for $\cos\theta$:

$$\cos\theta = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|\,\|\vec{b}\|}$$

This is **cosine similarity**. You take the dot product and *divide out the lengths*. What's left is purely about the angle — the direction agreement — with the magnitudes cancelled away.

Because it's a cosine, the answer always lands between $-1$ and $+1$:

- **+1** → vectors point the *exact same way* (angle 0°). Maximum similarity.
- **0** → vectors are perpendicular (angle 90°). Unrelated.
- **−1** → vectors point *exactly opposite* (angle 180°). Maximum dissimilarity.

This bounded, length-independent measure is *the* standard way AI systems compare things.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])     # same direction as a, just twice as long
c = np.array([-1, -2, -3])  # exact opposite direction

print(cosine_similarity(a, b))   # 1.0   -> identical direction
print(cosine_similarity(a, c))   # -1.0  -> opposite direction
print(cosine_similarity(a, np.array([3, -1.5, 0])))  # ~0 -> perpendicular
```

Look at `a` and `b`: `b` is just `a` scaled by 2. Same direction, different length. Their dot product is different from `a·a`, but their cosine similarity is exactly **1.0** — because cosine ignores length and only cares about direction. That's precisely the property we wanted.

### Why cosine similarity matters for AI: CLIP

Here's the payoff, and it's a big one. **CLIP** is a famous model from OpenAI that connects images and text. You've used its descendants every time an app lets you search photos by typing words ("show me pictures of a dog on a beach").

How does CLIP work? It has two encoders:
- an **image encoder** that turns a picture into a vector (say, 512 numbers),
- a **text encoder** that turns a sentence into a vector in the *same space* (also 512 numbers).

To check if the image matches the text, CLIP computes the **cosine similarity** between the two vectors. High cosine similarity → the image and the caption are about the same thing. Low → they don't match.

That's the entire idea. "A photo of a cat" and an actual photo of a cat produce vectors that point in nearly the same direction, so their cosine similarity is high. The image and "a photo of a truck" point in different directions, so low similarity. Search, captioning, zero-shot classification — all built on cosine similarity between vectors.

The same trick powers:
- **Semantic search** — find documents whose embedding vectors are most cosine-similar to your query's vector.
- **Recommendation** — find products/songs whose vectors point the same way as ones you liked.
- **RAG** (retrieval-augmented generation) — the "retrieval" step in modern chatbots finds relevant text chunks by cosine similarity to your question.

Learn cosine similarity well and you understand the retrieval half of modern AI.

---

## Orthogonality: when the dot product is zero

Two vectors are **orthogonal** when their dot product is zero. "Orthogonal" is just the fancy math word for "perpendicular" / "at a right angle" — but it keeps working in dimensions you can't picture, where "right angle" is hard to imagine but "dot product = 0" is easy to check.

$$\vec{a} \cdot \vec{b} = 0 \quad \Longleftrightarrow \quad \vec{a} \text{ and } \vec{b} \text{ are orthogonal}$$

Orthogonal vectors have cosine similarity 0 — they're the "completely unrelated" case. Why does AI care?

- **Independent information.** If two feature vectors are orthogonal, they carry non-overlapping information — knowing one tells you nothing about the other. Many algorithms (like PCA, which you'll meet later) work by finding orthogonal directions in data, each capturing a fresh, non-redundant piece of the story.
- **Coordinate axes.** The standard axes — east $\begin{bmatrix}1\\0\end{bmatrix}$ and north $\begin{bmatrix}0\\1\end{bmatrix}$ — are orthogonal. That's what makes them good axes: they don't overlap.

```python
import numpy as np

a = np.array([2, 1])
b = np.array([-1, 2])
print(np.dot(a, b))   # 0  -> orthogonal! (2*-1 + 1*2 = 0)

# A quick reusable check
def are_orthogonal(u, v, tol=1e-9):
    return abs(np.dot(u, v)) < tol

print(are_orthogonal(a, b))   # True
```

We use a small tolerance `tol` instead of checking `== 0` exactly, because with decimal (floating-point) numbers a dot product that *should* be zero might come out as `0.0000000001`. That's a general lesson: never test floating-point numbers for exact equality — check if they're *close enough*.

---

## A worked mini-example: ranking by similarity

Let's tie it all together with a tiny "search engine." We have a query vector and a few document vectors, and we want to rank the documents by how similar they are to the query. This is exactly what real retrieval systems do, just at massive scale.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Imagine these are tiny "embeddings" of text
query = np.array([1.0, 0.2, 0.1])

docs = {
    "doc_A": np.array([0.9, 0.3, 0.0]),   # very aligned with query
    "doc_B": np.array([0.0, 1.0, 0.0]),   # points a different way
    "doc_C": np.array([2.0, 0.4, 0.2]),   # same direction as query, longer
}

scores = {name: cosine_similarity(query, vec) for name, vec in docs.items()}

# Sort documents from most to least similar
for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{name}: {score:.3f}")
```

Run it. You'll see `doc_C` and `doc_A` score highest (they point the same way as the query), while `doc_B` scores low (different direction). Notice `doc_C` scores about the same as a perfectly-aligned vector *despite* being twice as long — because cosine similarity ignores length. Swap in real 768-dimensional embeddings and you've got the core of a semantic search engine.

---

## Summary

- The **dot product** $\vec{a} \cdot \vec{b} = \sum_i a_i b_i$ multiplies matching positions and adds them up. Two vectors in, one number out.
- Geometrically, $\vec{a} \cdot \vec{b} = \|\vec{a}\|\|\vec{b}\|\cos\theta$. The dot product measures **alignment**: positive = same direction, zero = perpendicular, negative = opposite.
- A **norm** is the length of a vector. The **L2 norm** $\sqrt{\sum v_i^2}$ is ordinary straight-line distance; the **L1 norm** $\sum|v_i|$ is taxicab distance. L2 punishes large values harshly (squares them); L1 treats them linearly and encourages sparsity.
- **Cosine similarity** $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\|\vec{b}\|}$ divides out the lengths to measure pure direction agreement. It lands in $[-1, +1]$ and is *the* standard similarity measure in AI.
- **CLIP** compares images and text by cosine similarity between their embedding vectors — the same trick powers semantic search, recommendation, and RAG.
- Two vectors are **orthogonal** when their dot product is zero — the higher-dimensional version of "perpendicular," meaning they carry independent, non-overlapping information.
