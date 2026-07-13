---
title: What is Python?
track: python
order: 1
estimatedTime: 20
difficulty: beginner
---

# What is Python?

Python is a programming language. A programming language is how you talk to a computer and tell it what to do.

You type instructions. The computer reads them and runs them. That's it.

Python is one of the most popular languages in the world right now — especially for AI and data science. When you see people training neural networks, processing data, or competing at IOAI, they're almost always using Python.

---

## Why Python?

There are hundreds of programming languages. Python is popular for a few simple reasons:

- **It reads like English.** `print("hello")` does exactly what it sounds like.
- **It's fast to write.** You can do in 5 lines what would take 50 lines in another language.
- **It has amazing libraries.** NumPy, PyTorch, scikit-learn — all the tools AI researchers use are in Python.

> **Goal of this course:** Take you from your very first Python line all the way to competing at the International Olympiad in Artificial Intelligence (IOAI).

---

## Your first Python program

The most famous first program is "Hello, World!". It just prints a message.

```python
print("Hello, World!")
```

Click **Run** to see it work. The output appears below the code.

---

## How Python runs code

Python reads your code line by line, from top to bottom. Each line is an instruction.

```python
print("First line")
print("Second line")
print("Third line")
```

The order matters. Python always does line 1 before line 2 before line 3.

---

## Comments

A **comment** starts with `#`. Python ignores everything after `#` on that line.

```python
# This is a comment — Python skips it
print("This runs")  # This part also skips
```

Comments are for you, not the computer. Use them to leave notes explaining your code.

---

## Python as a calculator

Python can do math directly.

```python
print(2 + 3)
print(10 - 4)
print(6 * 7)
print(15 / 4)
print(15 // 4)   # integer division (no decimal)
print(15 % 4)    # remainder (modulo)
print(2 ** 10)   # 2 to the power of 10
```

Run this and check each result. Notice that `15 / 4` gives `3.75` but `15 // 4` gives `3` — it drops the decimal.

---

## Where does Python run?

At IOAI and in this course, you'll use:

- **Colab / Jupyter** — notebooks where you run code cell by cell (like the notebooks your teacher shared)
- **This app** — has a built-in Python engine for exercises
- **Your computer** — for bigger projects

All three use the same Python. The only difference is how you run it.

---

## The IOAI journey ahead

Here's what you'll learn in this course, in order:

| Stage | What you'll learn |
|---|---|
| Python Basics | Variables, loops, functions, classes, numpy, pandas |
| Math for AI | Linear algebra, calculus, probability |
| Classical ML | Regression, trees, SVMs, evaluation |
| Neural Networks | Backprop, optimizers, PyTorch |
| Computer Vision | CNNs, YOLO, ViT, CLIP |
| NLP | Tokenization, BERT, Transformers |
| IOAI Competition | Strategy, past problems, competition tips |

Each track builds on the last. Don't skip — the later tracks will be confusing without the earlier ones.

---

## Summary

- Python is a programming language you'll use throughout this course and at IOAI.
- `print()` shows output.
- Code runs top to bottom.
- `#` starts a comment.
- Python can do math with `+`, `-`, `*`, `/`, `//`, `%`, `**`.
