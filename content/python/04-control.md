---
title: Control Flow
track: python
order: 4
estimatedTime: 40
difficulty: beginner
---

# Control Flow

So far your programs have run straight from top to bottom, one line after another, no choices, no repeats. That's fine for a calculator, but real programs need to make **decisions** ("if the score is above 90, print 'excellent'") and **repeat** actions ("do this for all 10,000 images"). Controlling which lines run, and how many times, is called **control flow**. It's where programming starts to feel powerful.

This is one of the most important lessons in the whole track. Every training loop that teaches a neural network, every pass over a dataset, every decision a model makes — it's all built from the tools you're about to learn: `if`, `for`, and `while`. Take your time here.

---

## Making decisions with `if`

The `if` statement runs a block of code **only when** a condition is true. Here's the simplest form:

```python
temperature = 35

if temperature > 30:
    print("It's hot outside!")
    print("Drink water.")
```

Run it. Because `35 > 30` is `True`, both indented lines run. Now change `temperature` to `20` and run again — nothing prints, because `20 > 30` is `False`, so Python skips the whole block.

Let's look carefully at the structure, because Python is strict about it:

- The line starts with `if`, then a **condition**, then a **colon** `:`.
- The colon says "a block of code follows."
- The lines that belong to the `if` are **indented** (pushed to the right, usually by 4 spaces).

### Indentation is not optional in Python

In most languages, indentation is just for looks. In Python, **indentation is how the language knows which lines belong together**. The indented lines under an `if` are the block that runs when the condition is true. When the indentation stops, the block is over.

```python
score = 95

if score > 90:
    print("This line is inside the if")
    print("So is this one")
print("This line is OUTSIDE — it always runs")
```

The first two prints only happen when `score > 90`. The third print isn't indented, so it's not part of the `if` — it runs no matter what.

> If your indentation is inconsistent — mixing tabs and spaces, or using different amounts — Python raises an `IndentationError`. Pick 4 spaces and stick with it. Most code editors do this automatically when you press Tab. This is the single most common beginner error in Python, so when you see `IndentationError`, check that your block lines up neatly.

---

## `else`: what to do otherwise

Often you want one thing to happen if the condition is true and a *different* thing if it's false. That's what `else` is for. The `else` block runs when the `if` condition is false:

```python
age = 15

if age >= 18:
    print("You can vote.")
else:
    print("Too young to vote.")
```

Exactly one of the two blocks runs, never both. If `age` is 18 or more, you get the first message; otherwise the second. There's no condition after `else` — it's the catch-all for "everything the `if` didn't catch."

---

## `elif`: checking several conditions

What if there are more than two possibilities? Use `elif` (short for "else if") to add extra conditions between the `if` and the `else`. Python checks them in order and runs the **first** one that's true, then skips the rest:

```python
score = 82

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")
```

With `score = 82`: Python checks `82 >= 90` (false, skip), then `82 >= 80` (true! print "Grade: B" and stop). It never even looks at the `C` or `else` cases. That "stop at the first match" behavior is important — order your conditions carefully.

You can have as many `elif` branches as you want. The `else` at the end is optional; without it, if nothing matches, nothing happens.

> A subtle trap: the order matters because Python stops at the first true condition. If you wrote `if score >= 70` first, then `82` would match that and print "C" — wrong! Always order your conditions from most specific/highest to least, or think carefully about which should win.

---

## Comparison operators

The conditions in `if` statements are built from **comparison operators**. Each one compares two values and produces a boolean (`True` or `False`). Here they all are:

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `==` | equal to | `5 == 5` | `True` |
| `!=` | not equal to | `5 != 3` | `True` |
| `<` | less than | `3 < 5` | `True` |
| `>` | greater than | `3 > 5` | `False` |
| `<=` | less than or equal | `5 <= 5` | `True` |
| `>=` | greater than or equal | `5 >= 6` | `False` |

```python
print(5 == 5)    # True
print(5 != 3)    # True
print(3 < 5)     # True
print(10 >= 10)  # True
print("cat" == "cat")   # True  (works on strings too)
print("cat" == "Cat")   # False (case matters!)
```

> The single biggest beginner mistake: using `=` when you mean `==`. Remember, `=` **assigns** a value (`x = 5` puts 5 into x), while `==` **compares** (`x == 5` asks "is x equal to 5?"). In an `if` condition you almost always want `==`. Writing `if x = 5:` is a syntax error in Python, which is actually helpful — it stops you before the bug does any damage.

---

## Logical operators: `and`, `or`, `not`

Sometimes a decision depends on more than one condition. Logical operators let you combine conditions:

- **`and`** is true only when **both** sides are true.
- **`or`** is true when **at least one** side is true.
- **`not`** flips a boolean: `not True` is `False`.

```python
age = 20
has_ticket = True

# Both must be true
if age >= 18 and has_ticket:
    print("Welcome to the show!")

# At least one must be true
temperature = 5
if temperature < 0 or temperature > 40:
    print("Extreme weather!")

# Flip a condition
is_raining = False
if not is_raining:
    print("No umbrella needed.")
```

Here's a truth table to make `and` and `or` crystal clear:

| A | B | A `and` B | A `or` B |
|---|---|---|---|
| True | True | True | True |
| True | False | False | True |
| False | True | False | True |
| False | False | False | False |

You can chain several together, and use parentheses to group them clearly:

```python
score = 85
attendance = 0.95

if score >= 80 and attendance >= 0.9:
    print("Passed with good attendance")

age = 25
if (age >= 13 and age <= 19):
    print("Teenager")
else:
    print("Not a teenager")
```

Combining conditions this way is exactly how models make layered decisions, and how you'll filter data later ("keep rows where age > 18 **and** country == 'UK'").

---

## Repeating with `for` loops

Now for the other superpower: doing something many times without writing it out many times. The `for` loop repeats a block of code once for each item in a sequence.

### Looping with `range()`

The `range()` function generates a sequence of numbers, and `for` walks through them one at a time. `range(5)` gives the numbers 0, 1, 2, 3, 4 (five numbers, starting at 0):

```python
for i in range(5):
    print(i)
# prints 0, 1, 2, 3, 4
```

Each time around the loop, the variable `i` takes the next value. `i` is just a name — you could call it anything, but `i` (for "index") is traditional.

`range()` has flexible forms:

```python
# range(stop): 0 up to stop-1
for i in range(3):
    print(i)          # 0, 1, 2

# range(start, stop): start up to stop-1
for i in range(2, 6):
    print(i)          # 2, 3, 4, 5

# range(start, stop, step): count by step
for i in range(0, 10, 2):
    print(i)          # 0, 2, 4, 6, 8
```

Notice that, just like slicing, `range` stops *before* the stop value. `range(2, 6)` gives 2, 3, 4, 5 — not 6. This exclusive-stop rule is everywhere in Python; you'll internalize it soon.

A classic use is repeating something a fixed number of times, even if you don't use the loop variable:

```python
for i in range(3):
    print("Hello!")
# prints Hello! three times
```

### Looping over lists and strings

`for` doesn't just work with `range()` — it works with any sequence. You can loop directly over the items of a list, or the characters of a string:

```python
# Loop over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# Loop over a string, character by character
for letter in "cat":
    print(letter)     # c, a, t
```

This is cleaner than using `range(len(...))` and indexing — you get each item directly. This "loop over the items" style is the Pythonic way, and you'll use it constantly to process datasets: "for each image in the training set, do this."

### Accumulating a result

A very common loop pattern is building up a total or a count as you go. You start with a variable, then update it each time around:

```python
numbers = [4, 8, 15, 16, 23, 42]

total = 0
for n in numbers:
    total = total + n   # or: total += n

print(f"Sum: {total}")   # Sum: 108
```

Trace it: `total` starts at 0, then becomes 4, then 12, then 27, and so on until it holds the full sum. This "start at zero, add each item" pattern is how you compute sums, averages, and — in AI — how you accumulate the total loss across a batch of training examples.

---

## `while` loops

A `for` loop runs a known number of times. A `while` loop runs **as long as a condition stays true** — you don't necessarily know in advance how many times that'll be. It checks the condition, runs the block, checks again, and repeats until the condition becomes false:

```python
count = 0

while count < 5:
    print(count)
    count += 1      # this is essential! see the warning below
# prints 0, 1, 2, 3, 4
```

Read it as: "while count is less than 5, print it and add 1." After the block runs five times, `count` reaches 5, the condition `5 < 5` is false, and the loop stops.

> **The infinite loop trap.** A `while` loop only stops when its condition becomes false. If you forget to change the variable in the condition, it never becomes false and the loop runs *forever*, freezing your program. In the example above, if you delete `count += 1`, then `count` stays 0, `0 < 5` is always true, and it never ends. Always make sure something inside the loop moves you toward stopping.

`while` loops shine when you don't know the number of repetitions ahead of time. In AI, you might train "while the model is still improving" — you don't know how many rounds that'll take, so a `while` loop (checking whether the error is still going down) fits perfectly.

---

## `break` and `continue`

Two keywords give you finer control inside loops.

**`break` exits the loop immediately**, even if there are more items left:

```python
for n in range(100):
    if n == 5:
        break        # stop the loop right here
    print(n)         # prints 0, 1, 2, 3, 4 — then breaks at 5
```

`break` is perfect for searching: loop through data until you find what you want, then stop — no point continuing.

```python
names = ["Ada", "Alan", "Grace", "Katherine"]
for name in names:
    if name == "Grace":
        print("Found Grace!")
        break
```

**`continue` skips the rest of the current loop pass** and jumps straight to the next one:

```python
for n in range(10):
    if n % 2 == 0:
        continue     # skip even numbers
    print(n)         # prints only 1, 3, 5, 7, 9
```

When `n` is even, `continue` jumps back to the top of the loop before reaching the `print`. So only odd numbers get printed. `continue` is handy for skipping over data you don't care about — like skipping corrupted images while processing a dataset.

---

## Nested loops

You can put a loop inside another loop. The inner loop runs completely for **each** pass of the outer loop. This is how you work with grids, tables, and 2D data — including images and matrices, which are central to AI.

```python
for row in range(3):
    for col in range(3):
        print(f"({row}, {col})", end="  ")
    print()   # newline after each row
```

This prints a 3×3 grid of coordinate pairs. The outer loop picks a row; for each row, the inner loop runs through all three columns. That means the inner block runs 3 × 3 = 9 times total.

A concrete example — a multiplication table:

```python
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i*j}")
    print("---")
```

Nested loops are exactly how you'd walk over every pixel of an image (loop over rows, then columns) or every cell of a matrix. When you get to NumPy, you'll learn faster ways to do this — but understanding the nested-loop version first makes the fast version make sense.

> Watch the cost: nested loops multiply. Two loops of 1,000 items each means 1,000,000 iterations. Three levels deep means a billion. This is why AI relies on NumPy and GPUs, which do these repeated operations far faster than plain Python loops. Keep an eye on how deep your loops nest.

---

## List comprehensions (a first look)

Here's a compact, very Pythonic pattern for building a list with a loop. Suppose you want a list of the squares of 0 through 4. The long way:

```python
squares = []
for n in range(5):
    squares.append(n * n)
print(squares)   # [0, 1, 4, 9, 16]
```

The short way, called a **list comprehension**, does the same thing in one line:

```python
squares = [n * n for n in range(5)]
print(squares)   # [0, 1, 4, 9, 16]
```

Read it left to right: "make a list of `n * n` for each `n` in `range(5)`." It's the loop turned inside out, with the thing you're collecting written first.

You can add a condition to filter, too:

```python
evens = [n for n in range(10) if n % 2 == 0]
print(evens)   # [0, 2, 4, 6, 8]
```

That reads: "collect `n` for each `n` in range(10), but only if `n` is even." List comprehensions are everywhere in real Python code because they're short and clear. We'll go deeper on them in the lists lesson — for now, just recognize the pattern and know it's equivalent to a `for` loop that appends.

---

## Why control flow matters in machine learning

Control flow isn't just a beginner topic you leave behind. It's the skeleton of every machine learning program. Here's how the pieces you just learned show up in real AI code:

**The training loop is a `for` loop.** Teaching a neural network means showing it the data many times. Each full pass over the dataset is called an **epoch**, and training runs for a fixed number of epochs:

```python
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    # (in real code: feed data through the model, measure error, adjust)
    loss = 1.0 / (epoch + 1)   # pretend the error shrinks each epoch
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} — loss: {loss:.3f}")
```

That's a real training loop's shape. Every model you'll ever train sits inside a `for epoch in range(...)` loop.

**Nested loops process batches within epochs.** Datasets are split into small groups called batches. So real training is a loop within a loop: for each epoch, for each batch, update the model. You just learned nested loops — this is where they pay off.

**`if` statements make decisions.** "If the validation error stopped improving, stop training" (called early stopping). "If this prediction is above 0.5, classify it as positive." Models are full of these threshold decisions.

**`while` loops handle unknown-length processes.** "Keep training while the model is still getting better." "Keep generating text until you hit the end-of-sentence token." When you don't know how many steps you need, `while` is the tool.

**`break` and `continue` handle special cases.** Break out of training early if something goes wrong; skip (continue past) a corrupted data sample. These come up constantly in real projects.

When you look at professional AI code, you'll recognize this structure immediately: loops over epochs and batches, `if` checks for thresholds and stopping conditions, and the occasional `break`. The control flow you learned today is the frame that everything else hangs on.

---

## Common mistakes with control flow

**Mistake 1: `=` instead of `==` in a condition.** `if x = 5:` is an error; you want `if x == 5:`. One equals assigns, two equals compares.

**Mistake 2: Forgetting the colon.** Every `if`, `elif`, `else`, `for`, and `while` line ends with a colon `:`. Miss it and you get a `SyntaxError`.

**Mistake 3: Wrong indentation.** The block under a control statement must be indented consistently (4 spaces). Inconsistent indentation causes `IndentationError`.

**Mistake 4: Infinite `while` loop.** Forgetting to update the variable in the condition makes the loop run forever. Always change something that moves toward the stop condition.

**Mistake 5: Ordering `elif` conditions wrong.** Python stops at the first true branch, so put more specific/tighter conditions before looser ones.

---

## Putting it all together

Let's build a small program that combines decisions and loops. It goes through a list of test scores, grades each one, and counts how many passed.

```python
scores = [95, 62, 78, 45, 88, 100, 53]
PASS_MARK = 60

pass_count = 0

for score in scores:
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "F"

    # Count passes
    if score >= PASS_MARK:
        pass_count += 1

    print(f"Score {score}: grade {grade}")

print("---")
print(f"{pass_count} out of {len(scores)} students passed")

# Compute the average with an accumulating loop
total = 0
for score in scores:
    total += score
average = total / len(scores)
print(f"Class average: {average:.1f}")
```

This one program uses a `for` loop, `if`/`elif`/`else`, comparison operators, a constant, an accumulating counter, `len()`, and an f-string with formatting. Trace it by hand before running, then check your prediction. This is the shape of a real data-processing program.

---

## Summary

- **Control flow** decides which lines run and how many times.
- `if` runs a block when a condition is true; `else` handles the false case; `elif` adds more conditions. Python runs the **first** true branch and skips the rest.
- **Indentation** (4 spaces) defines which lines belong to a block. It's required, not decoration.
- **Comparison operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`. Use `==` to compare, `=` to assign — don't mix them up.
- **Logical operators**: `and` (both true), `or` (at least one true), `not` (flip).
- `for` loops repeat over a sequence: `range(n)`, lists, strings. `range` stops *before* its stop value.
- `while` loops repeat as long as a condition is true. Always change something inside, or you get an infinite loop.
- `break` exits a loop early; `continue` skips to the next pass.
- **Nested loops** handle grids and 2D data; watch out — the work multiplies.
- **List comprehensions** (`[n*n for n in range(5)]`) are a compact way to build lists with a loop.
- In ML, the **training loop** is a `for` loop over epochs, often with nested batch loops, `if` checks for thresholds, and `while` loops for unknown-length processes. Control flow is the skeleton of every model.
