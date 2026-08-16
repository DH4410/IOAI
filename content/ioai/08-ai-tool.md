---
title: Using the Contest AI Wisely
track: ioai
order: 8
estimatedTime: 35
difficulty: intermediate
---

# Using the Contest AI Wisely

In IOAI 2025, you get something most coding competitions don't give you: a built-in AI assistant inside the contest platform. It's a compact language model that runs directly in the Bohrium environment. You cannot use ChatGPT, Claude, or any external AI. But you CAN use the built-in one.

This lesson teaches you how to use it well. The difference between a student who gets 60 points and one who gets 90 points often comes down to this: the 90-point student knows exactly what to ask and when.

---

## 1. What the Contest AI Can and Cannot Do

The contest AI is a **small, fast model**. Think of it as a very smart assistant that has read a lot of Python and ML documentation, but runs on limited hardware with capped input and output length.

**It is good at:**
- Explaining what an error message means
- Clarifying what a function does
- Suggesting why a piece of code might have a bug
- Explaining a math concept in plain language
- Describing what a metric measures

**It is NOT good at:**
- Writing your full solution from scratch
- Handling very long inputs (it has a character limit)
- Remembering what you asked earlier (each message is independent)
- Tasks that need actual computation

**Quick check:** You get a shape error: `ValueError: operands could not be broadcast together with shapes (100, 10) and (100,)`. Would you ask the AI to fix your whole script, or just ask it what this error means?

> Ask it to explain the error. Paste only the 2-3 lines of code that caused it. The AI can explain broadcasting rules in seconds. Writing your whole script yourself is still your job.

---

## 2. The Input/Output Limit Problem

The contest AI has limits on how long your message can be and how long its reply can be. This forces you to be specific.

A bad prompt wastes your limit and gets a vague answer:

```
Here is my full training script, 200 lines. What is wrong with it?
```

A good prompt is short and specific:

```
This line gives a RuntimeError: Expected all tensors on same device.
  loss = criterion(outputs, labels)
My model is on cuda, labels are a plain Python list. What do I do?
```

**Rule:** Paste the minimum code needed to show the problem. One function, one error line, never the whole file.

---

## 3. Good Prompt Patterns

Memorize these patterns. Use them during the contest.

### Pattern 1: Explain an error

```
I get this error:
  KeyError: 'label'
when I run: df['label']
df was loaded from train.csv. How do I check what columns exist?
```

### Pattern 2: Clarify a function

```
What does sklearn's StratifiedKFold do differently from regular KFold?
One sentence only.
```

Short prompts get short, useful answers. The "one sentence only" instruction prevents the AI from writing a 400-word essay when you need a 15-word answer.

### Pattern 3: Debug logic

```
I want to normalize each row of a 2D numpy array to sum to 1.
My code: arr / arr.sum()
Result: columns sum to 1, not rows. What am I missing?
```

### Pattern 4: Understand a metric

```
The task says minimize log-loss. My model outputs class probabilities.
What value of log-loss is "good" for a 3-class problem?
```

### Pattern 5: Dataset questions

```
My feature 'age' has values like 25, 30, 999, 0.
999 and 0 look like missing-value codes. How do I confirm this and handle it?
```

---

## 4. What NOT to Ask

The AI is a tool, not your teammate. The following prompts will not help you:

**"Write me a model that classifies images in the training folder."**
You will get generic boilerplate that may not match the task format, which takes longer to fix than writing it yourself.

**"My score is 0.72. How do I get above 0.80?"**
The AI has no idea what dataset you are working on, what your code looks like, or what the scoring curve is. This question has no answer.

**"Is my approach correct?"**
Too vague. Ask about a specific part: "I'm using softmax output and MSE loss. Why might this give poor results for a classification task?"

**The test:** Could a random stranger answer your question without seeing your screen? If not, add more context.

---

## 5. Using AI to Learn, Not to Cheat

Outside of the contest, you can use full AI tools. But there's a trap: if you ask an AI to write all your code, you will not be ready when the contest platform's AI gives you a partial answer.

**A better way to use AI while studying:**

1. Try to solve the problem yourself first. Write code that fails.
2. Ask the AI to explain WHY it fails, not HOW to fix it.
3. Fix it yourself based on the explanation.
4. Then ask if your fix makes sense.

This builds the mental model. The contest AI won't write your solution, so your mental model is what wins.

**Practice exercise:**

You are trying to one-hot encode a column called `'color'` with values `['red', 'blue', 'green']`. You write:

```python
pd.get_dummies(df)
```

But you get floats instead of integers. Write a prompt (2-4 lines) that would get a useful answer from the contest AI.

> Sample answer:
> ```
> pd.get_dummies(df['color']) gives float64 columns.
> I want integer 0/1 columns. What parameter do I add?
> ```

---

## 6. The Speed Advantage

Here is the real reason to use the AI well: time. In IOAI you have a set number of hours. Every minute you spend searching documentation or re-reading error messages is a minute you are not training models or submitting.

The AI makes you faster on things you already mostly know. It is not there to replace knowledge you never had.

A rough guide for when to use it vs. not:

| Situation | Use AI? |
|---|---|
| Error message you have never seen | Yes - ask what it means |
| Picking between two model architectures | No - you should know this |
| Forgetting the exact sklearn parameter name | Yes - quick fact lookup |
| Understanding why accuracy is low | No - look at your data and model |
| What does cross-entropy measure | Yes - concept reminder |
| Should I use Adam or SGD here | No - apply what you learned |

---

## Summary

- The contest has a built-in compact AI. No external tools.
- Short, specific prompts get useful answers. Long prompts waste your limit.
- Use it to understand errors and clarify concepts, not to write code for you.
- Outside contest: use AI to learn WHY, not to get the answer.
- Speed matters. The AI saves time on lookups so you can focus on the actual problem.
