---
title: Text Generation
track: nlp
order: 7
estimatedTime: 35
difficulty: advanced
---

# Text Generation

Text generation is the task of producing new text given a prompt or input. Models like GPT do this by predicting the next word, one at a time. This lesson covers the key concepts and how to use generation in practice.

---

## 1. How Generation Works

A language model learns the probability of a word given the previous words:

```
P("Paris" | "The capital of France is")   = very high
P("banana" | "The capital of France is")  = very low
```

To generate text, the model:
1. Takes the input prompt
2. Predicts the next token (by sampling from the probability distribution)
3. Appends that token to the input
4. Repeats until a stop condition (max length, or `[EOS]` token)

This is called **autoregressive generation**.

---

## 2. Decoding Strategies

How you sample the next token changes the output.

**Greedy:** Always pick the most likely next token. Fast but repetitive and boring.

**Beam search:** Keep the top-k most likely sequences at each step and pick the best complete sequence. Better than greedy but still sometimes too formulaic.

**Sampling with temperature:** Sample randomly from the distribution, scaled by temperature:
- Temperature = 1.0: normal sampling
- Temperature < 1.0: more focused (less random)
- Temperature > 1.0: more random/creative

**Top-p sampling (nucleus):** Sample only from the smallest set of tokens whose total probability exceeds p (e.g. p=0.9). Prevents picking very unlikely tokens while allowing diversity.

---

## 3. Generating with GPT-2

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

prompt = "The future of machine learning is"
inputs = tokenizer(prompt, return_tensors='pt')

# Greedy generation
output = model.generate(
    inputs['input_ids'],
    max_new_tokens=50
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```python
# Sampling with temperature and top-p
output = model.generate(
    inputs['input_ids'],
    max_new_tokens=100,
    do_sample=True,         # enable sampling
    temperature=0.8,        # slightly focused
    top_p=0.92,             # nucleus sampling
    repetition_penalty=1.2  # penalize repeating the same tokens
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Quick check:** You set `temperature=0.1` and the output becomes almost the same every time you run it. Why?
> Low temperature collapses the distribution toward the highest-probability token. Almost every sample picks the same top token, making generation nearly deterministic.

---

## 4. Sequence-to-Sequence Generation

Some tasks need the model to transform input text into output text (translation, summarization, question answering). These use an encoder-decoder architecture (T5, BART).

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# T5 uses text prompts to specify the task
input_text = "summarize: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It has applications in image recognition, natural language processing, and many other fields."

inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

outputs = model.generate(
    inputs['input_ids'],
    max_new_tokens=60,
    num_beams=4,        # beam search
    early_stopping=True
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

T5 treats every task as text-to-text. The task is specified as a prefix:
- `"summarize: ..."` for summarization
- `"translate English to French: ..."` for translation
- `"question: ... context: ..."` for QA

---

## 5. Perplexity

Perplexity measures how well a language model predicts a text. Lower = better.

```python
import torch
import math

def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss.item()
    return math.exp(loss)

ppl = compute_perplexity(model, tokenizer, "The quick brown fox jumps over the lazy dog.")
print(f'Perplexity: {ppl:.2f}')
```

Perplexity is used to evaluate language models and compare different models on the same text.

---

## 6. Generation in Competition Context

In IOAI, generation tasks usually appear as:
- **Conditional text classification:** Is this generated text of good quality?
- **Retrieval-augmented tasks:** Find relevant passages, then generate an answer
- **Summarization evaluation:** Score generated summaries

You rarely need to write a generation model from scratch. Instead:

```python
# Evaluation pattern for generated text
from transformers import pipeline

# Use a pipeline for quick results
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
result = summarizer(
    "Long article text here...",
    max_length=100,
    min_length=30,
    do_sample=False
)
print(result[0]['summary_text'])
```

---

## Sort the Generation Strategies

```widget
{
  "type": "concept-sort",
  "title": "More Creative or More Accurate?",
  "categories": [
    { "name": "Deterministic / Accurate", "color": "#5B5BD6" },
    { "name": "Creative / Diverse", "color": "#F97316" }
  ],
  "items": [
    { "text": "Greedy decoding", "category": "Deterministic / Accurate" },
    { "text": "Temperature > 1.0", "category": "Creative / Diverse" },
    { "text": "Beam search (num_beams=4)", "category": "Deterministic / Accurate" },
    { "text": "Top-p sampling (p=0.9)", "category": "Creative / Diverse" },
    { "text": "Temperature = 0 (argmax)", "category": "Deterministic / Accurate" },
    { "text": "Top-k sampling (k=50)", "category": "Creative / Diverse" }
  ]
}
```

---

## Practice Questions

**Quick check:** What does temperature=0.1 produce compared to temperature=2.0 from the same model?
> **Temperature=0.1**: very "peaked" distribution — the highest-probability token is chosen almost deterministically every time. Very predictable, repetitive output. **Temperature=2.0**: very "flat" distribution — many tokens have similar probability. Output is creative, diverse, and sometimes nonsensical. Temperature=1.0 is standard sampling.

**Quick check:** Beam search with num_beams=4 keeps track of 4 candidate sequences at each step. Why might it produce worse output than top-p sampling for open-ended creative text?
> Beam search optimizes for the most probable sequence — which tends to produce repetitive, generic text ("the the the the"). It can get stuck in local optima. Sampling (especially top-p) introduces randomness that allows exploration of diverse, creative continuations that would have low probability at each individual step.

**Quick check:** You're fine-tuning T5 for summarization. What's the input and output format, and which loss is used?
> **Input**: `"summarize: " + full_article_text` (encoder input). **Output**: the summary (decoder target). Loss is **cross-entropy** over the target tokens — the model is trained to predict each summary token given the article (encoder) and previous summary tokens (decoder). This is standard seq2seq training.

---

## Summary

| Strategy | When to use |
|---|---|
| Greedy (`do_sample=False`) | When you want deterministic output |
| Beam search (`num_beams=4`) | Better quality, less random |
| Sampling + temperature | Creative, diverse text |
| Top-p sampling | Balance creativity and coherence |
| T5 / BART | Sequence-to-sequence: summarization, translation |

For IOAI competition: use `pipeline()` for quick prototyping. For fine-tuning generation models, use `T5ForConditionalGeneration` with a sequence-to-sequence training loop.
