---
title: Tokenization Deep Dive
track: nlp
order: 2
estimatedTime: 30
difficulty: intermediate
---

# Tokenization Deep Dive

The previous lesson split text on spaces and called that "tokenization." Modern NLP models (BERT, GPT) use smarter tokenization that handles rare and unknown words better. This lesson explains how it works and why it matters.

---

## 1. The Problem with Word Tokenization

Simple word splitting has two big problems:

**Vocabulary explosion:** English has 500,000+ words. Storing a vector for each one is expensive. And competitions have domain-specific terms that won't be in any vocabulary.

**Unknown words:** If the word "IOAI" never appeared in training, word tokenization gives up and replaces it with `[UNK]`. You lose all information.

Subword tokenization solves both.

---

## 2. Subword Tokenization

The idea: break rare words into pieces, keep common words whole.

```
"playing"   -> ["play", "##ing"]
"IOAI"      -> ["IO", "##AI"]   or  ["I", "##O", "##A", "##I"]
"the"       -> ["the"]   (common, keep whole)
```

The `##` prefix means "this piece continues the previous token without a space."

This way the vocabulary stays small (30,000–50,000 pieces is enough) and unknown words can still be represented.

Try it live — type any sentence and see how it gets split into tokens:

```widget
{
  "type": "tokenizer-live",
  "title": "Live tokenizer — type text to see subword splitting",
  "default": "machine learning is amazing for IOAI"
}
```

Notice how uncommon words get split into pieces (shown with `##`), while common words like "is" stay whole. Real tokenizers (like BPE used in GPT) use more sophisticated splitting — but this shows the core idea.

---

## 3. Using a Pretrained Tokenizer

In practice, you use the tokenizer that came with the model. Each pretrained model has its own tokenizer:

```python
from transformers import AutoTokenizer

# Load the tokenizer for BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "Machine learning is fascinating!"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['machine', 'learning', 'is', 'fascinating', '!']

# Convert to token IDs (numbers)
ids = tokenizer.encode(text)
print(ids)
# [101, 3698, 4083, 2003, 17359, 999, 102]
# 101 = [CLS] token, 102 = [SEP] token
```

`[CLS]` (start) and `[SEP]` (separator) are special tokens BERT needs. The tokenizer adds them automatically.

**Quick check:** Why does the tokenizer add `[CLS]` and `[SEP]` tokens?
> These are special tokens BERT was trained to expect. `[CLS]` is used as a sentence-level representation for classification. `[SEP]` marks the end of a sentence (or separates two sentences in tasks like question answering).

---

## 4. Padding and Truncation

Models need all inputs to be the same length. Real text is not the same length. Fix: pad short texts with zeros, truncate long texts.

```python
# Batch of texts with different lengths
texts = [
    "Short text",
    "This is a much longer piece of text that has more words in it"
]

# Encode with padding and truncation
encoded = tokenizer(
    texts,
    padding=True,      # pad to the longest in the batch
    truncation=True,   # cut off at max_length
    max_length=20,
    return_tensors='pt'  # return PyTorch tensors
)

print(encoded['input_ids'].shape)        # (2, 20)
print(encoded['attention_mask'].shape)   # (2, 20)
```

The `attention_mask` tells the model which positions are real tokens (1) and which are padding (0). The model ignores padding positions.

---

## 5. The Attention Mask

```python
for i, text in enumerate(texts):
    print(f'Text: "{text}"')
    print(f'IDs:  {encoded["input_ids"][i].tolist()}')
    print(f'Mask: {encoded["attention_mask"][i].tolist()}')
    print()
```

Positions with `mask=1` are real tokens. Positions with `mask=0` are padding. The model will not attend to padding positions.

---

## 6. Tokenizing for Classification

Here is the complete pattern you will use for most NLP tasks:

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Example batch
texts = [
    "I really enjoyed this movie",
    "The film was absolutely terrible",
    "An okay experience, nothing special"
]

batch = tokenize_texts(texts)

print('Input IDs shape:', batch['input_ids'].shape)
print('Attention mask shape:', batch['attention_mask'].shape)
# Both: torch.Size([3, 128])
```

This is the input format BERT expects. Next lesson: turn these into embeddings and feed them into a classifier.

---

## 7. Vocabulary Size

Different tokenizers have different vocabulary sizes:

| Model | Vocabulary Size |
|---|---|
| BERT (base) | 30,522 |
| RoBERTa | 50,265 |
| GPT-2 | 50,257 |
| LLaMA 2 | 32,000 |

A larger vocabulary = more tokens are kept whole = shorter sequences but bigger embedding tables. A smaller vocabulary = sequences are longer but fewer embeddings to store.

**Quick check:** You are tokenizing code snippets. Would you expect more or fewer subword splits than for normal English?
> More. Code has many rare variable names, function names, and symbols that the tokenizer has not seen. They get split into smaller pieces.

---

## Sort the Tokenization Concepts

```widget
{
  "type": "concept-sort",
  "title": "Tokenizer Concepts: True or Word-Level Problem?",
  "categories": [
    { "name": "Advantage of subword tokenization", "color": "#22C55E" },
    { "name": "Problem with word-level tokenization", "color": "#EF4444" },
    { "name": "Required for model input", "color": "#5B5BD6" }
  ],
  "items": [
    { "text": "Unknown words replaced with [UNK], losing all information", "category": "Problem with word-level tokenization" },
    { "text": "Vocabulary of 500k+ words is too expensive to store", "category": "Problem with word-level tokenization" },
    { "text": "Rare words split into known subpieces — nothing is truly unknown", "category": "Advantage of subword tokenization" },
    { "text": "Vocabulary stays manageable at ~30k–50k pieces", "category": "Advantage of subword tokenization" },
    { "text": "attention_mask marks real tokens vs. padding positions", "category": "Required for model input" },
    { "text": "All sequences padded to the same length in a batch", "category": "Required for model input" },
    { "text": "[CLS] token at the start for sentence-level classification", "category": "Required for model input" }
  ]
}
```

---

## Summary

| Concept | Key point |
|---|---|
| Subword tokenization | Rare words split into pieces; common words kept whole |
| `[CLS]` / `[SEP]` | Special tokens added automatically by the tokenizer |
| Padding | Short texts padded with zeros to match batch length |
| `attention_mask` | 1 = real token, 0 = padding (model ignores padding) |
| Truncation | Long texts cut off at `max_length` |

Always use the tokenizer that matches your pretrained model. Do not mix tokenizers - each model was trained with specific token IDs, and using the wrong tokenizer will give meaningless inputs.
