---
title: Tokenization — BPE and WordPiece
track: nlp
order: 2
estimatedTime: 50
difficulty: intermediate
---

# Tokenization — BPE and WordPiece

## Why Simple Word Tokenization Fails

In Lesson 1 we split text on whitespace. That works well for clean English text, but breaks down quickly in the real world. Let's look at the problems.

### The Out-Of-Vocabulary Problem

Any word-level tokenizer has a fixed vocabulary built from training data. What happens when it sees a word not in that vocabulary?

```python
vocab = {"the", "cat", "sat", "on", "mat", "dog", "ran"}

test_sentences = [
    "the skateboarding cat", 
    "a cryptocurrency scam",
    "COVID-19 pandemic response",
    "TikTok influencer",
]

for sentence in test_sentences:
    tokens = sentence.lower().split()
    result = [t if t in vocab else '<UNK>' for t in tokens]
    print(f"'{sentence}' → {result}")

# 'the skateboarding cat' → ['the', '<UNK>', 'cat']
# 'a cryptocurrency scam' → ['<UNK>', '<UNK>', '<UNK>']
# 'COVID-19 pandemic response' → ['<UNK>', '<UNK>', '<UNK>']
# 'TikTok influencer' → ['<UNK>', '<UNK>']
```

When a sentence is entirely unknown words, the model sees nothing but `<UNK>` tokens. This is catastrophic — the model has no useful input at all.

A real GPT-3 was trained in 2020. If you feed it "metaverse" or "GPT-4", it has never seen these words. A purely word-level tokenizer would map them to `<UNK>`.

### Different Languages Are a Nightmare

Word splitting on whitespace assumes words are separated by spaces. Many languages don't work this way:

```python
# Chinese: no spaces at all
chinese = "我爱自然语言处理"
# Translation: "I love natural language processing"
# Word-level split: ["我爱自然语言处理"] — one giant token!

# German: compounds form new words
german_compound = "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz"
# Translation: "beef labelling supervision duties delegation law" — ONE word!

# Finnish: extensive morphology
# "talossanikin" = "in my house too" — ONE word

# Arabic: right-to-left, with clitics attached to words
# Many prefixes and suffixes change meaning fundamentally
```

A vocabulary of English words is completely useless for Chinese text. You would need a separate tokenizer for every language — or a smarter approach.

### Morphological Complexity

Languages form words by combining roots with prefixes and suffixes. English does this moderately; other languages do it extensively.

```python
# English morphology examples
base = "compute"
variants = [
    "compute",         # base form
    "computes",        # 3rd person singular
    "computed",        # past tense
    "computing",       # present participle
    "computation",     # noun form
    "computational",   # adjective
    "computationally", # adverb
    "recompute",       # with prefix
    "decompute",       # with different prefix
    "incomputable",    # with prefix + suffix
]

print("All forms of 'compute':")
for v in variants:
    print(f"  {v}")
```

In a word-level vocabulary, every one of these is a separate entry. A model cannot easily learn that "compute" and "computing" are related. The vocabulary grows explosively with morphological variants.

### Character N-Grams (A Partial Solution)

One idea: represent words as collections of overlapping character sequences of length n.

```python
def char_ngrams(word, n=3):
    """Extract character n-grams from a word."""
    # Add boundary markers
    padded = f"#{word}#"
    ngrams = []
    for i in range(len(padded) - n + 1):
        ngrams.append(padded[i:i+n])
    return ngrams

word = "playing"
print(f"Character 3-grams of '{word}':")
print(char_ngrams(word, n=3))
# ['#pl', 'pla', 'lay', 'ayi', 'yin', 'ing', 'ng#']

# Even unknown words share n-grams with known words
unknown = "swimming"
known = "running"
unknown_ngrams = set(char_ngrams(unknown))
known_ngrams = set(char_ngrams(known))
shared = unknown_ngrams & known_ngrams
print(f"\nShared 3-grams between '{unknown}' and '{known}': {shared}")
# {'ing', 'nin'} (or similar)
```

Character n-grams handle OOV words better — even a new word will share some n-grams with known words. But they produce many tokens per word and lose word-level structure.

The modern solution is **subword tokenization**: a principled way to split words into pieces, learned from data.

---

## Byte Pair Encoding (BPE)

**Byte Pair Encoding** was originally a data compression algorithm. It was adapted for NLP in 2016 (Sennrich et al.) and became the foundation for GPT-2 and many other models.

### The Core Idea

Start with individual characters. Repeatedly merge the most frequently co-occurring pair of symbols into a new symbol. Stop when you reach the desired vocabulary size.

### BPE Step-by-Step Worked Example

Let's work through a complete example. Our training corpus:

```
"low low low low lower lower newest newest newest widest"
```

**Step 0: Initialize**

Represent each word as a sequence of characters, with a special end-of-word marker `</w>` (so we know where words end):

```
Word          | Frequency | Character sequence
--------------|-----------|----------------------------
low           |     4     | l o w </w>
lower         |     2     | l o w e r </w>
newest        |     3     | n e w e s t </w>
widest        |     1     | w i d e s t </w>
```

**Initial symbol vocabulary:** `{l, o, w, e, r, n, s, t, i, d, </w>}`

**Step 1: Count all adjacent pairs**

We count every consecutive pair across all words, weighted by word frequency:

```python
# Let's compute this
corpus = {
    'l o w </w>': 4,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 3,
    'w i d e s t </w>': 1,
}

from collections import Counter

def get_pair_counts(corpus):
    """Count all adjacent symbol pairs, weighted by frequency."""
    counts = Counter()
    for word_seq, freq in corpus.items():
        symbols = word_seq.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            counts[pair] += freq
    return counts

pair_counts = get_pair_counts(corpus)
print("Pair counts:")
for pair, count in pair_counts.most_common(10):
    print(f"  {pair[0]} + {pair[1]} = {count}")
```

Output:
```
Pair counts:
  l + o = 6       (appears in 'low'×4 and 'lower'×2)
  o + w = 6       (same)
  e + s = 4       (newest×3 + widest×1)
  s + t = 4       (same)
  e + r = 2       
  w + e = 5       (lower×2 + newest×3)
  n + e = 3       
  w + </w> = 4    (low×4)
  e + st = 0      (not yet merged)
  ...
```

**Step 2: Merge most frequent pair**

The most frequent pair is `(l, o)` with count 6. We merge them into `lo`:

```python
def merge_pair(corpus, pair):
    """Merge a pair of symbols throughout the corpus."""
    new_corpus = {}
    bigram = ' '.join(pair)
    merged = ''.join(pair)
    
    for word_seq, freq in corpus.items():
        # Replace all occurrences of the pair with the merged symbol
        new_seq = word_seq.replace(bigram, merged)
        new_corpus[new_seq] = freq
    
    return new_corpus

# Merge ('l', 'o') → 'lo'
corpus = merge_pair(corpus, ('l', 'o'))
print("After merging (l, o):")
for word_seq, freq in corpus.items():
    print(f"  {word_seq!r}: {freq}")
```

After merge 1:
```
'lo w </w>': 4        (was: l o w </w>)
'lo w e r </w>': 2    (was: l o w e r </w>)
'n e w e s t </w>': 3
'w i d e s t </w>': 1
```

**Step 3: Merge (o, w) → 'ow'**

Next most frequent is `(o, w)` with count 6 — but wait, after merging `l+o`, we have `lo w`. The pair `(lo, w)` now has count 6.

Let's continue:

```python
# Full BPE training loop
def train_bpe(corpus, num_merges):
    """
    Train BPE on a corpus.
    
    Args:
        corpus: dict {word_string: frequency}
                Each word_string is space-separated characters,
                ending with </w>
        num_merges: how many merge operations to perform
    
    Returns:
        merges: list of (pair, merged_symbol) tuples in order
    """
    merges = []
    vocab = set()
    
    # Initialize vocabulary with all characters
    for word_seq in corpus:
        for symbol in word_seq.split():
            vocab.add(symbol)
    
    for merge_step in range(num_merges):
        # Count pairs
        pair_counts = get_pair_counts(corpus)
        
        if not pair_counts:
            break
        
        # Find best pair (most frequent; tie-break alphabetically for determinism)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]
        
        # Merge
        merged = ''.join(best_pair)
        corpus = merge_pair(corpus, best_pair)
        merges.append((best_pair, merged, best_count))
        vocab.add(merged)
        
        print(f"Merge {merge_step+1}: {best_pair[0]} + {best_pair[1]} "
              f"→ '{merged}' (count={best_count})")
    
    return merges, vocab, corpus

# Initial corpus (space-separated characters with </w> end marker)
initial_corpus = {
    'l o w </w>': 4,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 3,
    'w i d e s t </w>': 1,
}

print("=== BPE Training (10 merges) ===")
merges, final_vocab, final_corpus = train_bpe(initial_corpus, num_merges=10)

print("\nFinal corpus representation:")
for word_seq, freq in final_corpus.items():
    print(f"  {word_seq!r}: {freq}")

print("\nFinal vocabulary:", sorted(final_vocab))
```

After 10 merges, you get something like:
```
Merge 1: l + o → 'lo' (count=6)
Merge 2: lo + w → 'low' (count=6)
Merge 3: e + s → 'es' (count=4)
Merge 4: es + t → 'est' (count=4)
Merge 5: w + e → 'we' (count=5)
Merge 6: we + st → 'west' (count=3)
...
```

Key insight: **"low"** appears as a complete unit because it's frequent. **"lower"** is split as **"low" + "er" + "</w>"** — reusing the learned "low" unit! Common words become single tokens; rare words are broken into learned subwords.

### Encoding New Text with BPE

After training, you apply the learned merge rules in order to new text:

```python
def bpe_encode(word, merges):
    """
    Apply BPE merges to encode a new word.
    
    Args:
        word: string like "lowest"
        merges: list of (pair, merged) from training
    Returns:
        list of subword tokens
    """
    # Start with character-level split
    symbols = list(word) + ['</w>']
    
    # Apply merges in the order they were learned
    for (left, right), merged, _ in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == left and symbols[i+1] == right:
                symbols = symbols[:i] + [merged] + symbols[i+2:]
            else:
                i += 1
    
    return symbols

# Test on new words
test_words = ["low", "lower", "lowest", "new", "newest", "wide", "wider"]
print("BPE Encoding of test words:")
for word in test_words:
    encoded = bpe_encode(word, merges)
    print(f"  {word:12s} → {encoded}")
```

### BPE in Practice: Byte-Level BPE

GPT-2 and GPT-3/4 use **byte-level BPE**, where the initial alphabet is all 256 bytes rather than Unicode characters. This means:

- You can encode ANY text (even emojis, Chinese characters) without OOV
- The vocabulary starts at byte level and learns merges from there
- A Chinese character might be 3 bytes, so it takes 3-6 tokens to encode one character

```python
# Byte-level encoding concept
text = "Hello 🌍"
bytes_representation = text.encode('utf-8')
print(f"Text: {text!r}")
print(f"UTF-8 bytes: {list(bytes_representation)}")
print(f"As hex: {bytes_representation.hex()}")
# Text: 'Hello 🌍'
# UTF-8 bytes: [72, 101, 108, 108, 111, 32, 240, 159, 140, 141]
# The emoji uses 4 bytes: [240, 159, 140, 141]
```

---

## WordPiece

**WordPiece** is used by BERT, DistilBERT, and many other models. It is similar to BPE but differs in how it chooses which pairs to merge.

### BPE vs WordPiece

| Property | BPE | WordPiece |
|---|---|---|
| Merge criterion | Most frequent pair | Pair that maximizes likelihood of training data |
| Result | Deterministic | Probabilistic |
| Prefix marker | suffix `</w>` | prefix `##` on continuation pieces |
| Used by | GPT-2, GPT-3, RoBERTa | BERT, DistilBERT |

The key difference: WordPiece picks the merge that **increases the probability of the training data most**. This is equivalent to:

```
score(pair A, B) = freq(AB) / (freq(A) × freq(B))
```

This score is high when AB appears MORE often than you would expect given how often A and B appear independently. It finds truly "co-specialized" pairs.

### WordPiece Encoding Example

BERT's WordPiece uses `##` to mark continuation pieces:

```python
# WordPiece encoding (conceptual — actual model uses a learned vocabulary)
# The ## prefix means "this piece continues a word (not a new word)"

examples = [
    ("playing",    ["play", "##ing"]),
    ("unbelievable", ["un", "##believ", "##able"]),
    ("tokenization", ["token", "##ization"]),
    ("COVID-19",   ["CO", "##VI", "##D", "-", "19"]),
    ("hello",      ["hello"]),            # Common word = single token
]

for word, tokens in examples:
    print(f"  {word:20s} → {tokens}")
```

### Encoding a Sentence for BERT

When using BERT, you also add special tokens:

```python
# [CLS] = classification token (always first)
# [SEP] = separator token (at end, or between two sequences)
# [PAD] = padding token (fill to fixed length)
# [MASK] = masking token (used in pretraining)

# Original sentence
sentence = "The cat sat on the mat"

# After WordPiece tokenization + special tokens:
bert_tokens = ["[CLS]", "The", "cat", "sat", "on", "the", "mat", "[SEP]"]
token_ids   = [  101,    1996, 4937, 2938,  2006, 1996, 13523,  102]
#                ^CLS    the   cat   sat    on    the   mat     ^SEP
```

---

## SentencePiece

**SentencePiece** (Google, 2018) takes a different approach: it treats the text as a raw Unicode string and does not require pre-tokenization.

Key features:
1. Language-agnostic — works on Chinese, Japanese, Arabic without special handling
2. Deterministic encoding
3. Supports both BPE and Unigram model
4. Used by: T5, XLNet, ALBERT, mBERT, most multilingual models

```python
# SentencePiece treats spaces as special characters
# A space at the start of a word is represented as '▁' (U+2581)

examples = [
    "Hello, World!",
    "machine learning",
]
# Encoded as:
# ['▁Hello', ',', '▁World', '!']
# ['▁machine', '▁learning']

# The ▁ (underscore) marks the beginning of a new word
# This way the model knows where word boundaries were
```

The advantage: SentencePiece can be trained without any pre-tokenization step. You just give it raw text. This is crucial for languages without clear word boundaries like Chinese or Thai.

---

## The Unigram Language Model Tokenizer

The **Unigram** model (Kudo, 2018) works differently from BPE:

1. Start with a large vocabulary (all substrings up to some length)
2. Train a probabilistic model: what is the probability of each substring?
3. **Remove** subwords that decrease the likelihood the least, until the target vocabulary size is reached

This means it starts big and prunes, whereas BPE starts small and grows. The Unigram model also produces a probability for each tokenization, which enables sampling different segmentations at training time (data augmentation!).

```python
# Unigram model intuition
# Given the word "newest", possible segmentations:
segmentations = [
    ["n", "e", "w", "e", "s", "t"],    # character-level: low prob
    ["new", "e", "s", "t"],             # partial merge
    ["new", "est"],                     # likely high prob (est is common)
    ["newest"],                         # high prob if 'newest' is in vocab
    ["ne", "west"],                     # lower prob
]
# The Unigram model assigns probabilities to each and picks the best
# (or samples at training time)
```

---

## Implementing a Tiny BPE from Scratch

Here is a complete, runnable BPE implementation:

```python
import re
from collections import Counter, defaultdict

def get_vocab(corpus):
    """
    Convert a list of words into the initial BPE vocabulary.
    Each word is a space-separated sequence of characters + </w>.
    """
    vocab = Counter()
    for word in corpus:
        # Add spaces between characters and append end-of-word marker
        word_with_spaces = ' '.join(list(word)) + ' </w>'
        vocab[word_with_spaces] += 1
    return vocab

def get_pairs(vocab):
    """Get all adjacent symbol pairs and their frequencies."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge a given pair of symbols in the vocabulary."""
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe_train(corpus_words, num_merges=20):
    """
    Train BPE tokenizer on a corpus.
    
    Args:
        corpus_words: list of words (strings)
        num_merges: number of merge operations
    Returns:
        merges: list of merged pairs
        vocab: final vocabulary
    """
    vocab = get_vocab(corpus_words)
    merges = []
    
    print("Initial vocab sample:")
    for k, v in list(vocab.items())[:5]:
        print(f"  '{k}': {v}")
    
    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        
        best = max(pairs, key=lambda p: (pairs[p], p))
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        
        print(f"Merge {i+1:2d}: '{best[0]}' + '{best[1]}' → '{best[0]+best[1]}' "
              f"(freq={pairs[best]})")
    
    return merges, vocab

# Example corpus
corpus_words = (
    ['low'] * 5 +
    ['lower'] * 2 +
    ['newest'] * 6 +
    ['widest'] * 3 +
    ['news'] * 8 +
    ['new'] * 4 +
    ['mellow'] * 2 +
    ['yellow'] * 3
)

print("Training BPE on sample corpus...")
print(f"Unique words: {len(set(corpus_words))}, total: {len(corpus_words)}\n")
merges, final_vocab = bpe_train(corpus_words, num_merges=15)

print("\nFinal vocabulary:")
for word_rep, freq in sorted(final_vocab.items(), key=lambda x: -x[1])[:10]:
    print(f"  {freq:3d}x '{word_rep}'")
```

---

## HuggingFace Tokenizers

In practice, you use HuggingFace's tokenizers library. Here is the key code (reference — runs locally with `pip install transformers`):

```python
# Reference code (requires: pip install transformers)
from transformers import AutoTokenizer

# Load BERT's WordPiece tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The quick brown fox jumps over the lazy dog."
encoding = bert_tokenizer(text)

print("Token IDs:", encoding['input_ids'])
print("Attention mask:", encoding['attention_mask'])

# Decode back to readable tokens
tokens = bert_tokenizer.convert_ids_to_tokens(encoding['input_ids'])
print("Tokens:", tokens)
# ['[CLS]', 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', '[SEP]']

# GPT-2's BPE tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_encoding = gpt2_tokenizer(text)
gpt2_tokens = gpt2_tokenizer.convert_ids_to_tokens(gpt2_encoding['input_ids'])
print("\nGPT-2 tokens:", gpt2_tokens)
# ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.']
# Note: Ġ is GPT-2's representation of a leading space

# Tokenizing a pair of sentences (for BERT Q&A, entailment, etc.)
question = "What is the capital of France?"
context = "France is a country. Paris is the capital of France."

paired_encoding = bert_tokenizer(
    question,
    context,
    max_length=64,
    padding='max_length',
    truncation=True,
    return_tensors='np',  # or 'pt' for PyTorch, 'tf' for TensorFlow
)
print("\nPaired encoding keys:", list(paired_encoding.keys()))
# ['input_ids', 'token_type_ids', 'attention_mask']
# token_type_ids tells the model which sentence each token belongs to
```

---

## Special Tokens

Every modern tokenizer uses special tokens with reserved IDs. Understanding them is important.

```python
# BERT special tokens
special_tokens_bert = {
    '[PAD]': 0,    # Padding: fill shorter sequences to equal length
    '[UNK]': 100,  # Unknown: character not in vocabulary (rare with subwords)
    '[CLS]': 101,  # Classification: first token; its representation is used for classification
    '[SEP]': 102,  # Separator: marks end of sentence or boundary between sentences
    '[MASK]': 103, # Mask: replaces tokens during Masked Language Model pretraining
}

# GPT-2 special tokens (fewer, because it's decoder-only)
special_tokens_gpt2 = {
    '<|endoftext|>': 50256,  # End of document marker
}

# How padding works:
sentences = [
    "Short sentence.",
    "This is a much longer sentence with many more words in it.",
    "Medium length here.",
]

# Tokenize and pad to the same length
max_len = 15
padded = []
for sent in sentences:
    tokens = bert_tokenizer(sent, max_length=max_len, padding='max_length', 
                            truncation=True)
    padded.append(tokens['input_ids'])
    mask = tokens['attention_mask']
    # attention_mask: 1 for real tokens, 0 for [PAD] tokens
    # The model ignores positions where attention_mask = 0
    print(f"IDs:  {tokens['input_ids']}")
    print(f"Mask: {mask}")
    print()
```

### Why Attention Masks Matter

The attention mask tells the model which tokens are real and which are padding:

```
Sentence: "[CLS] The cat sat [SEP] [PAD] [PAD] [PAD]"
IDs:      [ 101,  1996, 4937, 2938, 102,   0,    0,    0  ]
Mask:     [  1,     1,    1,    1,   1,   0,    0,    0  ]
           ↑ real tokens ↑              ↑ padding — ignore ↑
```

Without the attention mask, the model would attend to padding tokens as if they were real words, producing garbage representations.

---

## Vocabulary Size Trade-offs

How big should the vocabulary be?

```python
# Trade-off analysis
import numpy as np

vocab_sizes = [1000, 5000, 16000, 32000, 50000, 100000]

print("Vocabulary size trade-offs:")
print(f"{'Vocab Size':>12} | {'Avg tokens/word':>17} | {'Model params (embedding)':>25}")
print("-" * 60)

embedding_dim = 768  # Like BERT

for v in vocab_sizes:
    # Larger vocab → fewer tokens per word (more info per token)
    # Rough approximation: tokens_per_word ≈ 1 + 2*(20000/v)
    avg_tokens = 1 + 2 * (20000 / v)
    avg_tokens = max(1.0, min(avg_tokens, 6.0))
    
    # Model size: embedding matrix is vocab_size × embedding_dim
    params = v * embedding_dim
    params_m = params / 1e6
    
    print(f"{v:>12,} | {avg_tokens:>17.1f} | {params_m:>22.1f}M")
```

**Small vocabulary (1,000–5,000):**
- Every word is split into many pieces → very long sequences
- Long sequences are slow and use more memory
- Common words still become single tokens

**Large vocabulary (100,000+):**
- Most words are single tokens → short sequences
- Embedding matrix is large → more model parameters
- Very rare words are still split

**Typical choices:** GPT-2 uses 50,257, BERT uses 30,522, T5 uses 32,100. These are found by experimentation to balance sequence length vs model size.

---

## How Tokenization Affects Model Performance

Tokenization is the first step in the pipeline, and mistakes here propagate everywhere.

### Case 1: Numbers and Dates

```python
# Numbers are problematic for all tokenizers
# GPT-2 tokenizes "2024" as:
# ["20", "24"] — two tokens!
# "1234567890" becomes many fragments

# This means models struggle with arithmetic:
# "What is 127 + 486?" — each digit might be a separate token
# The model has to "learn" arithmetic over token sequences, not numbers
```

### Case 2: Code

```python
# Code has special tokenization challenges:
code = "for i in range(10):"
# BPE might split 'range' but keep 'for', 'in', 'range' together
# This works better than expected — programming keywords are common

# But identifiers like 'my_variable_name' get split inconsistently:
# 'my_variable_name' → ['my', '_', 'variable', '_', 'name'] (maybe)
# This is why code models train their own specialized tokenizers (CodeBERT, CodeT5)
```

### Case 3: Morphologically Rich Languages

```python
# For Turkish, Finnish, Hungarian, Swahili...
# Words have many more forms than English

# Finnish "talossanikin" = "in my house too"
# A general vocabulary might split this as:
# ['talo', '##ssa', '##ni', '##kin'] — 4 tokens for one word
# An English word of 4 characters is often 1 token

# Turkish text will use ~3-4x more tokens than equivalent English text
# for the same information content
```

This token-length disparity is a fairness issue: non-English languages are essentially "more expensive" for the model to process, and the model gets worse-quality representations for them.

---

## Summary

| Method | Vocabulary | Handles OOV? | Handles Languages? | Speed |
|---|---|---|---|---|
| Character | 100 | Yes | Yes | Slow (long seqs) |
| Word | 50k–200k | No | Per-language | Fast |
| BPE | 32k–50k | Yes (subwords) | Byte-level: any | Fast |
| WordPiece | 30k | Yes | Multilingual | Fast |
| SentencePiece | 32k | Yes | Any | Fast |

**Key takeaways:**
1. Word tokenization fails on OOV words — unacceptable for production
2. BPE learns which character combinations are most useful from data
3. The merge order is crucial: merges are applied in learned order
4. Special tokens ([CLS], [SEP], [PAD], [MASK]) structure the input for the model
5. Vocabulary size is a trade-off between sequence length and model size
6. Tokenization differences between languages cause fairness and performance gaps

---

## Exercises

Practice in the quiz section:
- Implement one step of BPE
- Count tokens for different vocabulary sizes
- Analyze vocabulary coverage
