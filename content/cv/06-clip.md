---
title: "CLIP: Image + Text Together"
track: cv
order: 6
estimatedTime: 65
difficulty: advanced
---

# CLIP: Image + Text Together

Everything so far has kept vision and language in separate rooms. **CLIP** knocks down the wall. It learns a single shared space where a photo of a dog and the words "a photo of a dog" land in nearly the same spot. That one idea unlocks zero-shot classification, image search by text, and text search by image — and it is **exactly** the setup of the IOAI 2026 ViT/CLIP retrieval task.

This is one of the most important lessons in the whole track. Take your time. By the end you should be able to explain how CLIP is trained, how to use it for retrieval, how it is scored, and the practical gotchas that decide competition points.

---

## The big idea: a shared embedding space

An **embedding** is a vector that represents something — an image or a piece of text — as a list of numbers. Similar things get similar vectors.

CLIP (Contrastive Language-Image Pre-training, from OpenAI, 2021) learns embeddings for **both** images and text that live in the **same** space. The magic property:

> An image and a caption that *describe the same thing* get vectors that point in nearly the same direction.

Once images and text share a space, comparing across the two is just measuring vector similarity. "Which caption matches this image?" becomes "which text vector is closest to this image vector?" That is the entire game.

---

## Dual encoder architecture

CLIP has **two separate encoders** that never share weights:

1. **Image encoder** — usually a **ViT** (from the last lesson!) or a ResNet. Turns an image into a vector.
2. **Text encoder** — a Transformer. Turns a string into a vector.

```
  image  → [ Image Encoder (ViT) ] → image vector  ─┐
                                                     ├→ same D-dim space → compare
  text   → [ Text Encoder (Transf) ] → text vector ─┘
```

Both encoders output a vector of the same dimension `D` (e.g. 512). A final learned **projection** maps each encoder's output into the shared space, and the vectors are **L2-normalized** (scaled to length 1) so that comparing them with a dot product equals measuring the **cosine similarity** — the cosine of the angle between them.

$$\text{cosine\_sim}(u, v) = \frac{u \cdot v}{\|u\| \, \|v\|}$$

After normalization $\|u\| = \|v\| = 1$, so similarity is just $u \cdot v$, ranging from `-1` (opposite) to `1` (identical direction).

---

## Contrastive pretraining on 400M pairs

CLIP was trained on about **400 million** (image, text) pairs scraped from the internet — pictures with their captions/alt-text. No human labeled these; the pairing itself is the supervision. How do you learn from that? With a **contrastive** objective.

Here is the setup for one training batch of `B` pairs:

- Encode all `B` images → `B` image vectors.
- Encode all `B` texts → `B` text vectors.
- Compute a **`B × B` similarity matrix**: entry `(i, j)` is the similarity between image `i` and text `j`.

The training signal:

> For each image, its **matching** caption (the diagonal entry) should have the **highest** similarity; every other caption in the batch should have **lower** similarity. And symmetrically for each caption.

```
            text_0  text_1  text_2  ... text_B
  image_0 [  HIGH    low     low          low  ]   <- image_0 should match text_0
  image_1 [  low     HIGH    low          low  ]
  image_2 [  low     low     HIGH         low  ]
    ...
  image_B [  low     low     low          HIGH ]
```

We want the **diagonal to be bright** and everything else **dark**. The off-diagonal pairs are the "negatives" — an image and a caption that do *not* go together. With a large batch, each image is contrasted against hundreds of wrong captions at once, which is what makes the learning signal so rich.

---

## The symmetric contrastive loss

Mechanically, CLIP treats it as **two classification problems** solved with cross-entropy over each row and each column of the similarity matrix.

- **Image → text:** for each image (a row), predict which of the `B` texts is the match. Softmax over the row; the target is the diagonal.
- **Text → image:** for each text (a column), predict which of the `B` images is the match. Softmax over the column; target is the diagonal.

The similarities are first scaled by a learned **temperature** $\tau$ before the softmax (it sharpens or softens the distribution). The total loss is the **average** of the two directions:

$$L = \frac{1}{2}\left(L_{i \to t} + L_{t \to i}\right)$$

The symmetry matters: it forces images to find their texts *and* texts to find their images, producing a genuinely shared, bidirectional space. This is the whole training objective — beautifully simple for such a powerful result.

```python
# Pseudocode of the CLIP loss (conceptual)
image_features = l2_normalize(image_encoder(images))   # (B, D)
text_features  = l2_normalize(text_encoder(texts))     # (B, D)

logits = (image_features @ text_features.T) / temperature   # (B, B)
labels = arange(B)                                          # diagonal targets

loss_i2t = cross_entropy(logits, labels)        # rows: each image -> its text
loss_t2i = cross_entropy(logits.T, labels)      # cols: each text -> its image
loss = (loss_i2t + loss_t2i) / 2
```

---

## Zero-shot classification

The most famous CLIP trick: classify images into categories the model was **never explicitly trained on**, with **no training at all**. This is "zero-shot."

The recipe:

1. Turn each class name into a sentence ("prompt"): `"a photo of a {class}"` — e.g. `"a photo of a cat"`, `"a photo of a dog"`.
2. Encode all the class prompts → a text vector per class.
3. Encode the image → an image vector.
4. Compute cosine similarity between the image and each class text.
5. The **highest-similarity class wins**.

```python
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
classes = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

image = preprocess(Image.open("mystery.jpg")).unsqueeze(0)
text  = clip.tokenize(classes)

with torch.no_grad():
    img_feat = model.encode_image(image)      # (1, D)
    txt_feat = model.encode_text(text)        # (3, D)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    sims = (img_feat @ txt_feat.T)            # (1, 3) similarities

print(classes[sims.argmax()])                 # predicted class
```

Wrapping class names in a sentence ("a photo of a …") consistently beats bare words, because CLIP was trained on natural captions, not lone nouns. This is called **prompt engineering** and it is free accuracy.

---

## Image-text retrieval (the IOAI competition task!)

**Retrieval** is CLIP's home turf and the heart of the IOAI 2026 ViT/CLIP task. Two directions:

- **Text → image:** given a text query, find the images that best match. (Search your photos by description.)
- **Image → text:** given an image, find the best-matching text from a set of candidate captions/documents.

The mechanism is always the same:

1. Encode everything into the shared space (all images, all texts).
2. For a query, compute its similarity to every candidate.
3. **Rank** candidates by similarity, highest first.
4. Return the top-K.

```python
# Text-to-image retrieval over a gallery of images
image_feats = encode_and_normalize(all_images)   # (num_images, D)
query_feat  = encode_and_normalize([query_text]) # (1, D)

sims = query_feat @ image_feats.T                # (1, num_images)
topk = sims[0].argsort(descending=True)[:5]      # indices of top-5 images
```

Because everything reduces to a matrix multiply of normalized vectors, retrieval over thousands of items is fast. The competition then measures **how high the correct item ranks** — which brings us to the metrics.

---

## Recall@K and MRR²

Retrieval is judged by **ranking metrics**, not accuracy. Two you must know:

**Recall@K** — "Is the correct answer somewhere in my top K?" It is 1 if the true match appears in the top-K results, else 0, averaged over all queries.

$$\text{Recall@K} = \frac{1}{Q}\sum_{q=1}^{Q} \mathbb{1}[\text{correct item is in top } K \text{ for query } q]$$

Recall@1 is strict (must be #1); Recall@5, Recall@10 are more forgiving. Reporting several K values is standard.

**MRR (Mean Reciprocal Rank)** — rewards putting the correct answer *high*, not just present. If the correct item is at rank $r_q$ for query `q`, its reciprocal rank is $1/r_q$ (rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25). Average over queries:

$$\text{MRR} = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{r_q}$$

**MRR²** — the variant used in the IOAI 2026 ViT/CLIP task squares the reciprocal rank, punishing lower ranks even harder:

$$\text{MRR}^2 = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{r_q^2}$$

Compare the reward for each rank:

| Rank $r_q$ | $1/r_q$ (MRR) | $1/r_q^2$ (MRR²) |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.500 | 0.250 |
| 3 | 0.333 | 0.111 |
| 5 | 0.200 | 0.040 |
| 10 | 0.100 | 0.010 |

The squaring means **only near-perfect ranking pays**. Getting the answer to rank 2 instead of rank 1 costs you far more under MRR² than under MRR. The competition lesson is blunt: **optimize to get the correct item to rank 1**, because MRR² barely rewards anything else.

> **Always compute the exact metric the task specifies on your validation set.** If the task is MRR², measuring plain accuracy or Recall@5 during development can mislead you about which change actually helps.

---

## Fine-tuning CLIP: freeze backbone, train projections

Zero-shot CLIP is a strong baseline, but you can usually beat it by adapting CLIP to the competition's specific images and text. The **safe, data-efficient** recipe (echoing transfer learning):

1. **Freeze** both encoders (the expensive ViT and text Transformer).
2. **Train only the projection heads** — the small final layers that map into the shared space — on your data, using a contrastive loss over your (image, text) pairs.
3. Validate with the real metric (e.g. MRR²).
4. If it helps and you have data, unfreeze the top encoder blocks with a *tiny* learning rate.

Training only the projections has few parameters, so it resists overfitting on small competition datasets and runs fast. Full fine-tuning of both giant encoders on a few hundred pairs usually overfits and *loses* the general knowledge that made CLIP good — so start small and unfreeze cautiously, always checking validation.

---

## Long text chunking: the 77-token limit

A crucial, easy-to-miss practicality: CLIP's text encoder has a **maximum length of 77 tokens** (about 50–60 words). Anything longer is **silently truncated** — the tail of your text is thrown away without warning.

If your candidate "texts" are long documents (paragraphs, product descriptions, articles) and you naively feed them in, CLIP only reads the first ~60 words and ignores the rest. The signal you needed to match might be in the discarded part. This is a real, points-costing bug at competitions.

The fix is **chunking**: split a long text into overlapping pieces that each fit under 77 tokens, encode each chunk, and combine.

- **Split** the document into chunks of ≤ 77 tokens (often with a little overlap so ideas spanning a boundary aren't lost).
- **Encode** each chunk into a vector.
- **Aggregate** — for matching, a common choice is to take the **max** similarity over the chunks (the document matches if *any* chunk matches the query), or the **mean** vector.

```python
def encode_long_text(text, model, tokenizer, max_tokens=77, stride=60):
    tokens = tokenizer(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), stride)]
    vecs = [l2_normalize(model.encode_text(c)) for c in chunks]
    return vecs   # aggregate later (e.g. max similarity per query)
```

> **Never blindly truncate at 77 tokens** when the task involves long text. Chunk it. This one detail separated competitive from non-competitive solutions on text-heavy retrieval.

---

## This is EXACTLY the IOAI 2026 ViT/CLIP task

Bringing it together, the IOAI 2026 ViT/CLIP task is an **image-text retrieval** problem scored with **MRR²**. Your playbook:

1. **Use a pretrained CLIP** (a ViT-based image encoder + text encoder) — do not train from scratch.
2. **Establish a zero-shot baseline:** encode all images and texts, rank by cosine similarity, measure MRR² on validation.
3. **Handle long text with chunking**, never blind 77-token truncation.
4. **Improve prompts** (natural sentence templates beat bare words).
5. **Fine-tune the projections** on the provided pairs (freeze the backbones), validating MRR² at each step.
6. **Optimize for rank-1**, because MRR² rewards almost nothing else — small gains in ranking the true item to the top move the score the most.

Every concept in this lesson maps directly onto that task. Master CLIP and you master a very large fraction of modern multimodal AI — and a specific IOAI medal opportunity.

---

## Common pitfalls

- **Forgetting to L2-normalize** before computing similarity — then dot products are dominated by vector magnitude, not direction, and ranking breaks.
- **Blind 77-token truncation** on long documents — chunk instead.
- **Bare class names** instead of prompt sentences — costs zero-shot accuracy.
- **Full fine-tuning on tiny data** — overfits and erases CLIP's general knowledge; train projections first.
- **Optimizing the wrong metric** — if the task is MRR², measure MRR² on validation, not accuracy.
- **Mismatched preprocessing** — CLIP uses its *own* normalization constants (not standard ImageNet); use the provided `preprocess`.

---

## Sort CLIP vs Traditional Classification

```widget
{
  "type": "concept-sort",
  "title": "Zero-Shot CLIP or Standard Classifier?",
  "categories": [
    { "name": "Use CLIP (zero-shot)", "color": "#5B5BD6" },
    { "name": "Use standard classifier", "color": "#F97316" }
  ],
  "items": [
    { "text": "Need to classify 1000+ classes, no training data per class", "category": "Use CLIP (zero-shot)" },
    { "text": "Fixed set of 10 classes with thousands of labeled examples", "category": "Use standard classifier" },
    { "text": "Adding new categories without retraining", "category": "Use CLIP (zero-shot)" },
    { "text": "Maximum accuracy on a well-defined task", "category": "Use standard classifier" },
    { "text": "Image-text retrieval across large galleries", "category": "Use CLIP (zero-shot)" },
    { "text": "Specialized domain with available labeled data", "category": "Use standard classifier" }
  ]
}
```

---

## Practice Questions

**Quick check:** You're computing CLIP cosine similarity between image embeddings A and text embeddings B. A has shape (1000, 512) — 1000 images. B has shape (5000, 512) — 5000 captions. What shape is the similarity matrix, and what does entry [i, j] represent?
> Shape **(1000, 5000)**. Entry [i, j] = cosine similarity between image i and caption j. To find the best caption for image i, take `argmax` along row i.

**Quick check:** Two CLIP image embeddings have cosine similarity 0.95. What does this mean in practical terms?
> The images are very semantically similar — they likely show the same type of scene or object. CLIP embeddings capture semantic meaning (what's in the image), not pixel-level similarity. Two photos of cats taken from different angles could have cosine similarity ~0.9.

**Quick check:** CLIP's text encoder has a 77-token limit. You have product descriptions averaging 200 tokens. How should you handle this for IOAI retrieval?
> Don't blindly truncate — you'd lose the end of descriptions. Options: (1) **chunk** each description into overlapping 77-token segments and average their embeddings, (2) use only the first sentence (often most informative), or (3) summarize with an LLM first. Validate each approach on your metric.

**Quick check:** The IOAI 2026 metric is MRR² (mean reciprocal rank squared). Why does this metric almost exclusively reward getting the answer at rank 1?
> MRR² = (1/Q)·Σ(1/r²). If r=1: score = 1. If r=2: score = 0.25. If r=3: score = 0.11. The r² denominator makes rank 2 worth only ¼ of rank 1 — getting the right answer second is devastating. Optimize your retrieval to always put the true item first.

---

## Summary

- **CLIP** learns a **shared embedding space** for images and text via **two encoders** (a ViT image encoder + a text Transformer), so matching image and caption get nearly identical, L2-normalized vectors compared by **cosine similarity**.
- It is trained **contrastively** on ~400M pairs: build a `B×B` similarity matrix and make the **diagonal** (true pairs) high, off-diagonal low, with a **symmetric loss** $L = \tfrac12(L_{i\to t} + L_{t\to i})$.
- **Zero-shot classification:** compare an image to `"a photo of a {class}"` prompts and pick the most similar. Prompt templates beat bare words.
- **Retrieval:** encode everything, rank candidates by similarity, take top-K — this is the IOAI 2026 task.
- Metrics: **Recall@K** (is the answer in the top K?) and **MRR²** $= \frac{1}{Q}\sum 1/r_q^2$, which rewards almost only **rank-1** — so optimize to put the true item first.
- **Fine-tune the projections** (freeze backbones) on competition pairs; validate the real metric.
- **Chunk long text** — never blindly truncate at CLIP's **77-token** limit. Remember to **L2-normalize** and use CLIP's own preprocessing.
