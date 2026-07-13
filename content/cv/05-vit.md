---
title: Vision Transformers (ViT)
track: cv
order: 5
estimatedTime: 60
difficulty: advanced
---

# Vision Transformers (ViT)

For a decade, CNNs owned computer vision. Then in 2020 a paper with a bold title — *"An Image is Worth 16×16 Words"* — showed that the **Transformer**, the architecture that revolutionized language, could match or beat CNNs on images. That model is the **Vision Transformer (ViT)**, and it powers CLIP, many modern VLMs, and the IOAI 2026 ViT/CLIP retrieval task.

The core idea is almost cheeky: **treat an image like a sentence.** Cut it into patches, call each patch a "word," and feed the sequence to a standard Transformer. This lesson unpacks how that works, why it works, and when to reach for a ViT instead of a CNN.

If you have not yet done the NLP attention lesson, that is fine — we explain everything you need here, but attention is the beating heart of this model, so it is worth revisiting both.

---

## The patch idea: an image as a sequence of tokens

A Transformer operates on a **sequence of tokens** (vectors). Text is naturally a sequence of words. An image is not — it is a 2D grid of pixels. So the first job is to turn the image into a sequence.

We could treat every pixel as a token, but a 224×224 image has 50,176 pixels, and attention cost grows with the *square* of the sequence length. That is far too expensive.

The ViT trick: **split the image into fixed-size square patches**, and treat each patch as one token.

- Take a `224 × 224` image.
- Cut it into non-overlapping patches of size `P × P`, e.g. `16 × 16`.
- Along each side you get `224 / 16 = 14` patches, so a `14 × 14` grid = **196 patches**.
- Each patch is a little `16 × 16 × 3` image chunk. Flatten and embed it into a vector. Now it is a "token."

The number of patches (tokens) for a square image is:

$$N = \frac{H \times W}{P^2}$$

For `224 × 224` with `P = 16`: $N = \frac{224 \times 224}{16^2} = \frac{50176}{256} = 196$. This is the famous **196 patches** of ViT-B/16.

> **The analogy:** a sentence is a sequence of words; a ViT image is a sequence of patches. Each patch is a "visual word." Everything the Transformer knows how to do with words, it now does with patches.

---

## Patch embedding with convolution

Each patch must become a vector of the model's hidden size `D` (e.g. 768 for ViT-Base). There is a clean trick to do this: a single **convolution** whose kernel size *and* stride both equal the patch size `P`.

- Kernel size `P × P` and stride `P` means the conv slides in non-overlapping `P`-sized jumps — exactly the patch grid.
- Set the number of output channels to `D`. Each patch produces one `D`-dimensional vector.

```python
import torch.nn as nn

patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=768,     # hidden dim D
    kernel_size=16,       # patch size P
    stride=16,            # non-overlapping
)
# input:  (N, 3, 224, 224)
# output: (N, 768, 14, 14) -> flatten spatial -> (N, 196, 768)
```

The conv output `(N, 768, 14, 14)` is flattened over the spatial grid into `(N, 196, 768)`: **196 tokens, each a 768-dim vector.** That is the sequence the Transformer will process. Elegant — a convolution, the CNN's own tool, is repurposed to *tokenize* the image.

---

## The CLS token

We now have 196 patch tokens. But for **classification** we want a single vector that summarizes the whole image. Which patch do we read? None of them — instead ViT borrows a trick from BERT: prepend a special learnable **[CLS] token** ("classification token").

- A single extra vector (learned during training) is added at the front of the sequence.
- After all the Transformer layers, the [CLS] token's output has "gathered" information from every patch via attention.
- We feed **only the [CLS] token's final vector** into the classification head.

So the sequence grows from 196 to **197 tokens**: `[CLS] + 196 patches`. For ViT-B/16 on 224×224 images, this is where the number **197** comes from.

$$196 \text{ patches} + 1 \text{ [CLS]} = 197 \text{ tokens}$$

The [CLS] token is a clever way to give the Transformer a dedicated "scratchpad" whose only job is to summarize the image for the final decision. (Some ViT variants instead average all patch tokens — "mean pooling" — but the [CLS] approach is the classic one.)

---

## Positional embeddings

A Transformer's attention is **permutation-invariant**: by itself, it has no idea whether a patch came from the top-left or bottom-right of the image. Shuffle the tokens and attention gives the same result. But *position matters* in an image — a patch of sky belongs at the top, grass at the bottom.

The fix: add a **positional embedding** to each token — a learned vector that encodes *where* the patch was. Position 0 (the [CLS] token), position 1, position 2, … each get their own added vector.

$$\text{token}_i = \text{patch\_embed}_i + \text{pos\_embed}_i$$

Now the model can tell patches apart by location. Without positional embeddings a ViT would treat the image as a bag of unordered patches and lose all spatial structure. With them, the sequence "remembers" the 2D layout.

> A neat consequence: if you want to run a pretrained ViT at a *different* image resolution, you must interpolate the positional embeddings to the new number of patches — a detail that trips people up when fine-tuning at non-standard sizes.

---

## Multi-head self-attention in ViT

The engine of the Transformer is **self-attention**. Here is the intuition in vision terms: each patch looks at *every other patch* and decides how much to pay attention to it, then updates itself with a weighted blend of the others.

For a patch showing part of a dog's ear, attention lets it "look at" the patches showing the other ear, the nose, the body — even if they are far away — and integrate that context. This is the ViT's superpower: **global reasoning from the very first layer.** A CNN only sees a small local window at a time and needs many layers to combine distant regions; a ViT connects any two patches immediately.

The math (covered in depth in the NLP attention lesson) is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Each token produces a **Query**, a **Key**, and a **Value** vector.
- `Q · K` measures how relevant every token is to every other token — a full `N × N` grid of relevance scores.
- Softmax turns scores into weights; the output is a weighted sum of Values.

**Multi-head** means doing this several times in parallel with different learned projections — each "head" can focus on a different kind of relationship (color, texture, shape, position). Their outputs are concatenated. A ViT block is: multi-head self-attention → a small feed-forward network, each wrapped with residual connections and layer normalization, stacked `L` times (12 for ViT-Base).

---

## Complexity: the $O(N^2 D)$ cost

Self-attention computes a relevance score between **every pair** of tokens. With `N` tokens that is `N²` pairs, each involving `D`-dimensional vectors. So the compute and memory cost scales as:

$$O(N^2 \cdot D)$$

This quadratic term is the ViT's Achilles heel. Double the number of patches and attention gets **four times** more expensive. This is precisely why we use `16 × 16` patches instead of single pixels:

- Pixels as tokens: `N = 50,176` → `N²` ≈ 2.5 **billion** pairs. Impossible.
- 16×16 patches: `N = 196` → `N²` ≈ 38,000 pairs. Easy.

The quadratic cost also explains why higher-resolution images (more patches) make ViTs much slower, and why an entire research area exists to make attention cheaper (linear/sparse attention). For competitions, remember: **more/smaller patches = more detail but sharply higher cost.**

---

## ViT vs CNN: inductive bias vs pretraining data

Why did it take until 2020 for Transformers to work on images? The answer is **inductive bias**.

A **CNN is born knowing** two facts about images, baked into its architecture:

- **Locality:** nearby pixels are related (convolutions look at local windows).
- **Translation equivariance:** an object is the same wherever it appears (the same filter slides everywhere).

These built-in assumptions are exactly right for images, so a CNN learns efficiently even from *modest* data. This is its **inductive bias** — helpful prior knowledge.

A **ViT knows almost nothing** to start. It does not assume nearby patches are related; it must *learn* that from data via attention. This makes it more flexible (it can learn relationships a CNN's rigid structure forbids) but far **hungrier for data**.

The famous result:

- On **small/medium** datasets, CNNs win — their built-in bias compensates for scarce data.
- On **huge** datasets (ImageNet-21k, JFT-300M, or web-scale like CLIP's 400M pairs), ViTs **overtake** CNNs — with enough data they learn better, more flexible features than a CNN's hand-designed structure allows.

> **The trade:** CNN = strong prior, data-efficient, less ceiling. ViT = weak prior, data-hungry, higher ceiling. This is why you almost always use a **pretrained** ViT — someone already paid the enormous data cost, and you inherit the payoff.

---

## ViT-B/16: the numbers to know

"ViT-B/16" decodes as: **B** = Base size, **/16** = 16×16 patches. Its canonical configuration on 224×224 input:

| Property | Value |
|---|---|
| Input image | 224 × 224 × 3 |
| Patch size `P` | 16 |
| Patch grid | 14 × 14 |
| Patch tokens `N` | 196 |
| + CLS token | **197 total tokens** |
| Hidden dim `D` | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |

Trace the pipeline once and it will stick:

```
224×224×3 image
  → split into 16×16 patches → 14×14 = 196 patches
  → patch embedding (conv) → 196 tokens of dim 768
  → prepend CLS → 197 tokens
  → add positional embeddings
  → 12 Transformer blocks (multi-head attention + FFN)
  → take CLS output (768-dim) → classification head → prediction
```

---

## Using a ViT in practice

You will almost never build a ViT from scratch. You load a pretrained one, exactly as with CNNs in the transfer-learning lesson.

```python
import timm
import torch

# 'timm' is the go-to library for pretrained vision models
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=5)

x = torch.randn(8, 3, 224, 224)   # a batch of 8 images
logits = model(x)                 # (8, 5) class scores
```

Everything from the transfer-learning lesson still applies: freeze the backbone for a fast baseline, then unfreeze blocks with a small learning rate, validate at each step. ViTs are just another backbone — a very powerful one when good pretrained weights exist.

---

## When to use ViT vs CNN for IOAI tasks

A practical decision guide:

**Reach for a ViT / attention-based model when:**
- Excellent **large-scale pretrained** weights are available (which is usually the case now).
- The task benefits from **global context** — relationships between distant parts of the image.
- You are doing **image-text** work (CLIP, VLMs) — these are built on ViTs. The IOAI 2026 ViT/CLIP retrieval task is squarely here.

**Reach for a CNN (ResNet, EfficientNet) when:**
- Your dataset is **small** and you want data-efficiency and a fast, reliable baseline.
- You need **speed** on limited compute.
- The task is local/texture-driven (fine-grained details) where a CNN's bias helps.

> **Competition reality:** you rarely must choose blindly — you try both and let the **validation score** decide. But knowing *why* each shines tells you which to try first given your data size and task.

---

## Common pitfalls

- **Wrong input size.** A ViT/16 pretrained at 224 expects 224 (or interpolated positional embeddings for other sizes). Mismatched size breaks the patch count.
- **Forgetting normalization.** Pretrained ViTs expect their training normalization — often ImageNet stats, but CLIP's ViT uses different constants. Match them.
- **Expecting data-efficiency from a *randomly initialized* ViT.** From scratch on small data, a ViT underperforms a CNN badly. Always start pretrained.
- **Ignoring the quadratic cost.** Cranking image resolution to get more patches can explode memory. Watch `N²`.

---

## Summary

- A **ViT treats an image as a sequence of patches** (visual "words") and feeds them to a standard Transformer.
- Patches: split `H×W` into `P×P` squares → `N = (H×W)/P²` patches. For 224×224 with P=16, that is **196 patches**.
- **Patch embedding** uses a conv with kernel=stride=`P`; a learnable **[CLS] token** is prepended (196 → **197 tokens**) and read out for classification.
- **Positional embeddings** are added so the model knows where each patch was; attention alone is order-blind.
- **Multi-head self-attention** lets every patch attend to every other — **global reasoning from layer one** — at cost $O(N^2 D)$, which is why patches (not pixels) are used.
- **CNNs have strong inductive bias (locality, translation-equivariance)** → data-efficient; **ViTs have weak bias** → data-hungry but higher ceiling with large-scale pretraining.
- Use pretrained ViTs (via `timm`); pick ViT for global-context and image-text tasks (CLIP/VLM, IOAI 2026), CNNs for small-data/fast baselines — and let validation decide.
