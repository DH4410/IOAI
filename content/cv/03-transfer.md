---
title: Transfer Learning
track: cv
order: 3
estimatedTime: 50
difficulty: intermediate
---

# Transfer Learning

If you remember only one strategy from this entire track, make it this one:

> **Almost never train a vision model from scratch. Start from a pretrained model.**

This single idea — **transfer learning** — is the backbone of nearly every winning IOAI computer-vision solution. It is why a student with a few hundred images and a two-hour time limit can build a model that would have been world-class research a decade ago. This lesson explains why it works and exactly how to do it.

---

## Why training from scratch is hard

A modern CNN like ResNet-50 has about **25 million** parameters. A Vision Transformer can have hundreds of millions. Each parameter is a number the model must *learn* from data.

To learn 25 million good parameters, you need a **lot** of examples. ImageNet — the dataset these models were originally trained on — has **1.2 million** labeled images across 1000 categories. Training on it takes many GPU-days.

At IOAI you have:

- A **small** dataset — often hundreds to a few thousand images.
- A **short** time limit — hours, not days.
- **Limited** compute — a single Colab/Kaggle GPU.

Train a 25-million-parameter model from random weights on 500 images and it will **overfit catastrophically**: it memorizes the training images and fails on anything new. There simply is not enough data to pin down that many parameters. From-scratch training is the wrong tool for the competition setting.

---

## The key insight: features transfer

Here is the beautiful fact that makes transfer learning work.

When a CNN trains on ImageNet, it does not just learn "this is a dog." It learns a **hierarchy of visual features**, layer by layer:

- **Early layers** learn *generic* things: edges, corners, color blobs, simple textures. These are the alphabet of vision.
- **Middle layers** combine edges into *parts*: circles, stripes, fur patterns, wheel-like shapes.
- **Late layers** combine parts into *concepts* specific to the training task: dog faces, car bodies, particular ImageNet classes.

The crucial realization: **the early and middle features are useful for almost any vision task.** Edges and textures matter whether you are classifying chickens, matching restroom icons, or authenticating paintings. Those features were expensive to learn (1.2M images, days of compute) but they are **general**.

So instead of relearning "what an edge looks like" from your 500 images, you **borrow** it. You take a model that already knows edges, textures, and parts, and you only teach it the last little bit — how *your* specific classes map onto those features.

> **Analogy:** Hiring an experienced artist and teaching them your particular style takes an afternoon. Teaching someone to hold a brush, mix paint, and see shapes — from zero — takes years. Transfer learning hires the experienced artist.

---

## Anatomy of a pretrained model: backbone + head

Think of a classification network as two parts:

1. **The backbone (feature extractor).** All the convolutional layers. It takes an image and outputs a **feature vector** — a compact numerical summary, e.g. 2048 numbers for ResNet-50. This is where all the transferable knowledge lives.
2. **The head (classifier).** Usually a single linear layer that maps the feature vector to your class scores. For ImageNet it outputs 1000 numbers (one per class).

```
image → [ BACKBONE: conv layers ] → feature vector (e.g. 2048) → [ HEAD: linear ] → class scores
             (transferable)                                          (task-specific)
```

Transfer learning is: **keep the backbone, replace the head.** ImageNet's 1000-class head is useless to you, so you throw it away and bolt on a fresh head sized for *your* number of classes (say, 5 chicken breeds). Then you decide how much of the backbone to update.

---

## Two strategies: feature extraction vs fine-tuning

### Strategy 1 — Feature extraction (freeze the backbone)

**Freeze** the entire backbone: its weights do not change during training. Only the new head learns. You are treating the pretrained backbone as a fixed "feature machine" and training a small classifier on top of its outputs.

- **Fast:** only a tiny head is trained; the backbone just runs forward.
- **Safe with little data:** few trainable parameters means little overfitting.
- **Great baseline:** this is often your very first submission (a "linear probe").

Use feature extraction when your dataset is **small** and/or **similar** to ImageNet (natural photos).

### Strategy 2 — Fine-tuning (unfreeze some/all layers)

**Unfreeze** some backbone layers so they too get updated, adapting the borrowed features to your specific data. You keep the pretrained values as a *starting point* but let training nudge them.

- **More powerful:** the features specialize to your task and can beat pure feature extraction.
- **Riskier:** more trainable parameters means more overfitting risk on small data; needs a **small learning rate** so you gently adjust rather than destroy the pretrained knowledge.

Use fine-tuning when feature extraction plateaus and you have enough data (or good augmentation) to support it.

> **The two strategies are a spectrum, not a binary.** Freeze everything → freeze most → freeze half → freeze nothing. You slide along this spectrum and keep whatever gives the best *validation* score.

---

## Freezing in PyTorch

Loading a pretrained model and freezing its backbone is only a few lines.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Load a model pretrained on ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Freeze ALL parameters (feature extraction mode)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the head with a fresh one for OUR number of classes
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)
#   The new fc layer has requires_grad=True by default,
#   so ONLY it will train.
```

`requires_grad = False` means "do not compute gradients for this parameter, so the optimizer never updates it." That is what *freezing* means mechanically. The replaced `model.fc` is brand new, so it stays trainable.

> **Note on the API:** older code uses `resnet50(pretrained=True)`. Newer torchvision uses `weights=ResNet50_Weights.DEFAULT`. Both download ImageNet-pretrained weights; you will see both in competition notebooks.

To later unfreeze the last block for fine-tuning:

```python
# Unfreeze the final residual block (layer4) of a ResNet
for param in model.layer4.parameters():
    param.requires_grad = True
```

And point the optimizer only at the trainable parameters, ideally with a small learning rate:

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
```

---

## The layer-by-layer strategy for competitions

Here is a concrete, battle-tested recipe. It balances speed, safety, and performance — exactly what you want when the clock is ticking.

**Step 1 — Freeze everything, train only the head.**
Fast and safe. Get a baseline validation score in a few minutes. This tells you whether the pretrained features already separate your classes well (they often do).

> verify: validation accuracy is reasonable and *stable* (not wildly overfitting).

**Step 2 — Unfreeze the last block, keep the rest frozen.**
Use a **small** learning rate (e.g. `1e-4`). The last block specializes the highest-level features to your task. Re-validate.

> verify: validation score went **up** vs Step 1. If it dropped, your LR is too high or you are overfitting — revert.

**Step 3 — Unfreeze more blocks, one at a time.**
Continue unfreezing deeper into the backbone, always with a small LR, always checking validation after each change.

> verify: each unfreeze either helps or you undo it.

**Step 4 — Stop when validation stops improving.**
The moment unfreezing more layers stops helping (or starts hurting), you have found the sweet spot for your data size. Keep that configuration.

The intuition: **late layers are task-specific and benefit most from adapting; early layers are generic and rarely need changing.** So you thaw from the top down, and only as far as your data can support.

A common refinement is **discriminative learning rates**: give the layers nearest the head a slightly larger LR and the early layers a tiny one, since early features need only gentle adjustment.

---

## Practical recipe (copy this to your competition notes)

```
1. Load pretrained backbone (ResNet / ViT / EfficientNet).
2. Replace head -> your number of classes.
3. Freeze backbone. Train head only. Validate.      # baseline
4. Unfreeze last block. LR ~ 1e-4. Validate.         # fine-tune top
5. Unfreeze more blocks if it still helps. Validate. # go deeper
6. Add augmentation. Validate.
7. Keep the config with the best VALIDATION score.
8. Save that model. Submit.
```

Notice how every step ends in **Validate**. Transfer learning without a trustworthy validation set is flying blind. The whole method is: make one change, measure, keep or revert.

---

## Choosing a backbone

You do not need the biggest model. Good default choices at IOAI:

- **ResNet-50 / ResNet-18** — reliable, fast, well-understood. ResNet-18 for tiny data or speed; ResNet-50 for a bit more power.
- **EfficientNet / ConvNeXt** — strong accuracy per parameter; good when you have a little more data.
- **ViT (Vision Transformer)** — excellent when large-scale pretrained weights are available (covered in a later lesson).

Bigger models are not automatically better on small competition datasets — they overfit faster and train slower. Start medium (ResNet-50), and only scale up if the data supports it.

---

## Common mistakes

**1. Forgetting to replace the head.** ImageNet's 1000-class head cannot output your 5 classes. Replace `model.fc` (ResNet) / `model.classifier` (many others) / `model.heads` (ViT).

**2. Learning rate too high when fine-tuning.** A big LR wipes out the carefully pretrained weights in the first few steps — you destroy the very knowledge you came for. Fine-tune with small LRs (`1e-4` or lower).

**3. Not actually freezing.** If you set `requires_grad = False` but then pass *all* parameters to the optimizer, some frameworks still update them. Pass only the trainable params, or double-check nothing you froze is moving.

**4. Wrong normalization.** Pretrained backbones expect ImageNet normalization (previous lesson). Feed them differently-scaled images and the borrowed features misfire.

**5. Overfitting during fine-tuning.** More trainable params + small data = overfitting. Watch the train/validation gap; add augmentation; freeze more if the gap explodes.

---

## Why this dominates at IOAI

Transfer learning wins competitions because it turns the impossible into the routine:

- **It fits the data budget.** A frozen-backbone linear probe needs only enough data to fit a small head — hundreds of images is plenty.
- **It fits the time budget.** Feature extraction trains in minutes; you can iterate many times.
- **It fits the compute budget.** A single GPU forward-pass over a frozen backbone is cheap.
- **It is a launchpad, not a ceiling.** Start with feature extraction, then fine-tune upward as far as your validation score allows.

In the IOAI 2025 CV tasks — restroom icon matching, painting authentication — and the 2026 ViT/CLIP retrieval task, the winning move was the same: **grab powerful pretrained features and adapt them carefully.** Learn this pattern cold.

---

## Sort Transfer Learning Strategies

```widget
{
  "type": "concept-sort",
  "title": "Freeze or Fine-Tune?",
  "categories": [
    { "name": "Freeze backbone", "color": "#5B5BD6" },
    { "name": "Fine-tune all layers", "color": "#F97316" }
  ],
  "items": [
    { "text": "Your dataset is very small (<500 images)", "category": "Freeze backbone" },
    { "text": "Large labeled dataset (>10k images)", "category": "Fine-tune all layers" },
    { "text": "Use pretrained features as fixed extractors", "category": "Freeze backbone" },
    { "text": "Use small lr (1e-5) to update all weights", "category": "Fine-tune all layers" },
    { "text": "Train only the classification head", "category": "Freeze backbone" },
    { "text": "Target domain very different from ImageNet", "category": "Fine-tune all layers" }
  ]
}
```

---

## Practice Questions

**Quick check:** You have 200 images of rare butterfly species, 10 per class, 20 classes. Which transfer learning strategy should you use and why?
> **Freeze backbone, train head only.** With only 200 images, fine-tuning all layers will massively overfit — the model will memorize the 200 training images. Freezing the backbone uses ImageNet features as a fixed, powerful extractor and only learns to map those features to your 20 classes. Consider also heavy augmentation.

**Quick check:** When you replace ResNet's final `model.fc = nn.Linear(512, num_classes)`, what happens to the weights of the rest of the model?
> They remain as pretrained ImageNet weights — nothing is reset. Only the new `fc` layer has randomly initialized weights. You're attaching a fresh classification head to a pre-learned feature extractor.

**Quick check:** Why do you use a 10x smaller learning rate when fine-tuning all layers compared to training the head only?
> The pretrained backbone's weights already encode rich, useful features. A normal LR would destroy them through large updates — this is called **catastrophic forgetting**. A small LR (e.g., 1e-5 instead of 1e-3) makes small adjustments to adapt to the new domain without forgetting ImageNet knowledge.

---

## Summary

- Training big vision models from scratch needs millions of images and days of compute — impossible in a competition. **Start pretrained.**
- Pretrained backbones learn **general features** (edges → textures → parts) that transfer to almost any vision task. You borrow them instead of relearning them.
- A network = **backbone (transferable)** + **head (task-specific)**. Transfer learning keeps the backbone and replaces the head.
- **Feature extraction** = freeze backbone, train head only (fast, safe, great baseline). **Fine-tuning** = unfreeze some/all layers with a small LR (more powerful, more overfitting risk).
- **Competition recipe:** freeze → validate → unfreeze last block → validate → unfreeze more if it helps → keep the best-validation config → save.
- In PyTorch: `requires_grad = False` freezes a parameter; replace `model.fc`; optimize only trainable params with a small LR.
- Always use the pretrained model's normalization, and let **validation** decide every step.
