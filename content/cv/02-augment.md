---
title: Data Augmentation
track: cv
order: 2
estimatedTime: 40
difficulty: intermediate
---

# Data Augmentation

Deep learning has a golden rule: **more data means better models.** But at a competition you get a fixed dataset. You cannot go photograph 10,000 more chickens the night before the leaderboard closes.

**Data augmentation** is the trick that gets you more data without collecting any. You take the images you already have and create modified copies — flipped, rotated, brightened, cropped — and train on those too. Each modified copy is a *new* example as far as the network is concerned.

> A photo of a cat that is flipped left-to-right is still a photo of a cat. But to the network it looks like a brand-new image. Free data.

Augmentation is one of the highest-value, lowest-effort things you can do in an IOAI computer-vision task. This lesson shows you what the transforms are, when to use each one, and — just as importantly — **when not to**.

---

## Why augmentation works

A neural network learns from patterns in the training data. If every cat in your dataset faces left, the network might secretly learn "cats face left." Then at test time a right-facing cat confuses it. That is **overfitting** — memorizing accidental details instead of the real concept.

Augmentation fights overfitting in two ways:

1. **It grows the effective dataset.** One image becomes dozens of variants. The network sees more variety and generalizes better.
2. **It teaches invariances.** By showing flipped, rotated, and recolored versions, you tell the network "these differences do not matter — a cat is a cat regardless of orientation or lighting." The network stops relying on those accidental cues.

The result is almost always a **higher validation score**, which is what actually matters at IOAI. Augmentation is essentially free accuracy.

---

## The core geometric transforms

### Horizontal flip

Mirror the image left-to-right. This is the single most useful augmentation for natural photos, because the real world is roughly left-right symmetric. A dog, a car, a chicken, a person — all still make sense mirrored.

```python
import numpy as np
# arr has shape (H, W, C)
flipped = arr[:, ::-1, :]   # reverse the width axis
```

Use it almost always for object/scene classification. It effectively **doubles** your dataset for free.

### Vertical flip

Mirror top-to-bottom. This is **much more dangerous**. Most photos have a clear "up." A flipped-upside-down car, face, or building is unnatural and never appears at test time. Training on it can *hurt*.

Vertical flip is fine only when there is **no natural orientation**: satellite/aerial imagery, microscope slides, textures, medical scans sometimes. Ask yourself: "Would this image ever appear upside-down in the test set?" If no, skip vertical flip.

### Rotation

Rotate the image by a small angle (e.g. ±15°). This helps when objects can appear tilted. Big rotations (90°, 180°) are usually too aggressive unless orientation truly does not matter.

Rotation introduces **empty corners** (the image is a rectangle; rotate it and the corners have no pixels). Libraries fill these with black, reflection, or edge-repeat. Keep angles small so the empty regions stay small.

### Crop and resize

- **Random crop:** cut out a random rectangular region and use just that piece. This forces the network to recognize objects from partial views and different positions.
- **Resize:** scale the image to the fixed size your model needs (e.g. 224×224). Nearly every model requires a fixed input size, so resize is almost always in the pipeline.

A very common combo is **RandomResizedCrop**: randomly pick a crop of varying size and location, then resize it back to the target. This single transform gives you scale, position, and aspect-ratio variety all at once. It is the workhorse of ImageNet training.

---

## Color transforms: ColorJitter

The world's lighting is never identical. The same chicken photographed at noon and at dusk has different brightness and color. **Color jitter** randomly nudges these properties so the network becomes robust to lighting changes.

The four knobs:

- **Brightness** — makes the whole image lighter or darker
- **Contrast** — stretches or compresses the gap between dark and light pixels
- **Saturation** — makes colors more vivid or more gray
- **Hue** — shifts colors around the color wheel (red → orange → yellow …)

```python
import torchvision.transforms as T

jitter = T.ColorJitter(
    brightness=0.2,   # up to +/-20%
    contrast=0.2,
    saturation=0.2,
    hue=0.05,         # hue is sensitive; keep it small
)
```

**Hue is the risky one.** Large hue shifts can turn a red apple green, and if color is a real clue for your task (traffic lights! fruit ripeness!), scrambling hue destroys information. Keep hue small (≤ 0.05) or turn it off for color-sensitive tasks.

---

## Normalization: the ImageNet mean and std

Normalization is technically a transform, and it is **not optional** when you use pretrained models. As we saw in the Images lesson, networks want inputs scaled to small values.

Pretrained models (ResNet, ViT, most of torchvision) were trained on **ImageNet** with a specific per-channel mean and standard deviation. If you want to reuse those weights, you **must** normalize your images with the *exact same* numbers, or the pretrained features will be miscalibrated.

$$x_{\text{norm}} = \frac{\frac{x}{255} - \mu}{\sigma}$$

The magic ImageNet constants (memorize these — you will type them constantly):

```python
mean = [0.485, 0.456, 0.406]   # R, G, B
std  = [0.229, 0.224, 0.225]   # R, G, B
```

```python
import torchvision.transforms as T

normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

> **Rule:** whatever normalization the pretrained model was trained with, you must apply the **same** at both training and inference time. Mismatched normalization silently wrecks accuracy — the model still runs, it just predicts badly.

---

## The train vs validation split

A critical subtlety: you augment the **training** set, but **not** the validation or test set. Why?

- **Training:** you *want* variety, so you apply random flips, crops, jitter. Every epoch the image looks a little different — that is the point.
- **Validation / test:** you want a **stable, honest** measurement. Random augmentation would make your score jump around randomly and stop being comparable. So the validation pipeline uses only the deterministic steps: resize, center-crop, normalize. No randomness.

```python
import torchvision.transforms as T

train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),                         # -> CHW, /255
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])
```

Notice both pipelines end with the **same** `ToTensor` + `Normalize`. Only the random, data-growing steps differ.

---

## When NOT to augment

Augmentation assumes the transform preserves the label. When it does not, augmentation becomes **poison**. Watch for these traps — they show up in real competitions.

**1. Text in images.** If the task involves reading text or symbols (digits, letters, signs), a horizontal flip turns `b` into `d` and `p` into `q`. The image now has a *different* correct answer, but you kept the old label. You are teaching the model wrong things.

**2. Asymmetric / orientation-dependent tasks.** The IOAI 2025 **Restroom Icon Matching** task is about distinguishing symbols where orientation matters. Aggressive rotation or vertical flip can make one icon look like another. Left/right and up/down carry meaning.

**3. Color-critical tasks.** Ripeness of fruit, medical staining, traffic signals — hue jitter destroys the very signal you need. Turn color augmentation off or way down.

**4. Fine-grained distinctions.** The IOAI 2025 **Antique Painting Authentication** task hinges on tiny details — brush texture, subtle color, craquelure. Heavy blur, strong jitter, or aggressive crops can wash out the exact micro-features that separate real from fake. Augment gently.

**5. When the test set is "clean."** If test images are all perfectly centered, upright, and well-lit, extreme augmentation trains the model for conditions it will never face — wasted capacity.

> **The one question to always ask:** *"After this transform, is the label still correct — and does this variant resemble something the test set could contain?"* If either answer is no, drop that augmentation.

---

## Augmentation in PyTorch: torchvision.transforms

`torchvision.transforms` is the standard toolkit. You chain transforms with `Compose`, and they run in order on every image as it is loaded. The pipeline is applied **on the fly** each epoch, so the model rarely sees the exact same pixels twice — that is what makes it powerful.

```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = ImageFolder("data/train", transform=train_tf)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
```

A few library facts worth knowing:

- Order matters: geometric transforms (crop, flip, rotate) usually come **before** `ToTensor`; `Normalize` always comes **last**.
- `ToTensor()` does two jobs: converts to a `(C, H, W)` float tensor **and** divides by 255. So `Normalize` after it operates on `[0, 1]` values — which is why the ImageNet mean/std are small decimals, not 0–255 numbers.
- Newer projects often use **Albumentations** (faster, more transforms) or `torchvision.transforms.v2`. The concepts are identical.

---

## How augmentation improves IOAI scores

Concretely, here is why augmentation earns you leaderboard points:

- **It closes the train/test gap.** Without augmentation, models overfit small competition datasets and score much lower on the hidden test set than on training. Augmentation shrinks that gap.
- **It is nearly free.** No extra data collection, a few lines of code, and it stacks with everything else (transfer learning, better architectures).
- **It is a reliable first improvement.** After you build a baseline, adding sensible augmentation is one of the first things to try — it almost never hurts (when chosen correctly) and often gives a solid bump.
- **Test-Time Augmentation (TTA):** an advanced trick — at *inference*, run several augmented versions of each test image, then average the predictions. This often squeezes out a final fraction of a percent, sometimes the difference between medals.

> **Competition heuristic:** start with horizontal flip + RandomResizedCrop + mild ColorJitter + ImageNet normalize. Validate. Then add or remove transforms one at a time and keep whatever raises the *validation* score. Let the data decide.

---

## A practical recipe

1. **Always:** resize/crop to the model's input size + normalize with the pretrained model's mean/std.
2. **Almost always:** horizontal flip (unless text/orientation matters).
3. **Usually helpful:** RandomResizedCrop, mild ColorJitter (brightness/contrast/saturation).
4. **Careful:** rotation (keep angles small), hue jitter (keep tiny).
5. **Rarely:** vertical flip (only when there is no natural "up").
6. **Validate every change.** If validation score drops, that augmentation is wrong for this task — remove it.

---

## Summary

- Augmentation creates new training examples by transforming existing images, fighting overfitting and teaching invariances — usually raising your validation score for almost no effort.
- **Horizontal flip** is the safest, highest-value transform. **Vertical flip** is dangerous unless orientation is meaningless.
- **RandomResizedCrop** gives scale/position/aspect variety; **ColorJitter** gives lighting robustness (keep **hue** small).
- **Normalize with the pretrained model's exact mean/std** — for ImageNet models that is `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Same normalization at train and test.
- Augment the **train** set only; keep **validation/test** deterministic (resize + center-crop + normalize).
- Do **not** augment when it breaks the label: text, orientation-sensitive icons, color-critical or fine-grained tasks.
- Always ask: *is the label still correct, and could the test set contain this variant?*
