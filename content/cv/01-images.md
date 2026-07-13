---
title: Images as Tensors
track: cv
order: 1
estimatedTime: 35
difficulty: intermediate
---

# Images as Tensors

Before a neural network can "see" a picture, that picture has to become **numbers**. A computer has no eyes. It has memory full of numbers. So the first thing you must understand in computer vision is this:

> An image is just a big grid of numbers.

Once you truly believe that — once "cat photo" and "a 3D array of integers" feel like the same thing in your head — everything else in this track becomes easier. YOLO, ViT, CLIP: they are all just clever math applied to that grid of numbers.

This lesson is the foundation. We go slowly.

---

## What is a digital image?

Take your phone and zoom into a photo as far as it will go. Eventually you stop seeing a face or a tree and start seeing little colored squares. Those squares are **pixels** (short for "picture elements").

A digital image is a rectangular grid of pixels. Two numbers describe the shape of that grid:

- **Width (W)** — how many pixels across (left to right)
- **Height (H)** — how many pixels down (top to bottom)

A "1920×1080" image has 1920 pixels across and 1080 pixels down. That is `1920 * 1080 = 2,073,600` pixels — about 2 megapixels.

Each pixel stores a **color**. How a color becomes a number is the next piece.

---

## Grayscale: one number per pixel

The simplest kind of image is **grayscale** (black and white). Each pixel is a single number describing how bright it is:

- `0` = pure black
- `255` = pure white
- everything in between = a shade of gray

Why 255 and not 100 or 1000? Because each pixel value is usually stored in **one byte** (8 bits). One byte can hold $2^8 = 256$ different values, from `0` to `255`. This is the famous `uint8` format — an unsigned 8-bit integer.

A grayscale image is therefore a **2D grid** of numbers. We can write it as a matrix:

```
[[  0,  50, 100],
 [150, 200, 255],
 [ 30,  90, 180]]
```

That is a tiny 3×3 grayscale image. Top-left pixel is black (`0`), the pixel next to `200` is white (`255`).

In array terms its **shape** is `(3, 3)` — meaning `(height, width)`. Notice: height comes first. This trips people up constantly. In image arrays, **rows come before columns**, so it is always `(H, W)`, not `(W, H)`.

---

## Color: three numbers per pixel (RGB)

Most images are in **color**. The standard way to store color is **RGB**: every pixel gets **three** numbers.

- **R** = how much Red (0–255)
- **G** = how much Green (0–255)
- **B** = how much Blue (0–255)

Mixing these three primary colors of light produces every color you see on a screen.

| R | G | B | Color |
|---|---|---|---|
| 255 | 0 | 0 | Pure red |
| 0 | 255 | 0 | Pure green |
| 0 | 0 | 255 | Pure blue |
| 255 | 255 | 0 | Yellow |
| 255 | 255 | 255 | White |
| 0 | 0 | 0 | Black |
| 128 | 128 | 128 | Gray |

So each pixel is a little list of 3 numbers. We call these three the **channels** of the image. RGB has **C = 3** channels.

---

## The 3D tensor: (H, W, C)

Now put it together. A color image is:

- `H` rows
- `W` columns
- and at each position, `C = 3` channel values

That is a **3-dimensional array** — a **tensor** with shape `(H, W, C)`. In deep learning we say "tensor" instead of "3D array," but it means the same thing: a grid of numbers with more than two dimensions.

Picture it as **three stacked sheets**:

```
      Red sheet          Green sheet         Blue sheet
   [[ r r r r ]        [[ g g g g ]        [[ b b b b ]
    [ r r r r ]         [ g g g g ]         [ b b b b ]
    [ r r r r ]]        [ g g g g ]]        [ b b b b ]]
```

To read the color of the pixel at row `y`, column `x`, you look at the same `(y, x)` spot on all three sheets and collect `(R, G, B)`.

A real photo of shape `(1080, 1920, 3)` therefore contains `1080 * 1920 * 3 = 6,220,800` numbers. That is what the neural network actually receives. Not "a cat" — six million numbers.

---

## Loading an image with PIL (Pillow)

**Pillow** (imported as `PIL`) is the most common Python library for opening and saving images. Combined with NumPy, it is the standard way to turn an image file into a tensor.

```python
from PIL import Image
import numpy as np

# Open an image file
img = Image.open("cat.jpg")

print(img.size)   # (width, height)  -- PIL uses (W, H) order!
print(img.mode)   # "RGB"

# Convert to a NumPy array
arr = np.array(img)
print(arr.shape)  # (height, width, 3) -- NumPy uses (H, W, C) order!
print(arr.dtype)  # uint8
```

> **Watch out:** `PIL.Image.size` returns `(width, height)`, but the moment you call `np.array(img)` you get shape `(height, width, channels)`. The order flips. This inconsistency has cost people hours of debugging. When in doubt, print the shape.

To save an array back to a file:

```python
out = Image.fromarray(arr)   # arr must be uint8 (H, W, C)
out.save("output.png")
```

---

## Working with the pixels in NumPy

Once the image is a NumPy array, you can index and slice it like any array.

```python
import numpy as np

# Pretend this is a loaded RGB image, shape (H, W, 3)
arr = np.random.randint(0, 256, size=(4, 5, 3), dtype=np.uint8)

# The pixel at row 2, column 3 -> its (R, G, B)
print(arr[2, 3])        # e.g. [123  8  240]

# The entire RED channel (a 2D grid)
red = arr[:, :, 0]
print(red.shape)        # (4, 5)

# The top-left 2x2 patch, all channels
patch = arr[0:2, 0:2, :]
print(patch.shape)      # (2, 2, 3)
```

Slicing rules to remember:

- `arr[y, x]` → one pixel (a length-3 vector for RGB)
- `arr[:, :, 0]` → the whole red channel
- `arr[:, :, 1]` → green, `arr[:, :, 2]` → blue
- `arr[a:b, c:d]` → a rectangular crop

Cropping an image is literally slicing rows and columns. Flipping an image horizontally is `arr[:, ::-1]`. Everything you learned about NumPy arrays applies directly.

---

## Pixel values 0–255 and normalization to 0–1

Raw pixels are integers from `0` to `255` (`uint8`). Neural networks do **not** like these large integer values. They prefer small floating-point numbers, usually centered near zero. Large inputs cause big, unstable gradients and slow training.

So almost the first thing every CV pipeline does is **normalize**: convert pixels to `float32` and scale them down.

The simplest normalization divides by 255 to get the range `[0, 1]`:

```python
arr = np.array(img)              # uint8, range 0..255
arr_float = arr.astype(np.float32) / 255.0
print(arr_float.min(), arr_float.max())   # 0.0 ... 1.0
```

$$x_{\text{norm}} = \frac{x}{255}$$

A stronger version — used by almost every pretrained model — is **standardization** with a mean and standard deviation per channel:

$$x_{\text{std}} = \frac{x_{\text{norm}} - \mu}{\sigma}$$

We cover the exact ImageNet numbers ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) in the Data Augmentation lesson. For now, just remember the sequence:

> **uint8 [0,255] → float32 [0,1] → standardized (roughly [-2, 2])**

---

## Image statistics: mean and std per channel

You will frequently compute the **mean** and **standard deviation** of an image, per channel. This tells you the overall brightness and contrast of each color, and it is exactly what you need to build your own normalization constants for a dataset.

```python
import numpy as np

arr = np.array(img).astype(np.float32) / 255.0   # (H, W, 3), range 0..1

# Mean over height and width, keeping the channel dimension
mean = arr.mean(axis=(0, 1))   # -> array of 3 numbers, one per channel
std  = arr.std(axis=(0, 1))    # -> 3 numbers

print("per-channel mean:", mean)   # e.g. [0.48 0.46 0.41]
print("per-channel std: ", std)    # e.g. [0.23 0.22 0.22]
```

The key is `axis=(0, 1)`. We average over **rows and columns** (all the spatial positions) but **not** over channels — we want a separate statistic for red, green, and blue.

To normalize a whole dataset properly, you compute the mean and std over **all** the training images combined, then subtract and divide. Every image gets the same treatment.

---

## Why PyTorch uses (C, H, W) instead of (H, W, C)

Here is the biggest source of confusion for beginners. NumPy and PIL give you images in `(H, W, C)` order — channels **last**. But PyTorch wants images in `(C, H, W)` order — channels **first**.

```
NumPy / PIL:  (Height, Width, Channels)   e.g. (224, 224, 3)
PyTorch:      (Channels, Height, Width)   e.g. (3, 224, 224)
```

Why does PyTorch flip it? Two practical reasons:

1. **Convolutions process one channel-map at a time.** Putting channels first means each channel is a contiguous 2D block of memory, which is faster for the operations CNNs perform.
2. **Consistency with how filters are stored.** Convolution weights are shaped `(out_channels, in_channels, kH, kW)`, so channel-first data lines up naturally.

You convert between the two orders with a **transpose** (called `permute` in PyTorch):

```python
import numpy as np

# NumPy array from an image: (H, W, C)
hwc = np.zeros((224, 224, 3), dtype=np.float32)

# Move channels to the front: (C, H, W)
chw = np.transpose(hwc, (2, 0, 1))
print(chw.shape)   # (3, 224, 224)

# ...and back again:
back = np.transpose(chw, (1, 2, 0))
print(back.shape)  # (224, 224, 3)
```

In PyTorch the same move is `tensor.permute(2, 0, 1)`. And when you use `torchvision.transforms.ToTensor()`, it does **both** jobs at once: it divides by 255 **and** rearranges `(H, W, C)` → `(C, H, W)`. That single line hides two important conversions, which is why people get confused when their shapes or value ranges look wrong.

---

## The batch dimension: (N, C, H, W)

Neural networks almost never process one image at a time. They process a **batch** — a bunch of images stacked together — because GPUs are fast at doing the same operation on many things at once.

Stacking `N` images, each `(C, H, W)`, gives a **4D tensor**:

$$(N, C, H, W)$$

- `N` = batch size (number of images), e.g. 32
- `C` = channels, e.g. 3
- `H` = height, e.g. 224
- `W` = width, e.g. 224

A batch of 32 RGB images at 224×224 has shape `(32, 3, 224, 224)` — that is `32 * 3 * 224 * 224 = 4,816,896` numbers flowing through the network in one step. When you read a PyTorch error like `expected input [32, 3, 224, 224]`, now you know exactly what each number means.

> **Mnemonic: N-C-H-W.** "Number, Channels, Height, Width." Say it out loud a few times. You will read it in error messages for the rest of your CV career.

---

## Common pitfalls (read this twice)

These bugs show up in nearly every beginner's competition notebook. Memorize them now and save yourself hours later.

**1. uint8 vs float32.** If you divide a `uint8` array by 255 without converting to float first, integer math can silently give you zeros. Always `astype(np.float32)` **before** dividing.

```python
bad  = arr / 255            # works in NumPy (auto-floats), but be explicit
good = arr.astype(np.float32) / 255.0
```

Worse: if you feed a float image (range 0–1) to a function expecting `uint8`, or try to save a float array with `Image.fromarray`, you get garbage or an error. `Image.fromarray` needs `uint8`.

**2. HWC vs CHW.** Feeding an `(H, W, C)` array to a PyTorch model that expects `(C, H, W)` produces a shape error or, worse, treats width as channels. If a model complains about channel counts, check your axis order first.

**3. Forgetting the batch dimension.** A single image `(3, 224, 224)` will error in most models. They want `(1, 3, 224, 224)`. Add a batch axis with `arr[None, ...]` (NumPy) or `tensor.unsqueeze(0)` (PyTorch).

**4. RGB vs BGR.** Pillow gives RGB. OpenCV (`cv2`) gives **BGR** — red and blue swapped! If your image looks blue-tinted, this is almost always why. Convert with `img[:, :, ::-1]`.

**5. Values out of range.** After you normalize, do not accidentally clip or re-scale. And when you want to *display* a normalized image, you must **un-normalize** it back to `[0, 255]` uint8 first, or it will look wrong.

---

## Putting it all together

Here is the canonical "image to model-ready tensor" pipeline. Read each line and say what shape and dtype you have after it.

```python
from PIL import Image
import numpy as np

img = Image.open("cat.jpg").convert("RGB")  # force 3 channels
img = img.resize((224, 224))                # (W, H) for PIL!

arr = np.array(img)                         # (224, 224, 3), uint8
arr = arr.astype(np.float32) / 255.0        # (224, 224, 3), float, [0,1]

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
arr = (arr - mean) / std                    # standardized

arr = np.transpose(arr, (2, 0, 1))          # (3, 224, 224)  CHW
arr = arr[None, ...]                        # (1, 3, 224, 224) add batch
print(arr.shape)                            # ready for the model
```

Every competition CV notebook you will ever write starts with some version of these eight lines. Understand them and you are ready for the rest of the track.

---

## Summary

- A digital image is a grid of pixels; each pixel is one or more numbers.
- **Grayscale** = one number per pixel, shape `(H, W)`. **RGB** = three numbers (channels), shape `(H, W, C)`.
- Height comes **before** width in array shapes: `(H, W)`, always.
- Pixel values are `uint8` in `[0, 255]`. Networks want `float32`, normalized to `[0, 1]` then standardized with per-channel mean/std.
- Compute per-channel stats with `arr.mean(axis=(0, 1))` and `arr.std(axis=(0, 1))`.
- **PIL/NumPy use `(H, W, C)`; PyTorch uses `(C, H, W)`.** A batch is `(N, C, H, W)` — remember **N-C-H-W**.
- Top pitfalls: uint8 vs float32, HWC vs CHW, missing batch dim, RGB vs BGR.
