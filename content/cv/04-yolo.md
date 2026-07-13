---
title: Object Detection with YOLO
track: cv
order: 4
estimatedTime: 55
difficulty: advanced
---

# Object Detection with YOLO

So far our models answer one question about a whole image: "What is this?" But many real tasks need more. *How many* chickens are in this photo? *Where* is each one? That is **object detection**, and at IOAI 2025 the **Chicken Counting** task was exactly this problem.

This lesson covers detection from the ground up: how it differs from classification, how boxes are described, the two metrics you must know (IoU and mAP), the cleanup step (NMS), and how to actually run **YOLOv8** with the `ultralytics` library.

---

## Classification vs Detection vs Segmentation

Three levels of "understanding" an image, in increasing detail:

| Task | Question | Output |
|---|---|---|
| **Classification** | What is in this image? | One label for the whole image ("chicken") |
| **Detection** | What is where, and how many? | A list of **boxes**, each with a label + confidence |
| **Segmentation** | Which exact pixels belong to each object? | A per-pixel mask |

- **Classification** gives one answer per image. A photo of five chickens → the label "chicken." It cannot count.
- **Detection** draws a **bounding box** around every object and labels each one. Five chickens → five boxes. Now you can count, locate, and measure.
- **Segmentation** is the most detailed: it outlines objects pixel-by-pixel. More precise, more expensive.

For "count the chickens" or "find every object," you need **detection**. That is our focus.

---

## Bounding boxes

A **bounding box** is the rectangle that tightly encloses an object. There are two common ways to write one down.

**Corner format (xyxy):** the top-left and bottom-right pixel coordinates.

$$(x_1, y_1, x_2, y_2)$$

**Center format (YOLO format):** the center point plus width and height.

$$(x_{\text{center}}, y_{\text{center}}, w, h)$$

YOLO uses the center format, and — importantly — it stores the numbers **normalized** to the range `[0, 1]` by dividing by the image width and height:

$$x_{\text{center}}^{\text{norm}} = \frac{x_{\text{center}}}{W}, \quad w^{\text{norm}} = \frac{w}{W}$$

Normalizing makes the labels independent of image resolution — a box in the center of the image is `(0.5, 0.5, ...)` whether the image is 400px or 4000px wide.

A YOLO **label file** is a plain `.txt` with one line per object:

```
class_id  x_center  y_center  width  height
0         0.51      0.42      0.18   0.25
0         0.73      0.66      0.15   0.22
```

Each line: the class index, then the four normalized box numbers. Two lines here = two objects of class 0. Converting between corner and center formats is simple arithmetic, and getting it right is a frequent source of bugs.

```python
def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h
```

---

## IoU: Intersection over Union

How do we measure whether a predicted box is "correct"? A prediction almost never matches the true box pixel-perfectly. We need a score for *how much they overlap*. That score is **IoU**.

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{area of overlap}}{\text{area of union}}$$

- `A` = predicted box, `B` = ground-truth box.
- **Intersection** = the overlapping area (where both boxes cover).
- **Union** = the total area covered by either box.
- IoU ranges from **0** (no overlap) to **1** (perfect match).

Intuition: IoU asks "of all the area these two boxes touch, what fraction do they *share*?" Two boxes that overlap a lot but also stick out a lot get a middling score; two nearly-identical boxes score near 1.

A prediction is usually counted as **correct** if its IoU with a true box exceeds a threshold, commonly **0.5**. Higher thresholds (0.75) demand tighter boxes.

```python
def iou(boxA, boxB):
    # boxes in xyxy pixel coords
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0
```

The `max(0, ...)` handles the case where the boxes do not overlap at all (negative width/height → clamp to 0).

---

## NMS: Non-Maximum Suppression

Detectors are enthusiastic. For a single chicken, the model often predicts **several overlapping boxes**, all pointing at the same bird with slightly different positions and confidences. If you kept them all, you would count one chicken as five. You need to collapse each cluster of duplicates down to a single box.

**Non-Maximum Suppression (NMS)** does exactly that:

1. Take the box with the **highest** confidence. Keep it.
2. Remove every other box that overlaps it too much (IoU above a threshold, e.g. 0.5) — these are duplicates of the same object.
3. From the boxes that remain, take the next highest-confidence one. Keep it.
4. Repeat until no boxes are left.

```python
def nms(boxes, scores, iou_thresh=0.5):
    # sort indices by score, highest first
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) < iou_thresh]
    return keep
```

After NMS, each real object has exactly one box. This is why the `iou_thresh` for NMS matters: too high and you keep duplicates (over-count); too low and you delete genuinely separate, nearby objects (under-count). For crowded scenes like a flock of chickens, this threshold is a real knob to tune.

---

## YOLO architecture overview

**YOLO** stands for "You Only Look Once." The name captures its key idea: unlike older detectors that scanned an image with thousands of region proposals, YOLO looks at the **whole image once** and predicts all boxes in a single forward pass. That is why it is fast enough for real-time video — and convenient in a competition.

The core mechanics (you do not need every detail, but the shape of it helps):

- The image is divided into a **grid** of cells.
- Each cell is responsible for predicting objects whose center falls inside it. For each cell the model outputs candidate boxes: position, size, an **objectness** (confidence there is an object), and class scores.
- Historically, cells predicted boxes relative to a few preset **anchor boxes** — templates of common object shapes (tall, wide, square). The model learns offsets from the nearest anchor rather than raw coordinates, which is easier to learn. (Modern YOLO versions like YOLOv8 are *anchor-free*, predicting box centers directly, but the grid idea remains.)
- All those raw predictions then pass through **NMS** to remove duplicates, leaving the final clean list of detections.

```
image → YOLO network → grid of raw box predictions → NMS → final boxes
```

You rarely implement this yourself at a competition. You use a library — and the library of choice is `ultralytics`.

---

## YOLOv8 with the ultralytics library

The `ultralytics` package makes training and running YOLO astonishingly short. This is what you would actually write for Chicken Counting.

### Inference with a pretrained model (zero training)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")     # 'n' = nano (smallest/fastest); s, m, l, x are bigger
results = model("photo.jpg")   # run detection on one image

for r in results:
    boxes = r.boxes                    # detected boxes
    print("num objects:", len(boxes))  # <- this is your count!
    for b in boxes:
        print(b.cls, b.conf, b.xyxy)   # class, confidence, coords
```

The pretrained COCO model already knows 80 everyday categories (including "bird"). For a rough baseline you might get a chicken count with **no training at all** — a perfect first submission to beat.

### Training / fine-tuning on your own data

To specialize the model, you point it at a small **dataset YAML** describing your images and classes, and call `.train()`.

```yaml
# chickens.yaml
path: ./chicken_data
train: images/train
val: images/val
names:
  0: chicken
```

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")        # start from pretrained weights (transfer learning!)
model.train(
    data="chickens.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)

metrics = model.val()             # evaluate on the val split
print(metrics.box.map)            # mAP@[.5:.95]
print(metrics.box.map50)          # mAP@0.5
```

Notice we start from `yolov8n.pt` (pretrained) rather than random weights — the transfer-learning principle from the last lesson applies to detection too.

---

## Training vs inference workflow

Keep these two phases straight:

**Training** (you have labeled boxes):
1. Prepare images + YOLO-format label `.txt` files.
2. Write the dataset YAML (paths + class names).
3. `model.train(...)` starting from pretrained weights.
4. `model.val(...)` to read mAP on the validation split.
5. Adjust (augmentation, epochs, image size, model size) and repeat.

**Inference** (you want predictions on new images):
1. Load the trained weights.
2. Run `model(image)`.
3. Read `results[i].boxes` — the model already applied NMS for you.
4. Count / locate / post-process as the task requires (for counting: `len(boxes)`, maybe after filtering by a confidence threshold).

> **Counting tip:** the confidence threshold hugely affects the count. Too low and you count noise as chickens; too high and you miss real ones. Tune this threshold on your validation set against the true counts.

---

## mAP: the detection metric

Accuracy does not make sense for detection — there is no single label. The standard metric is **mAP (mean Average Precision)**.

Building it up:

- For a given class, sort predictions by confidence and, using an IoU threshold (say 0.5), mark each as a true positive (matches a real box) or false positive (duplicate/wrong). From this you compute **precision** (of the boxes I predicted, how many were right?) and **recall** (of the real objects, how many did I find?).
- Sweeping the confidence threshold traces a **precision-recall curve**. The area under it is the **Average Precision (AP)** for that class.
- **mAP** is the AP averaged over all classes.

Two common variants you will see in `ultralytics` output:

- **mAP@0.5** (`map50`): counts a detection correct if IoU > 0.5. More forgiving.
- **mAP@[.5:.95]** (`map`): averages mAP over IoU thresholds from 0.5 to 0.95 in steps of 0.05. Stricter — rewards tight boxes. This is the primary COCO metric.

> The exact competition metric may differ (Chicken Counting cares about the **count**, so an error like mean absolute count error can matter more than mAP). **Always read the task's stated metric** and optimize *that*, not whatever the library prints by default.

---

## IOAI 2025 connection: Chicken Counting

The **Chicken Counting** task at IOAI 2025 (Beijing) asked competitors to count chickens in images — a classic detection-and-count problem. The winning pattern was the detection playbook in this lesson:

1. **Start from a pretrained YOLO** (transfer learning) rather than training from scratch — small data, short time.
2. **Fine-tune** on the provided chicken images with sensible **augmentation** (careful with flips/rotations if orientation matters, gentle otherwise).
3. **Tune the confidence and NMS thresholds** so the *count* — not just the boxes — matches the validation ground truth. In crowded, overlapping flocks, NMS settings make or break the count.
4. **Optimize the actual scoring metric** (count error), not the default mAP.

The broader lesson: detection at IOAI is rarely about inventing architectures. It is about wielding a strong pretrained detector well — correct box formats, sensible augmentation, and careful post-processing (NMS + confidence thresholds) tuned against the real metric.

---

## Common pitfalls

- **Box format mix-ups.** xyxy vs xywh vs normalized YOLO format. A silent format bug makes every box wrong. Print a box on top of an image to sanity-check.
- **Forgetting NMS / wrong NMS threshold.** Duplicates inflate counts; overly aggressive NMS merges distinct objects. Tune it.
- **Confidence threshold not tuned.** The single biggest lever for counting accuracy.
- **Training from scratch.** Always start from pretrained weights.
- **Optimizing mAP when the metric is count error** (or vice versa). Read the metric first.

---

## Summary

- **Detection** finds *what is where* — a labeled bounding box per object — unlike classification (one label) or segmentation (per-pixel masks). Counting objects requires detection.
- **YOLO boxes** use center format, normalized: `(class, x_center, y_center, w, h)` in `[0, 1]`, one line per object in a `.txt` label file.
- **IoU** = overlap / union, from 0 to 1; a prediction is "correct" when IoU beats a threshold (often 0.5).
- **NMS** removes duplicate overlapping boxes, keeping the highest-confidence one per object — essential for correct counts.
- **YOLO** looks at the whole image once via a grid; use the **`ultralytics`** library: `YOLO("yolov8n.pt")`, `.train(data=...)`, `.val()`, or just call the model for inference (NMS is applied for you).
- **mAP** (mAP@0.5 and mAP@[.5:.95]) is the standard detection metric — but always optimize the *task's actual* metric.
- **IOAI 2025 Chicken Counting** = pretrained YOLO + fine-tune + carefully tuned confidence/NMS thresholds, optimized for count error.
