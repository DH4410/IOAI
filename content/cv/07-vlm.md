---
title: Vision-Language Models (VLMs)
track: cv
order: 7
estimatedTime: 60
difficulty: advanced
---

# Vision-Language Models (VLMs)

Vision-Language Models (VLMs) are AI systems that understand both images and text — and can reason about their relationship. They can answer questions about images, describe what they see, find objects described in text, and even generate images from descriptions.

VLMs represent one of the most exciting frontiers in AI right now. Understanding them is increasingly important for IOAI competitions.

---

## 1. What Is a VLM?

A Vision-Language Model is a neural network trained to process and relate information from two modalities:

1. **Vision** — images, video frames, diagrams
2. **Language** — text descriptions, questions, captions, instructions

The core challenge is **alignment**: images and text are fundamentally different data types. Pixels and words live in completely different spaces. VLMs learn to bridge this gap.

### What VLMs Can Do

| Task | Input | Output |
|------|-------|--------|
| Image captioning | Image | Text description |
| Visual QA (VQA) | Image + question | Answer |
| Zero-shot classification | Image + class names | Class label |
| Visual grounding | Image + description | Bounding box |
| Image-text retrieval | Query text | Matching images |
| Document understanding | Document image | Extracted text/answers |
| Multimodal reasoning | Image + complex question | Reasoning + answer |

### Why VLMs Are Powerful

Before VLMs, you needed a separate model for each visual task: one model for classification, another for captioning, another for VQA. A VLM is a single model that handles all of these — and can generalise to new tasks it was not explicitly trained on.

**Zero-shot ability** is the key superpower. Give a CLIP model the text "a photo of a tabby cat" and an image — it can tell you how well they match, even if it never saw that exact phrase during training. This is zero-shot because you never trained it on that specific prompt.

---

## 2. Building Blocks of a VLM

Every VLM has at least two components that must be connected:

### Vision Encoder

Processes the image and produces a vector representation (embedding).

- Classical choice: **Convolutional Neural Network** (ResNet, EfficientNet)
- Modern choice: **Vision Transformer (ViT)** — treats image patches as tokens

We covered ViT in an earlier lesson. To recap:
- The image is divided into fixed-size patches (e.g. 16×16 pixels)
- Each patch is linearly projected to a vector
- These vectors are fed to a standard Transformer encoder
- The output is a sequence of patch embeddings + a [CLS] token embedding

The [CLS] token embedding is used as the global image representation.

### Text Encoder / Language Model

Processes the text and produces a vector or sequence of vectors.

- **Text encoder**: bidirectional transformer (like BERT) — good for understanding
- **Language model decoder**: autoregressive transformer (like GPT) — good for generation

The vision encoder output must be connected to the language component. How they are connected defines the architecture.

### The Alignment Problem

The fundamental challenge: a vector of 512 numbers from a vision encoder means something completely different from a vector of 512 numbers from a text encoder — unless they were trained together to share the same space.

Different VLMs solve this differently:
- **CLIP**: contrastive training on image-text pairs
- **Flamingo**: cross-attention from text to image tokens
- **LLaVA**: a learned MLP projector

---

## 3. CLIP: The Foundation

CLIP (Contrastive Language-Image Pre-Training, OpenAI 2021) is the most influential VLM foundation. Many modern VLMs build on CLIP's ideas.

### Contrastive Pretraining

CLIP is trained on 400 million (image, text) pairs collected from the internet. For each pair, the text is the alt text or caption associated with the image.

The training objective: **make matching pairs close, and non-matching pairs far apart.**

**Architecture:**
- An **image encoder** (ViT or ResNet) maps images to embedding vectors: `I = image_encoder(image)` → shape (D,)
- A **text encoder** (Transformer) maps text to embedding vectors: `T = text_encoder(text)` → shape (D,)
- Both are normalised to unit length: `I = I / ||I||`, `T = T / ||T||`

**Training a batch of N pairs:**

```
Batch: N images, N captions (matched)

Compute N image embeddings: I1, I2, ..., IN
Compute N text embeddings:  T1, T2, ..., TN

Build N×N similarity matrix:
    S[i][j] = dot(I_i, T_j)   (= cosine similarity since vectors are normalised)

Diagonal entries S[i][i] → matching pairs (should be HIGH)
Off-diagonal S[i][j] (i≠j) → non-matching pairs (should be LOW)

Loss: cross-entropy on rows (each image should match its text)
    + cross-entropy on columns (each text should match its image)
```

```python
# Conceptual implementation (not executable without CLIP/numpy only shown)
import numpy as np

def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    image_embeddings: (N, D) numpy array, L2-normalised
    text_embeddings:  (N, D) numpy array, L2-normalised
    """
    N = image_embeddings.shape[0]

    # Similarity matrix: S[i,j] = cosine_sim(image_i, text_j)
    S = image_embeddings @ text_embeddings.T  # (N, N)
    S = S / temperature   # scale for sharper distribution

    # Labels: diagonal is correct (image_i matches text_i)
    labels = np.arange(N)

    # Cross-entropy loss in both directions (here shown conceptually)
    # loss_i2t = cross_entropy(S[i], label=i) for each image row
    # loss_t2i = cross_entropy(S[:,j], label=j) for each text column

    return S, labels   # actual CE loss requires softmax + log

# Example with 3 pairs
image_embs = np.array([[1.0, 0.0, 0.0],   # image 0: "cat"
                         [0.0, 1.0, 0.0],   # image 1: "dog"
                         [0.0, 0.0, 1.0]])  # image 2: "bird"

text_embs  = np.array([[0.9, 0.1, 0.0],   # text 0: "a photo of a cat"
                         [0.0, 0.9, 0.1],   # text 1: "a photo of a dog"
                         [0.1, 0.0, 0.9]])  # text 2: "a photo of a bird"

S, labels = clip_loss(image_embs, text_embs)
print("Similarity matrix:")
print(S.round(2))
print("Diagonal (matching pairs) should be highest in each row:")
print(S.diagonal().round(2))
```

### The Shared Embedding Space

After CLIP training, images and texts that are semantically related end up near each other in the shared vector space:
- An image of a cat and the text "a photo of a cat" will have high cosine similarity.
- An image of a cat and the text "a photo of an airplane" will have low cosine similarity.

This shared space makes CLIP extremely versatile.

### Zero-Shot Classification with CLIP

You can classify any image into any set of categories — without any task-specific training:

```python
# Pseudocode — requires the 'clip' package or HuggingFace
# pip install git+https://github.com/openai/CLIP.git

import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Your image
image = preprocess(Image.open("my_image.jpg")).unsqueeze(0).to(device)

# Your candidate class names — ANY set you want
class_names = ["a cat", "a dog", "an airplane", "a car", "a flower"]
text_inputs  = clip.tokenize([f"a photo of {c}" for c in class_names]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)     # (1, 512)
    text_features  = model.encode_text(text_inputs)  # (5, 512)

    # Normalise
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    # Similarity scores
    similarities = (image_features @ text_features.T).squeeze()  # (5,)
    probs = similarities.softmax(dim=0)

for name, prob in zip(class_names, probs):
    print(f"{name:20s}: {prob.item()*100:.1f}%")
```

This is zero-shot because the class names were never in the training loop for this specific task.

### Limitations of CLIP

- Cannot generate text (it is an encoder, not a decoder)
- Poor at detailed visual reasoning ("is the red ball to the LEFT or RIGHT of the blue cube?")
- Struggles with counting, OCR, fine-grained spatial relationships
- Biased by the internet text data it was trained on

---

## 4. Flamingo: Interleaved Vision and Language

Flamingo (DeepMind, 2022) was a major step toward a general-purpose multimodal model. It can handle interleaved sequences of images and text:

```
Input: "Here is an image of a dog [IMAGE_1]. Here is another image [IMAGE_2].
        What is the difference between these two animals?"
Output: "The first image shows a golden retriever, while the second shows a German Shepherd.
         The golden retriever has a lighter, golden coat..."
```

### Flamingo Architecture

**Vision Encoder**: A pretrained CLIP-like visual encoder (frozen during Flamingo training). Outputs a sequence of visual feature vectors.

**Perceiver Resampler**: Because different images have different numbers of visual tokens, Flamingo uses a Perceiver module with a fixed set of learned query vectors that "attend" to the visual features and output a fixed-size representation regardless of image size.

**Cross-Attention Layers**: Injected into the language model at regular intervals. These layers allow the language model to attend to the image features when generating each text token.

**Frozen Language Model**: Flamingo keeps a large pretrained language model (Chinchilla, ~70B parameters) frozen, and only trains the newly added cross-attention layers and the Perceiver.

### Key Insights from Flamingo

1. **Few-shot in-context learning**: Like GPT-3 for language, you can give Flamingo a few examples directly in the prompt:
   ```
   [IMAGE_1] Caption: "A yellow lab playing in the park."
   [IMAGE_2] Caption: "A husky running through snow."
   [IMAGE_3] Caption: ???
   ```
   Flamingo generates an appropriate caption for IMAGE_3.

2. **Freeze the good parts**: By freezing both the vision encoder and language model, Flamingo avoids catastrophic forgetting of language ability. Only the bridge layers are trained.

3. **Scale wins**: Larger language models → better multimodal performance.

---

## 5. LLaVA: Visual Instruction Tuning

LLaVA (Large Language and Vision Assistant, 2023) made building capable VLMs accessible. The key insight: **you don't need complex architecture — just a clever training recipe and a simple connector.**

### LLaVA Architecture

```
Image → CLIP Vision Encoder → [Visual Tokens]
                                     ↓
                              Linear Projection (MLP)
                                     ↓
[System Prompt] → LLM Tokenizer → [Text Tokens]
                                     ↓
                              Concatenate
                                     ↓
                          Large Language Model (LLaMA/Vicuna)
                                     ↓
                               Generated Text
```

The **projector** is just a small MLP (1-2 layers) that transforms visual token embeddings from CLIP's space into the LLM's token embedding space. The LLM then sees visual tokens mixed with text tokens and processes everything with standard self-attention.

### Training LLaVA

**Stage 1 — Feature alignment**: Freeze the vision encoder and LLM. Train only the projector MLP on image-caption pairs. The goal: get visual tokens to "look like" word embeddings to the LLM.

**Stage 2 — Visual instruction tuning**: Unfreeze the LLM. Train on instruction-following data:
```
Human: [image of a graph] What is the trend in CO2 emissions since 1990?
Assistant: The graph shows CO2 emissions increasing steadily from 1990...
```
This teaches the model to follow visual instructions, not just describe images.

### LLaVA-1.5 Improvements

- Replace linear projector with 2-layer MLP
- Add academic task data (VQA, OCR, charts)
- Results comparable to much larger models

### Using LLaVA with HuggingFace

```python
# Pseudocode — requires: pip install transformers
# These models are multi-GB, run locally with GPU

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load an image
image = Image.open("my_image.jpg")

# Create a conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do you see in this image? Describe it in detail."},
        ],
    },
]

# Format and generate
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)
    
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## 6. GPT-4V and Gemini: Closed-Source VLMs

OpenAI's GPT-4V and Google's Gemini are the most capable VLMs as of 2024-2025. They are accessible only via API.

### GPT-4V Capabilities

```python
# Pseudocode — requires OpenAI API key
# pip install openai

from openai import OpenAI
import base64

client = OpenAI()

# Encode image as base64
with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": "Analyse this chart. What is the main trend and what was the peak value?"
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

GPT-4V is particularly strong at:
- Understanding complex diagrams, charts, and graphs
- Reading text in images (OCR)
- Multi-step visual reasoning
- Understanding spatial relationships

### Gemini

Google's Gemini (multimodal from ground up, not added later) processes text, images, audio, and video natively. Particularly strong at:
- Document understanding
- Very long contexts with mixed modalities
- Scientific and mathematical visual reasoning

For IOAI, these closed-source models may be available via API budget. Always try them first for non-code tasks.

---

## 7. Visual Question Answering (VQA)

VQA is the task of answering natural language questions about images.

### Example Questions (from VQA v2 dataset)

```
Image: [photo of a kitchen]
Q: "What colour is the refrigerator?"
A: "White"

Image: [photo of a road at night]
Q: "Is it daytime or nighttime?"
A: "Nighttime"

Image: [photo of a birthday cake with candles]
Q: "How many candles are on the cake?"
A: "7"
```

### VQA Datasets

- **VQA v2** (Visual QA, balanced): 1.1M questions about 265K COCO images. The v2 version was designed to reduce language bias (the model can't just answer from common sense without looking at the image).
- **GQA**: Compositional questions with scene graph annotations.
- **OK-VQA**: Questions requiring outside knowledge ("What country is this landmark in?").
- **TextVQA**: Questions about text in images.

### VQA Metrics

**Accuracy** is the standard metric but with a twist: since different human annotators may give slightly different correct answers, VQA accuracy is:

```
score(predicted_answer) = min(count_of_humans_who_said_that / 3, 1.0)
```

If 10 annotators answered the question and 9 said "white", answering "white" gives score 1.0. Answering "off-white" might give 0.67 if 2 annotators said it.

### Approach to VQA

Modern approach: treat VQA as text generation.
- Input: image + question
- Output: generated text answer
- Loss: cross-entropy on the generated tokens

Older approach: treat it as classification over a fixed vocabulary of ~3000 most common answers. Works well for closed-answer questions but cannot handle novel answers.

---

## 8. Image Captioning

Image captioning generates a natural language description of an image.

### Architecture: Encoder-Decoder

```
Image → Vision Encoder → Fixed-size embedding
                                ↓
                    LSTM or Transformer decoder
                         ↓     ↓     ↓     ↓
                        "A"  "dog"  "is"  "running" ...
```

Modern approach: treat captioning as a sequence generation problem with a visual prefix.

```python
# Using HuggingFace BLIP model for captioning
# pip install transformers

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("my_image.jpg").convert("RGB")

# Unconditional captioning
inputs = processor(image, return_tensors="pt").to(model.device, torch.float16)
out    = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))

# Conditional captioning (guided by a prompt)
inputs = processor(image, "a photo of", return_tensors="pt").to(model.device, torch.float16)
out    = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))
```

### Beam Search

Instead of greedy decoding (always pick the most probable next token), beam search maintains K candidate sequences (beams) at each step:

```
Step 1: Generate top-5 first tokens
         "A" (p=0.3), "The" (p=0.25), "An" (p=0.15), "Two" (0.1), "Several" (0.08)

Step 2: For each beam, generate top-5 continuations:
         "A dog" (0.3*0.4=0.12), "A cat" (0.3*0.35=0.105), ...
         "The red" (0.25*0.3=0.075), ...

Keep top-5 total sequences across all beams.
Continue until all beams hit <EOS> token.
Select highest total log-probability sequence.
```

Beam search typically gives better captions than greedy decoding.

### Captioning Metrics

- **BLEU** (Bilingual Evaluation Understudy): counts n-gram overlaps between generated and reference captions. Widely used but criticised for not capturing semantics.
- **CIDEr** (Consensus-based Image Description Evaluation): weights n-grams by their frequency in reference captions. Correlates better with human judgement.
- **METEOR**: considers synonyms and stemming. More linguistically aware.
- **CLIPScore**: uses CLIP to measure semantic alignment between caption and image — no reference needed!

---

## 9. Grounding: Text-to-Region Matching

**Visual grounding** (or phrase grounding) means finding the region in an image described by a text phrase.

Input: image + text description ("the red ball on the left")
Output: bounding box around that object

### Models

- **GLIP** (Grounded Language-Image Pre-training): unifies detection and grounding by treating both as phrase-region matching.
- **OWL-ViT**: Open-vocabulary detection. Detect any object described in text.
- **Grounding DINO**: Combines DINO (self-supervised vision) with grounded detection.

```python
# Using OWL-ViT from HuggingFace (zero-shot object detection)
# pip install transformers

from transformers import pipeline
from PIL import Image

detector = pipeline(
    model="google/owlvit-base-patch32",
    task="zero-shot-object-detection"
)

image = Image.open("my_image.jpg")
predictions = detector(
    image,
    candidate_labels=["a cat", "a dog", "a person", "a car"],
)

for pred in predictions:
    print(f"Label: {pred['label']:15s} | Score: {pred['score']:.2f} | "
          f"Box: {[round(v) for v in pred['box'].values()]}")
```

---

## 10. Practical Usage Patterns

### Pattern 1: Zero-Shot Classification with CLIP

Use when: you have many categories and no training data.

```python
# Reference pattern — run locally with PyTorch + transformers
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("my_image.jpg")
texts = ["a dog", "a cat", "a car", "a tree"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Logits: how well each text matches the image
logits = outputs.logits_per_image.squeeze()    # (4,)
probs  = logits.softmax(dim=0)

for text, prob in zip(texts, probs):
    print(f"{text:15s}: {prob.item()*100:.1f}%")
```

### Pattern 2: Image Embedding for Retrieval

Use when: you want to find similar images or retrieve images by text description.

```python
# Reference — run locally
from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def get_image_embedding(image):
    """Get CLIP image embedding, L2-normalised."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return (emb / emb.norm(dim=-1, keepdim=True)).squeeze().numpy()

def get_text_embedding(text):
    """Get CLIP text embedding, L2-normalised."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return (emb / emb.norm(dim=-1, keepdim=True)).squeeze().numpy()

# Build an image database
# image_db = [get_image_embedding(img) for img in images]  # (N, 512)

# Query by text
# query_emb = get_text_embedding("a golden retriever running")
# similarities = np.array(image_db) @ query_emb   # (N,) cosine similarities
# best_match = np.argmax(similarities)
```

### Pattern 3: Using a VLM for Structured Output

Ask the VLM to return JSON-formatted responses for downstream processing:

```python
# Pattern using GPT-4V or any instruction-following VLM

prompt = """
Analyse this image and respond with a JSON object containing:
{
  "main_objects": ["list", "of", "objects"],
  "scene_type": "indoor/outdoor/other",
  "dominant_colours": ["colour1", "colour2"],
  "estimated_time_of_day": "day/night/unclear",
  "confidence": 0.0 to 1.0
}
Only respond with valid JSON, no other text.
"""
```

---

## 11. Understanding Attention in VLMs

A key concept in VLMs is how attention connects vision and language.

### Self-Attention (within modality)

In the vision encoder (ViT), each image patch attends to every other patch. This lets the model integrate global context: a patch near a dog's nose can be informed by patches near its eyes, ears, and body.

### Cross-Attention (across modalities)

In models like Flamingo, the text tokens use cross-attention to query the image tokens:

```
Query: from text token at current position
Key, Value: from image patch embeddings

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

This allows each generated word to "look at" the most relevant image regions. When generating "retriever" in a caption, the attention might focus on the dog's fur colour and body shape.

### Visualising Attention

Research has shown that cross-attention in VLMs often produces interpretable patterns:
- The word "red" attends to red regions in the image
- The word "ball" attends to round objects
- Spatial words ("left", "behind") correspond to specific image regions

---

## 12. IOAI Relevance

### VLM Tasks in Recent Competitions

IOAI has included tasks such as:

1. **Image-text matching**: Given pairs, predict if an image and text are related (binary classification using cosine similarity).

2. **Zero-shot classification**: Classify images using CLIP without any labelled training data for the target classes.

3. **Feature extraction**: Use CLIP or DINO embeddings as input features for a downstream classifier. Train only a small head.

4. **VQA-style tasks**: Answer questions about images using a pretrained VLM via API.

5. **Embedding similarity**: Given a query image, rank a gallery by visual similarity.

### What the Judges Look For

- **Correct use of pretrained models**: Using the right model for the task (CLIP for classification/retrieval, BLIP for captioning, OWL-ViT for detection).

- **Proper normalisation**: CLIP embeddings should always be L2-normalised before computing cosine similarity.

- **Prompt engineering**: For zero-shot tasks, the phrasing of class names matters. "a photo of a {class}" typically outperforms just "{class}".

- **Ensemble strategies**: Averaging embeddings from multiple text templates improves CLIP accuracy significantly:
  ```python
  templates = [
      "a photo of a {}",
      "a close-up photo of a {}",
      "a picture of a {}",
      "a {} in the wild",
  ]
  # Average text embeddings across templates for each class
  ```

- **Efficient inference**: Use `torch.no_grad()` and batched inference. Don't compute embeddings one image at a time.

### Competition-Ready CLIP Code

```python
# Reference — run locally with PyTorch + transformers

import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from pathlib import Path

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ---- Text embeddings for all classes ----
CLASS_NAMES = ["cat", "dog", "car", "airplane", "flower"]
TEMPLATES   = [
    "a photo of a {}",
    "a photograph of a {}",
    "a picture of a {}",
]

@torch.no_grad()
def encode_classes(class_names, templates):
    """Compute averaged text embeddings for each class."""
    class_embeddings = []
    for cls in class_names:
        prompts = [t.format(cls) for t in templates]
        inputs  = processor(text=prompts, return_tensors="pt",
                            padding=True, truncation=True).to(device)
        embs    = model.get_text_features(**inputs)           # (T, D)
        embs    = embs / embs.norm(dim=-1, keepdim=True)
        class_embeddings.append(embs.mean(dim=0))             # average over templates
    stacked = torch.stack(class_embeddings)                   # (C, D)
    return stacked / stacked.norm(dim=-1, keepdim=True)       # renormalise

class_embs = encode_classes(CLASS_NAMES, TEMPLATES)

# ---- Image embeddings (batched) ----
@torch.no_grad()
def encode_images_batch(image_list, batch_size=32):
    """Encode a list of PIL Images in batches."""
    all_embs = []
    for i in range(0, len(image_list), batch_size):
        batch   = image_list[i:i+batch_size]
        inputs  = processor(images=batch, return_tensors="pt").to(device)
        embs    = model.get_image_features(**inputs)
        embs    = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)   # (N, D)

# ---- Zero-shot classification ----
# images = [Image.open(p) for p in image_paths]
# image_embs = encode_images_batch(images)                    # (N, D)
# logits = (image_embs @ class_embs.T) * 100                  # (N, C) — scale like CLIP paper
# predictions = logits.argmax(dim=1).cpu().numpy()            # (N,)
# predicted_classes = [CLASS_NAMES[i] for i in predictions]
```

---

## 13. Summary

| Model      | Year | Key Innovation                                    | Best For                    |
|------------|------|---------------------------------------------------|-----------------------------|
| CLIP       | 2021 | Contrastive image-text pretraining at scale       | Zero-shot classification, retrieval, embeddings |
| Flamingo   | 2022 | Cross-attention from LM to visual features; few-shot VLM | Interleaved image-text; in-context learning |
| BLIP/BLIP-2 | 2022/23 | Q-Former bridge between vision and LM | Captioning, VQA, retrieval |
| LLaVA      | 2023 | Visual instruction tuning with projector MLP     | General-purpose visual assistant |
| GPT-4V     | 2023 | Closed-source, strongest reasoning               | Complex diagrams, OCR, reasoning |
| Gemini     | 2023 | Natively multimodal from pretraining             | Long context, mixed modalities |
| InternVL   | 2024 | Dynamic resolution, strong open-source           | High-resolution image understanding |

### Key Formulas to Remember

**Cosine similarity** (the core operation in CLIP):
```
cos_sim(a, b) = dot(a, b) / (||a|| * ||b||)

If both vectors are L2-normalised: cos_sim(a, b) = dot(a, b)
```

**CLIP zero-shot prediction**:
```
scores = image_embedding @ class_embeddings.T    # (N, C)
probs  = softmax(scores * temperature)            # (N, C)
pred   = argmax(probs, axis=1)                    # (N,)
```

---

## Practice Problems

1. **CLIP similarity**: Why is cosine similarity better than Euclidean distance for comparing CLIP embeddings? (Hint: think about what normalisation does.)

2. **Contrastive learning**: In a batch of 64 (image, text) pairs, how many negative pairs are there? If you increase batch size to 4096, how does this affect training?

3. **Architecture design**: Flamingo freezes the vision encoder and language model, only training the cross-attention layers. What are the advantages and disadvantages of this design choice?

4. **Prompt engineering**: Why does "a photo of a {class}" outperform just "{class}" as a CLIP text prompt? What might you try for medical images?

5. **Evaluation**: A model achieves 70% VQA accuracy. Is this good? What baseline should you compare against? (Hint: what accuracy would a model achieve by always answering "yes"?)
