---
title: Vision-Language Models
track: cv
order: 7
estimatedTime: 40
difficulty: advanced
---

# Vision-Language Models (VLMs)

Vision-Language Models (VLMs) can understand both images and text at the same time. They power image captioning, visual question answering, zero-shot classification, and image-text retrieval. This lesson covers the key models and how to use them.

---

## 1. What VLMs Do

A VLM learns to connect images and text in the same embedding space. Once connected:
- You can search for an image using a text description
- You can answer questions about an image
- You can classify images using text labels (zero-shot, no training examples needed)

The two most important VLM families: **CLIP** (image-text alignment) and **image captioning models** (image to text generation).

---

## 2. CLIP: Image-Text Alignment

CLIP (Contrastive Language-Image Pre-Training) was trained on 400 million image-text pairs scraped from the internet. It learned to embed similar images and text close together in a shared vector space.

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

# Load model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Load an image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Classify with text labels (zero-shot)
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=texts, images=image, return_tensors='pt', padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# Probabilities for each text label
probs = outputs.logits_per_image.softmax(dim=-1)
for text, prob in zip(texts, probs[0]):
    print(f'{text}: {prob:.2%}')
```

**Quick check:** CLIP does zero-shot classification. What does "zero-shot" mean?
> No training examples for the specific task. You just provide text descriptions of the classes and CLIP picks the best match. No fine-tuning needed.

---

## 3. Image Captioning

Image captioning generates a text description from an image. Common use: generating alt-text, understanding scene content.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# Open an image
image = Image.open('your_image.jpg').convert('RGB')

# Generate caption (no prompt)
inputs = processor(image, return_tensors='pt')
output = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(output[0], skip_special_tokens=True)
print(f'Caption: {caption}')

# Generate caption with a prompt (conditional)
inputs = processor(image, text="A photo of", return_tensors='pt')
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 4. Visual Question Answering

VQA: given an image and a question, produce an answer.

```python
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')

image = Image.open('your_image.jpg').convert('RGB')
question = "What color is the car?"

inputs = processor(image, question, return_tensors='pt')
output = model.generate(**inputs, max_new_tokens=10)
answer = processor.decode(output[0], skip_special_tokens=True)
print(f'Answer: {answer}')
```

---

## 5. Using CLIP for Image Retrieval

Given a text query, find the most similar image in a database:

```python
import numpy as np

def encode_images(images, model, processor):
    inputs = processor(images=images, return_tensors='pt', padding=True)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)  # normalize

def encode_text(texts, model, processor):
    inputs = processor(text=texts, return_tensors='pt', padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)

# Precompute image features for your database
# image_features = encode_images(your_images, model, processor)  # (N, 512)

# At query time
query = "a red sports car"
text_features = encode_text([query], model, processor)  # (1, 512)

# Cosine similarity (dot product since features are normalized)
similarity = (text_features @ image_features.T).squeeze()   # (N,)
best_match = similarity.argmax().item()
print(f'Best match: image {best_match}, score: {similarity[best_match]:.3f}')
```

**Quick check:** Why do we normalize the feature vectors before computing dot products?
> Normalizing makes the dot product equal to the cosine similarity, which measures angle (semantic similarity) rather than magnitude. This is more meaningful for retrieval.

---

## 6. Fine-Tuning CLIP

For a custom classification task with specific domain images, fine-tuning CLIP on your data often beats zero-shot:

```python
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn

# Load pretrained
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Add a classification head on top of image features
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pixel_values):
        features = self.clip.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)

classifier = CLIPClassifier(model, num_classes=5)
```

---

## Sort the VLM Tasks

```widget
{
  "type": "concept-sort",
  "title": "CLIP, BLIP Captioning, or BLIP VQA?",
  "categories": [
    { "name": "CLIP", "color": "#5B5BD6" },
    { "name": "BLIP (Captioning)", "color": "#22C55E" },
    { "name": "BLIP (VQA)", "color": "#F97316" }
  ],
  "items": [
    { "text": "Given image, generate a text description", "category": "BLIP (Captioning)" },
    { "text": "\"Is there a cat in this image?\" → yes/no", "category": "BLIP (VQA)" },
    { "text": "Find the most similar image to this text query", "category": "CLIP" },
    { "text": "Zero-shot classification via text prompts", "category": "CLIP" },
    { "text": "Auto-annotate images for a dataset", "category": "BLIP (Captioning)" },
    { "text": "Visual question answering with text output", "category": "BLIP (VQA)" }
  ]
}
```

---

## Summary

| Model | Task | Key use |
|---|---|---|
| CLIP | Image-text similarity | Zero-shot classification, retrieval |
| BLIP (captioning) | Image to text | Auto-generate descriptions |
| BLIP (VQA) | Image + question to answer | Visual question answering |

**IOAI tip:** For image classification with no training data (or very little), CLIP zero-shot is a strong baseline. For tasks with labeled examples, fine-tune a CNN or ViT instead - they usually win when you have data.
