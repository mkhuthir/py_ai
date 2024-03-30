#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import CLIPModel, AutoProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

image = Image.open("../media/kittens.jpeg")
image.show

labels = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=labels,
                   images=image,
                   return_tensors="pt",
                   padding=True)

outputs = model(**inputs)
print(outputs)
print(outputs.logits_per_image)

probs = outputs.logits_per_image.softmax(dim=1)[0]
print(probs)

probs = list(probs)
for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
  