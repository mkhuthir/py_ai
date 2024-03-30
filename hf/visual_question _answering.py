#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import BlipForImageTextRetrieval, AutoProcessor
from PIL import Image

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("../media/hf_friends.jpg")
image.show

question = "who is in the picture?"
inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
