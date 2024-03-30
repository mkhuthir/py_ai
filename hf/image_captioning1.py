#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("../media/hf_friends.jpg")
image.show()

# Conditional Image Captioning
print("Conditional Image Captioning")
text = "a photograph of"
inputs = processor(image, text, return_tensors="pt")
print(inputs)

out = model.generate(**inputs)
print(out)
print(processor.decode(out[0], skip_special_tokens=True))

# Unconditional Image Captioning
print("Unconditional Image Captioning")
inputs = processor(image,return_tensors="pt")
print(inputs)

out = model.generate(**inputs)
print(out)
print(processor.decode(out[0], skip_special_tokens=True))