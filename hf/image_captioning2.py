#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
from PIL import Image

pipe = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-base",
                max_new_tokens=20,
                device=0)

image = Image.open("../media/kittens.jpeg")
image.show()

output=pipe(image)
print(output[0]['generated_text'])

