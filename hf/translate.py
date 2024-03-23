#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024



from transformers import pipeline 
from transformers.utils import logging
import torch
import gc

logging.set_verbosity_error()

translator = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16,
                      device=0)

text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""

text_translated = translator(text,
                             src_lang="eng_Latn",
                             tgt_lang="arz_Arab")

print (text_translated)

# free memory and garbage collect
del translator
gc.collect()
