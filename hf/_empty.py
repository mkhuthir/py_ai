#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from transformers import pipeline 
from transformers.utils import logging
import gc

logging.set_verbosity_error()



# free memory and garbage collect
# del xxxx
gc.collect()
