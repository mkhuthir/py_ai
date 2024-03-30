#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
import torch

summarizer = pipeline(task="summarization",
                      model="facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16,
                      device=0)

text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""

summary = summarizer(text,
                     min_length=10,
                     max_length=100)

print(summary)