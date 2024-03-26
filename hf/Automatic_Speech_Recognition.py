#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from transformers import pipeline 
from transformers.utils import logging
from datasets import load_dataset

logging.set_verbosity_error()

dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)

example = next(iter(dataset))
dataset_head = dataset.take(5)

list(dataset_head)
list(dataset_head)[2]

print(example)
