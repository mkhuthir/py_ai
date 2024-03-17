#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from transformers import pipeline, Conversation 
from transformers.utils import logging
import gc

logging.set_verbosity_error()

zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli")

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']

classification = zero_shot_classifier(
    sequence_to_classify,
    candidate_labels,
    multi_label=True)

print(classification)
print(classification['labels'])
print(classification['scores'])

# free memory and garbage collect
del zero_shot_classifier
gc.collect()
