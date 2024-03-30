#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline, Conversation 

zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0)

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']

classification = zero_shot_classifier(
    sequence_to_classify,
    candidate_labels,
    multi_label=True)

print(classification)
print(classification['labels'])
print(classification['scores'])