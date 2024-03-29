#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers import pipeline 
from transformers.utils import logging
import gc

logging.set_verbosity_error()

from datasets import load_dataset, load_from_disk

# This dataset is a collection of different sounds of 5 seconds
# dataset = load_dataset("ashraq/esc50",
#                       split="train[0:10]")
dataset = load_from_disk("/ashraq/esc50/train")


audio_sample = dataset[0]

print (audio_sample)

zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused",
    device=0)

zero_shot_classifier.feature_extractor.sampling_rate

print(audio_sample["audio"]["sampling_rate"])

from datasets import Audio

dataset = dataset.cast_column(
    "audio",
     Audio(sampling_rate=48_000))

audio_sample = dataset[0]

print(audio_sample)

candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)

candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]

zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)



# free memory and garbage collect
# del xxxx
gc.collect()
