#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
from datasets import load_dataset, Audio

# This dataset is a collection of different sounds of 5 seconds
# will take first sample
dataset = load_dataset("ashraq/esc50",
                        split="train[0:1]")

# change the sampling rate to match classifier sampling rate
dataset = dataset.cast_column("audio",
                              Audio(sampling_rate=48_000))

audio_sample = dataset[0]
print("audio sample rate = ",audio_sample["audio"]["sampling_rate"])

pipe = pipeline(task="zero-shot-audio-classification",
                model="laion/clap-htsat-unfused",
                device=0)

print("classifier sample rate = ",pipe.feature_extractor.sampling_rate)

candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a dog",
                    "Sound of a bird singing",
                    "Sound of an airplane"]

output= pipe(audio_sample["audio"]["array"],
             candidate_labels=candidate_labels)

print(output)
