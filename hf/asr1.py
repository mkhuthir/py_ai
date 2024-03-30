#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
from datasets import load_dataset

asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en",
               device=0)

print("ASR sampling rate =",asr.feature_extractor.sampling_rate)

dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)

example = next(iter(dataset))
dataset_head = dataset.take(5)
# print(list(dataset_head)[2])
# print(example)

print("example sampling rate",example['audio']['sampling_rate'])

asr(example["audio"]["array"])
print(example["text"])
