#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

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

# print(list(dataset_head)[2])
# print(example)

asr = pipeline(task="automatic-speech-recognition",
               model="./models/distil-whisper/distil-small.en",
               device=0)

print("asr sampling rate =",asr.feature_extractor.sampling_rate)
print("example sampling rate",example['audio']['sampling_rate'])

asr(example["audio"]["array"])
print(example["text"])
