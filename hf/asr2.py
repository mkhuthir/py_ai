#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from transformers import pipeline 
from transformers.utils import logging
from datasets import load_dataset
import soundfile as sf
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level
logging.set_verbosity_error()

asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en",
               device=0)

print("ASR sampling rate =",asr.feature_extractor.sampling_rate)

audio, sampling_rate = sf.read('../media/narration2.mp3')

print("Audio sampling rate =",sampling_rate)
print("Audio shape =",audio.shape)
