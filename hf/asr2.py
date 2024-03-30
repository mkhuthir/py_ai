#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
from datasets import load_dataset
import soundfile as sf
import numpy as np
import librosa
import io

asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en",
               device=0)

print("ASR sampling rate =",asr.feature_extractor.sampling_rate)

# read audio file
audio, sampling_rate = sf.read('../media/narration2.mp3')
print("Audio sampling rate =",sampling_rate)
print("Audio shape =",audio.shape)

# transpose array to match librosa requirements 
audio_transposed = np.transpose(audio)
print("Audio transposed shape =",audio_transposed.shape)

# change audio from stereo to mono
audio_mono = librosa.to_mono(audio_transposed)
print("Audio mono shape =",audio_mono.shape)

# adjust audio sampling rate to match ASR sampling rate
audio_16KHz = librosa.resample(audio_mono,
                               orig_sr=sampling_rate,
                               target_sr=16000)

output=asr(
            audio_16KHz,
            chunk_length_s=30, # 30 seconds
            batch_size=4,
            return_timestamps=True)

print("output text =",output["text"])