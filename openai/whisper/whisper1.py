#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])

