#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
import sounddevice as sd

text = """
They run. They laugh. I see the glow shining on their eyes. Not like hers. She seems distant, \
strange, somehow cold. A couple of days after, I receive the call. I curse, scream, and cry. \
They're gone. I drink and cry and dream over and over. Why? Time drags me, expending days, \
months, or maybe years. But the pain still remains. It grows. It changes me. Someone tells \
me she got released from the psychiatric ward. 426 days after. I got confused. \
the psychiatric ward. 426 days after. My head spins. I got confused. The loneliness. \
It's time. The road has become endless. I feel the cold wind on my face. My eyes burn. \
I get to the house. It all looks the same. I can hear them, laughing like there were \
no souls taken. And then she comes. She sees me with kindness in her eyes. She looks \
at the flowers and she says she still loves me. Those words hurt me like a razor blade. \
Good bye, my love.
"""

print(text)

narrator = pipeline("text-to-speech",
                    model="kakao-enterprise/vits-vctk",     # vits-ljs is another alternative
                    device=0)

narrated_text = narrator(text)
sd.play(narrated_text["audio"][0],narrated_text["sampling_rate"])
sd.wait()


