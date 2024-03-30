#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline 
from transformers import Conversation

chatbot = pipeline(task="conversational",
                   model="facebook/blenderbot-400M-distill",
                   device=0)

user_message = """
What are some fun activities I can do in the winter?
"""

conversation = Conversation(user_message)
print(conversation)

conversation = chatbot(conversation)
print(conversation)

# you need to add if you are expecting the bot to remember the old conversation
conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })

print(conversation)

conversation = chatbot(conversation)
print(conversation)