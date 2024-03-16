#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from transformers import Conversation
from transformers import pipeline
from transformers.utils import logging
import gc

logging.set_verbosity_error()

chatbot = pipeline(task="conversational",
                   model="facebook/blenderbot-400M-distill")

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

# free memory and garbage collect
del chatbot
gc.collect()