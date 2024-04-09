#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama

response = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response['message']['content'])