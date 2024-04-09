#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama
import time

timer_start = time.time()
#------------------------
response = ollama.chat(model='llama2', 
                       messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
#------------------------
timer_end = time.time()
print(f"\ntime taken (ms): {1000 * (timer_end - timer_start)}\n\n")

print(response['message']['content'])

