#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama
import time

timer_start = time.time()
#------------------------
embedding = ollama.embeddings(model='nomic-embed-text',
                              prompt='Llamas are members of the camelid family')
#------------------------
timer_end = time.time()
print(f"\ntime taken (ms): {1000 * (timer_end - timer_start)}\n\n")

print(embedding['embedding'][:5])