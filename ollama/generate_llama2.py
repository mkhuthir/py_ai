#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama
import time

timer_start = time.time()
#------------------------
output =ollama.generate(model='llama2',
                        prompt='Why is the sky blue?')
#------------------------
timer_end = time.time()
print(f"\ntime taken (ms): {1000 * (timer_end - timer_start)}\n\n")

print(output['response'])