#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama

print(ollama.generate(model='llama2', prompt='Why is the sky blue?')['response'])
