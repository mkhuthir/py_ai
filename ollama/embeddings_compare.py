#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama
import time


text='Llamas are members of the camelid family'

embedding1 = ollama.embeddings(model='all-minilm'       ,prompt=text)
embedding2 = ollama.embeddings(model='nomic-embed-text' ,prompt=text)
embedding3 = ollama.embeddings(model='mxbai-embed-large',prompt=text)


print(embedding1['embedding'][:5])
print(embedding2['embedding'][:5])
print(embedding3['embedding'][:5])