#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import ollama
from sklearn.metrics.pairwise import cosine_similarity


text='Llamas are members of the camelid family'
model1 = 'all-minilm'
model2 = 'nomic-embed-text'
model3 = 'mxbai-embed-large'

embedding1 = ollama.embeddings(model= model1 ,prompt=text)['embedding']
embedding2 = ollama.embeddings(model= model2 ,prompt=text)['embedding']
embedding3 = ollama.embeddings(model= model3 ,prompt=text)['embedding']


print(f"{model1} \t\t length {len(embedding1)}")
print(f"{model2} \t length {len(embedding2)}")
print(f"{model3} \t length {len(embedding3)}")