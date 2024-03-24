#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

PROJECT_ID = 'Muth1'
REGION = 'us-central1'

import vertexai
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)

from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

embedding = embedding_model.get_embeddings(
    ["life"])

vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])

embedding = embedding_model.get_embeddings(
    ["What is the meaning of life?"])

vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])
