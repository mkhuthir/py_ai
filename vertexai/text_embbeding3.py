#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

#--------- Authenticate to Vertex AI
key_path = '/home/mkhuthir/apps/vertexaiproj-418218-cf52d0c8ffb4.json'
credentials = Credentials.from_service_account_file(key_path,
                                                    scopes=['https://www.googleapis.com/auth/cloud-platform'])
if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = 'vertexaiproj-418218'
REGION = 'us-central1'
#----------------------------------- 

import vertexai
from vertexai.language_models import TextEmbeddingModel

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import time

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
# load data
so_df = pd.read_csv('../media/so_database_app.csv')
print(so_df)

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# Generator function to yield batches of sentences
def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

so_questions = so_df[0:200].input_text.tolist() 
batches = generate_batches(sentences = so_questions)

batch = next(batches)
print(len(batch))

def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

batch_embeddings = encode_texts_to_embeddings(batch)

print(f"{len(batch_embeddings)} embeddings of size {len(batch_embeddings[0])}")