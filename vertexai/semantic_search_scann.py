#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai

#--------- Authenticate to Vertex AI
key_path = '/home/mkhuthir/apps/vertexaiproj-418218-cf52d0c8ffb4.json'
credentials = Credentials.from_service_account_file(key_path,
                                                    scopes=['https://www.googleapis.com/auth/cloud-platform'])
if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = 'vertexaiproj-418218'
REGION = 'us-central1'

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
#----------------------------------- 
# select model
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
#----------------------------------- 

import time
import pickle
import pandas as pd
import numpy as np

# load the stack overflow dataframe from csv file
so_database = pd.read_csv('../media/so_database_app.csv')

# load embeddings from pickle file
with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)

# add embeddings to so_database
so_database['embeddings'] = question_embeddings.tolist()

# Semantic Search using cosine similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# embed the query
query = ['How to concat dataframes pandas']
query_embedding = model.get_embeddings(query)[0].values

# build scann index


# search using scann
start = time.time()
neighbors, distances = index.search(query_embedding, final_num_neighbors = 1)
end = time.time()
print("\nLatency (ms):", 1000 * (end - start))
for id, dist in zip(neighbors, distances):
    print(f"[docid:{id}] [{dist}] -- {so_database.input_text[int(id)][:125]}...")