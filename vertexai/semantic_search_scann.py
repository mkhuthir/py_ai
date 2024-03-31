#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

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
import scann

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

# Create index using scann
def create_index(embedded_dataset, 
                 num_leaves,
                 num_leaves_to_search,
                 training_sample_size):
    
    # normalize data to use cosine sim as explained in the paper
    normalized_dataset = embedded_dataset / np.linalg.norm(embedded_dataset, axis=1)[:, np.newaxis]
    
    searcher = (
        scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
        .tree(
            num_leaves = num_leaves,
            num_leaves_to_search = num_leaves_to_search,
            training_sample_size = training_sample_size,
        )
        .score_ah(2, anisotropic_quantization_threshold = 0.2)
        .reorder(100)
        .build()
    )
    return searcher

index = create_index(embedded_dataset = question_embeddings, 
                     num_leaves = 25,
                     num_leaves_to_search = 10,
                     training_sample_size = 2000)

# search using scann
start = time.time()
neighbors, distances = index.search(query_embedding, final_num_neighbors = 1)
end = time.time()
print(f"\nscann search latency (ms): {1000 * (end - start)}\n")

for id, dist in zip(neighbors, distances):
    print(f"[docid:{id}] [{dist}] -- {so_database.input_text[int(id)][:125]}...")

# search using cos_sim
start = time.time()
cos_sim_array = cosine_similarity([query_embedding], list(so_database.embeddings.values))
index_doc = np.argmax(cos_sim_array)
end = time.time()
print(f"\ncos-sim search latency (ms): {1000 * (end - start)}\n")

print(f"[docid:{index_doc}] [{np.max(cos_sim_array)}] -- {so_database.input_text[int(index_doc)][:125]}...")
