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

import pickle
import pandas as pd
import numpy as np

# load the stack overflow dataframe from csv file
so_database = pd.read_csv('../media/so_database_app.csv')
print(f"\ndata shape: {so_database.shape}\n")
print(so_database)

# load embeddings from pickle file
with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
print("\n",question_embeddings)

# add embeddings to so_database
so_database['embeddings'] = question_embeddings.tolist()
print(f"\ndatabase+embeddings shape: {so_database.shape}\n")

# Semantic Search using cosine similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# embed the query
query = ['How to concat dataframes pandas']
query_embedding = model.get_embeddings(query)[0].values

# compare embeddings
cos_sim_array = cosine_similarity([query_embedding],
                                  list(so_database.embeddings.values))
print(f"\ncos_sim_array shape: {cos_sim_array.shape}\n")

# find the max match
index_doc_cosine = np.argmax(cos_sim_array)

# print selected match
print(f"\nthe similar question index: {index_doc_cosine}\n")
print(f"\ninput: {so_database.input_text[index_doc_cosine]}\n")
print(f"\noutput: {so_database.output_text[index_doc_cosine]}\n")