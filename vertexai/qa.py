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

# load the stack overflow dataframe from csv file
so_df = pd.read_csv('../media/so_database_app.csv')
print("Shape: " + str(so_df.shape))
print(so_df)

# load embeddings from pickle file
with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
print(question_embeddings)

