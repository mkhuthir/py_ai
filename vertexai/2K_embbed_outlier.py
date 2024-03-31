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
from sklearn.ensemble import IsolationForest

# load embeddings from pickle file
with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
print("embeddings data shape = ",question_embeddings.shape)

# load the stack overflow dataframe from csv file
so_df = pd.read_csv('../media/so_database_app.csv')
print("data shape (y) = ",so_df.shape)

# new outlier input
input_text = """I am making cookies but don't 
                remember the correct ingredient proportions. 
                I have been unable to find 
                anything on the web."""

# embed and add the outlier
emb = model.get_embeddings([input_text])[0].values
embeddings_l = question_embeddings.tolist()
embeddings_l.append(emb)
embeddings_array = np.array(embeddings_l)
print("new embeddings data shape = " + str(embeddings_array.shape))

# Add the outlier text to the end of the stack overflow dataframe
new_row = pd.Series([input_text, None, "baking"], 
                    index=so_df.columns)
so_df.loc[len(so_df)] = new_row
print(so_df.tail())

# Use Isolation Forest to identify potential outliers
clf = IsolationForest(contamination=0.005, 
                      random_state = 2) 
preds = clf.fit_predict(embeddings_array)

print(f"\n{len(preds)} predictions. Set of possible values: {set(preds)}")
print("\nOutliers list :\n",so_df.loc[preds == -1])

# remove the added outlier
print ("\nremoving the added outlier....\n")
so_df = so_df.drop(so_df.index[-1])
print(so_df.tail())
