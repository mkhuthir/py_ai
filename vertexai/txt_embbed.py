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

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
# select model
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

#----------------------------------- 
embedding = embedding_model.get_embeddings(
    ["life"])

vector = embedding[0].values
print("embedding for 'life'")
print(f"Length = {len(vector)}")
print(vector[:10])

embedding = embedding_model.get_embeddings(
    ["What is the meaning of life?"])

vector = embedding[0].values
print("embedding for 'What is the meaning of life?'")
print(f"Length = {len(vector)}")
print(vector[:10])

# Calculate the similarity between two sentences as a number between 0 and 1.

from sklearn.metrics.pairwise import cosine_similarity

emb_1 = embedding_model.get_embeddings(
    ["What is the meaning of life?"]) # 42!

emb_2 = embedding_model.get_embeddings(
    ["How does one spend their time well on Earth?"])

emb_3 = embedding_model.get_embeddings(
    ["Would you like a salad?"])

# Wrap the embeddings python list in another list because the cosine_similarity function expects either a 2D numpy array or a list of lists.
vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]

print("Similarity with 'What is the meaning of life?' ",cosine_similarity(vec_1,vec_2)) 
print("Similarity with 'How does one spend their time well on Earth?' ",cosine_similarity(vec_2,vec_3))
print("Similarity with 'Would you like a salad?' ",cosine_similarity(vec_1,vec_3))

in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."

in_pp_1 = ["kids", "play", "park"]
in_pp_2 = ["play", "kids", "park"]

embeddings_1 = [emb.values for emb in embedding_model.get_embeddings(in_pp_1)]

import numpy as np
emb_array_1 = np.stack(embeddings_1)
print(emb_array_1.shape)

embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]
emb_array_2 = np.stack(embeddings_2)
print(emb_array_2.shape)

emb_1_mean = emb_array_1.mean(axis = 0) 
print(emb_1_mean.shape)

emb_2_mean = emb_array_2.mean(axis = 0)

print(emb_1_mean[:4])
print(emb_2_mean[:4])

print(in_1)
print(in_2)

embedding_1 = embedding_model.get_embeddings([in_1])
embedding_2 = embedding_model.get_embeddings([in_2])

vector_1 = embedding_1[0].values
print(vector_1[:4])
vector_2 = embedding_2[0].values
print(vector_2[:4])

