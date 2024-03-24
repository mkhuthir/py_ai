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

