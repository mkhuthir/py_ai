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
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
# load data
so_df = pd.read_csv('../media/so_database_app.csv')
print(so_df)

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

#------------
# Generator function to yield batches of sentences
def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

#------------
def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]
#------------
def encode_text_to_embedding_batched(sentences, api_calls_per_second = 0.33, batch_size = 5):
    # Generates batches and calls embedding API
    
    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total = math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return embeddings_list_successful

so_questions = so_df.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
                            sentences=so_questions,
                            api_calls_per_second = 20/60, 
                            batch_size = 5)

print(f"{len(question_embeddings)} embeddings of size {len(question_embeddings[0])}")

