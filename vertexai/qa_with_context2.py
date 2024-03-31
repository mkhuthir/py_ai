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
# select models
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextGenerationModel

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
generation_model = TextGenerationModel.from_pretrained("text-bison@001")
#----------------------------------- 

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

# the query
query = ['How to make the perfect lasagna']
print(f"\nquery:\n {query[0]}\n")

# embed the query
query_embedding = embedding_model.get_embeddings(query)[0].values

# compare embeddings
cos_sim_array = cosine_similarity([query_embedding],
                                  list(so_database.embeddings.values))

# find the max semantic match
index_doc_cosine = np.argmax(cos_sim_array)

# build prompt using the match as a context
context = "Question: " + so_database.input_text[index_doc_cosine] +\
"\n Answer: " + so_database.output_text[index_doc_cosine]

prompt = f"""Here is the context: {context}
             Using the relevant information from the context,
             provide an answer to the query: {query}."
             If the context doesn't provide \
             any relevant information, \
             answer with \
             [I couldn't find a good match in the \
             document database for your query]
             """

temperature = 0.2

# get the response
response = generation_model.predict(prompt = prompt,
                                    temperature = temperature,
                                    max_output_tokens = 1024)

# show response with MarkDown
from rich.console import Console
from rich.markdown import Markdown
console = Console()
md = Markdown(response.text)
console.print(md)
