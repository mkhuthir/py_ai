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
from vertexai.language_models import TextGenerationModel

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
# select model
model = TextGenerationModel.from_pretrained("text-bison@001")

#----------------------------------- 
prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."

# The decoding strategy applies top_k, then top_p, then temperature (in that order)
# To adjust top_p and top_k and see different results, remember to set temperature to be greater than zero
# otherwise the model will always choose the token with the highest probability.

top_k = 20          # values between 1 and 40
top_p = 0.7         # sample the minimum set of tokens whose probabilities add up to probability p or greater.
temperature = 0.9   # creativity, values between 0-1, 

response = model.predict(prompt=prompt,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p)

print(f"\nprompt:\n{prompt}\n\nresponse with temperature={temperature} top_k={top_k} top_p={top_p}:\n{response.text}")

