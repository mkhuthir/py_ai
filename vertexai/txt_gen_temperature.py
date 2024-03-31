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
prompt = "Complete the sentence: \
As I prepared the picture frame, \
I reached into my toolkit to fetch my:"

temperature = 0.0 # model to consistently output the same result for the same input
response = model.predict(prompt=prompt,
                         temperature=temperature,
)

print(f"\nprompt:\n{prompt}\n\nresponse with temperature = {temperature}:\n{response.text}\n")

temperature = 1.0 # more creativity, such as brainstorming, summarization
response = model.predict(prompt=prompt,
                         temperature=temperature,
)

print(f"\nresponse with temperature = {temperature}:\n{response.text}\n")
