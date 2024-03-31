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
prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."

output = model.predict(prompt=prompt).text
print(f"\nprompt:\n{prompt}\n\nanswer:\n{output}\n")

#----------------------------------- 
prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""
output = model.predict(prompt=prompt).text
print(f"\nprompt:\n{prompt}\n\nanswer:\n{output}\n")

#----------------------------------- 
prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world. 
Jesse moves to Texas for what she believes is her dream job, 
only to discover a dark secret that will make \
her question everything. 
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen), 
and the lab director Sam Porter (Fredrik Johansson). 
Together the trio work long hours on their research \
in a hope to change the world for good. 
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute, 
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""
output = model.predict(prompt=prompt).text
print(f"\nprompt:\n{prompt}\n\nanswer:\n{output}\n")

#----------------------------------- 
