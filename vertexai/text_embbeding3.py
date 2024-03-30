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

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
