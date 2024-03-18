#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

PROJECT_ID = 'Muth1'
REGION = 'us-central1'

import vertexai
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
