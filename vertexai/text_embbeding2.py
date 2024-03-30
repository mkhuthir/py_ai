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

#----------------------------------- 

import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors

# initialize vertex
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)

in_1 = "Missing flamingo discovered at swimming pool"
in_2 = "Sea otter spotted on surfboard by beach"
in_3 = "Baby panda enjoys boat ride"
in_4 = "Breakfast themed food truck beloved by all!"
in_5 = "New curry restaurant aims to please!"
in_6 = "Python developers are wonderful people"
in_7 = "TypeScript, C++ or Java? All are great!" 

in_list = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

embeddings = []
for in_text in in_list:
    emb = embedding_model.get_embeddings([in_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings)

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

# Reduce embeddings using PCA from 768 to 2 dimensions for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)

print("Shape: " + str(new_values.shape))
print(new_values)

# Create scatter plot
x_values = new_values[:,0]
y_values = new_values[:,1]
labels = in_list
fig, ax = plt.subplots()
scatter = ax.scatter(x_values,
                     y_values,
                     alpha = 0.5,
                     edgecolors='k',
                     s = 40) 
# Create a mplcursors object to manage the data point interaction
cursor = mplcursors.cursor(scatter, hover=True)
ax.set_title('Embedding visualization in 2D')  # Add a title
ax.set_xlabel('X_1')  # Add x-axis label
ax.set_ylabel('X_2')  # Add y-axis label
# Define how each annotation should look
@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(labels[sel.index])
    sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.5) # Set annotation's background color
    sel.annotation.set_fontsize(12) 
plt.show()