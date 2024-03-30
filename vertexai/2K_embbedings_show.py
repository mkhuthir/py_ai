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
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors

so_df = pd.read_csv('../media/so_database_app.csv')
print(so_df.head())

with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)

print("Shape: " + str(question_embeddings.shape))
print(question_embeddings)

# use only 1500 out of 2000
clustering_dataset = question_embeddings[:1500]

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0, 
                n_init = 'auto').fit(clustering_dataset)

kmeans_labels = kmeans.labels_

PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)

#-------------- Create scatter plot
x_values = new_values[:,0]
y_values = new_values[:,1]
labels = so_df[:1500]
k_labels = kmeans_labels

fig, ax = plt.subplots()
scatter = ax.scatter(x_values, 
                     y_values, 
                     c = k_labels, 
                     cmap='Set1', 
                     alpha=0.5, 
                     edgecolors='k', 
                     s = 40)  # Change the denominator as per n_clusters

# Create a mplcursors object to manage the data point interaction
cursor = mplcursors.cursor(scatter, hover=True)

#axes
ax.set_title('Embedding clusters visualization in 2D')  # Add a title
ax.set_xlabel('X_1')  # Add x-axis label
ax.set_ylabel('X_2')  # Add y-axis label

# Define how each annotation should look
@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(labels.category[sel.index])
    sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.95) # Set annotation's background color
    sel.annotation.set_fontsize(14)  
plt.show()
