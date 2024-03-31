#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load embeddings from pickle file
with open('../media/question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
X = question_embeddings
print("embeddings data shape (X) = ",X.shape)

# load the dataset from csv file
so_df = pd.read_csv('../media/so_database_app.csv')
y = so_df['category'].values
print("category data shape (y) = ",y.shape)


