#! /usr/bin/python3

# Muthanna Alwahash
# Mar 2024

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set tensorflow logs level

from transformers.utils import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import gc

logging.set_verbosity_error()

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
print(embeddings1)

embeddings2 = model.encode(sentences2, convert_to_tensor=True)
print(embeddings2)

cosine_scores = util.cos_sim(embeddings1,embeddings2)
print(cosine_scores)

for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))
    
# free memory and garbage collect
del model
gc.collect()
