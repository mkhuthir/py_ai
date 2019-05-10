#! /usr/bin/python3

# coding: utf-8

# # Python NLTK Natural Language Tool Kit
# ## Topics covered in this video
# - Exploring the NLTK corpus  
# - Dictionary definitions   
# - Punctuation and stop words  
# - Stemming and lemmatization  
# - Sentence and word tokenizers 
# - Parts of speech tagging  
# - word2vec  
# - Clustering and classifying
# 
# ### NLTK Setup
# First you need to install the nltk library with 'pip install nltk' or some equivalent shell command.  
# Then you need to download the nltk corpus by running  
# ```python  
# import nltk  
# nltk.download()```
# This will open the NLTK downloader dialog window where you should just click Download All. The corpus is a large and varied body of sample documents that you'll need for this video, including dictionaries and word lists like stop words. You can uninstall it later if you have a shortage of disk space with *pip uninstall nltk*.
# 

# In[1]:


import nltk
from nltk.book import *

print(type(text1))
print(len(text1))
print(len(set(text1)))
print(text1[:10])
print(text2[:10])


# In[2]:


from nltk.corpus import gutenberg
print(gutenberg.fileids())
hamlet = gutenberg.words('shakespeare-hamlet.txt')
print(len(hamlet))
hamlet_sentences = gutenberg.sents('shakespeare-hamlet.txt')
print(len(hamlet_sentences))
print(hamlet_sentences[1024])
print(len(gutenberg.paras('shakespeare-hamlet.txt')))


# ### Get the count of a word in a document, or the context of every occurence of a word in a document.

# In[68]:


print(text1.count('horse'))
print(text1.concordance('passion'))
print(text2.concordance('passion'))


# **FreqDist and most_common**  
# We can use FreqDist to find the number of occurrences of each word in the text.  
# By getting len(vocab) we get the number of unique words in the text (including punctuation).  
# And we can get the most common words easily too.

# In[69]:


vocab = nltk.FreqDist(text1)
print(len(vocab))
print(vocab.most_common(20))


# Here we got the 80 most common words, filtered only the ones with at least 3 characters, then sorted them descending by number of occurences.  
# A better way is to first remove all the *stop words* (see below), then get the FreqDist.

# In[70]:


mc = sorted([w for w in vocab.most_common(80) if len(w[0]) > 3], key=lambda x: x[1], reverse=True)
print(mc)


# ### A dispersion plot shows you where in the document a word is used. You can pass in a list of words.

# In[71]:


text1.dispersion_plot(['capture', 'whale', 'life', 'death', 'kill'])


# ### Dictionary definitions
# Use wordnet synsets to get word definitions and examples of usage.  
# The [0] is required because synsets returns a list, with an entry for each POS.

# In[72]:


from nltk.corpus import wordnet as wn
w = wn.synsets("unmitigated")[0]
print(w.name(), '-', w.definition())
print(w.examples())


# ### Punctuation and Stop Words
# Text analysis is often faster and easier if you can remove useless words.  
# NLTK provides a list of these stop words so it's easy to filter them out of your text prior to processing.  
# Here, 15% of our text is punctuation, and 40% is stop words. So we shrink the text by more than half by stripping out punctuation and stop words.

# In[73]:


from string import punctuation
print(punctuation)
without_punct = [w for w in text1 if w not in punctuation]  # this is called a list comprehension

from nltk.corpus import stopwords
sw = stopwords.words('english')
print(sw)
without_sw = [w for w in without_punct if w not in sw] 

print(len(text1))
print(len(without_punct))
print(len(without_sw))


# ### Stemming and Lemmatization
# These term normalization algorithms strip the word endings off to reduce the number of root words for easier matching.  
# This is useful for search term matching. [NLTK stemming docs](https://www.nltk.org/api/nltk.stem.html)

# In[74]:


from nltk.stem.porter import PorterStemmer
st = PorterStemmer()
words = ['is', 'are', 'bought', 'buys', 'giving', 'jumps', 'jumped', 'birds', 'do', 'does', 'did', 'doing']
for word in words:
    print(word, st.stem(word))


# **WordNet Lemmatizer**   
# The difference is that the result of stemming may not be an actual word, but lemmatization returns the root word. NLTK supports both.  
# You can also try the Lancaster or Snowball stemmers. The Snowball stemmer supports numerous languages: Arabic, Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish and Swedish.

# In[75]:


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
words = ['is', 'are', 'bought', 'buys', 'giving', 'jumps', 'jumped', 'birds', 'do', 'does', 'did', 'doing']
for word in words:
    print(word, wnl.lemmatize(word))


# ### Sentence and Word Tokenizers
# Sentence tokenizer breaks text down into a list of sentences. It's pretty good at handling punctuation and decimal numbers.  
# [Word tokenizer](https://www.nltk.org/api/nltk.tokenize.html) breaks a string down into a list of words and punctuation.  
# It is also easy to get parts of speech using nltk.pos_tag. There are different tagsets, depending on how much detail you want. I like universal.

# In[76]:


from nltk.tokenize import sent_tokenize, word_tokenize
s = 'Hello. I am Joe! I like Python. 263.5 is a big number.'  # 4 sentences
print(sent_tokenize(s))

w = word_tokenize('The quick brown fox jumps over the lazy dog.')
print(w)


# ### Parts of Speech Tagging
# To break a block of text down into its parts of speech use pos_tag.  
# The default tagset uses 2 or 3 letter tokens that are hard for me to understand. [StackOverflow](https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk) has a great decoder for the default POS tags.  
# The Universal tagset gives a more familiar looking tag (noun, verb, adj).  
# NLTK includes several other tagsets you can try.

# In[77]:


w = word_tokenize('The quick brown fox jumps over the lazy dog.')
print(w)
print(nltk.pos_tag(w))
print(nltk.pos_tag(w, tagset='universal'))


# ### Word2Vec
# [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) uses neural networks to analyze words in a corpus by using the contexts of words. 
# It takes as its input a large corpus of text, and maps unique words to a vector space, such that 
# words that share common contexts in the corpus are located in close proximity to one another in the space.  
# Word2Vec does NOT look at word meanings, it only finds words that are used in combination with other words. So *frying* and *pan* may have a high similarity.  
# You can see here the context of one word (pain) for two different corpora.  
# This uses the popular gensim library, which is not part of NLTK.

# In[78]:


from gensim.models import Word2Vec
emma_vec = Word2Vec(gutenberg.sents('austen-emma.txt'))
leaves_vec = Word2Vec(gutenberg.sents('whitman-leaves.txt'))
print(emma_vec.wv.most_similar('pain', topn=6))
print(leaves_vec.wv.most_similar('pain', topn=6))


# In[79]:


from gensim.models import Word2Vec
from nltk.corpus import stopwords
from string import punctuation
import pprint as pp

bible_sents = gutenberg.sents('bible-kjv.txt')
sw = stopwords.words('english')
bible = [[w.lower() for w in s if w not in punctuation and w not in sw] for s in bible_sents]
print(len(bible))

bible_vec = Word2Vec(bible)
pp.pprint(bible_vec.wv.most_similar('god', topn=8))
pp.pprint(bible_vec.wv.most_similar('creation', topn=5))


# ### k-Means Clustering
# [Clustering](http://www.nltk.org/api/nltk.cluster.html) groups similar items together.  
# The K-means clusterer starts with k arbitrarily chosen means (or centroids) then assigns each vector to the cluster with the closest mean. It then recalculates the means of each cluster as the centroid of its vector members. This process repeats until the cluster memberships stabilize. [NLTK docs on this example](https://www.nltk.org/_modules/nltk/cluster/kmeans.html)  
# This example clusters int vectors, which you can think of as points on a plane. But you could also use clustering to cluster similar documents by vocabulary/topic.

# In[80]:


import numpy as np
from nltk.cluster import KMeansClusterer, euclidean_distance

vectors = [np.array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
means = [[4, 3], [5, 5]]

clusterer = KMeansClusterer(2, euclidean_distance, initial_means=means)
clusters = clusterer.cluster(vectors, True, trace=True)

print('Clustered:', vectors)
print('As:', clusters)
print('Means:', clusterer.means())


# **k-Means Clustering, Example-2**  
# In this example we cluster an array of 6 points into 2 clusters.  
# The initial centroids are randomly chosen by the clusterer, and it does 10 iterations to regroup the clusters and recalculate centroids. 

# In[103]:


vectors = [np.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test k-means using 2 means, euclidean distance, and 10 trial clustering repetitions with random seeds
clusterer = KMeansClusterer(2, euclidean_distance, repeats=10)
clusters = clusterer.cluster(vectors, True)
centroids = clusterer.means()
print('Clustered:', vectors)
print('As:', clusters)
print('Means:', centroids)

# classify a new vector
vector = np.array([2,2])
print('classify(%s):' % vector, end=' ')
print(clusterer.classify(vector))


# **Plot a Chart of the Clusters in Example-2**  
# Make a Scatter Plot of the two clusters using matplotlib.pyplot.   
# We plot all the points in cluster-0 blue, and all the points in cluster-1 red. Then we plot the two centroids in orange.  
# I used list comprehensions to create new lists for all the x0, y0, x1 and y1 values.

# In[104]:


import matplotlib.pyplot as plt

x0 = np.array([x[0] for idx, x in enumerate(vectors) if clusters[idx]==0])
y0 = np.array([x[1] for idx, x in enumerate(vectors) if clusters[idx]==0])
plt.scatter(x0,y0, color='blue')
x1 = np.array([x[0] for idx, x in enumerate(vectors) if clusters[idx]==1])
y1 = np.array([x[1] for idx, x in enumerate(vectors) if clusters[idx]==1])
plt.scatter(x1,y1, color='red')

xc = np.array([x[0] for x in centroids])
yc = np.array([x[1] for x in centroids])
plt.scatter(xc,yc, color='orange')
plt.show()

