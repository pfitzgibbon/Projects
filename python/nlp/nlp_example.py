import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

#sklearn libraries
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

#nltk libraries
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

#from gensim.test.utils import common_texts, get_tmpfile
#from gensim.models import Word2Vec
import sqlite3
from sqlite3 import Error

#similar questions to what this is aiming to do 
#https://python.developreference.com/article/12873061/Clustering+synonym+words+using+NLTK+and+Wordnet
#https://stackoverflow.com/questions/47757435/clustering-synonym-words-using-nltk-and-wordnet

#taken from https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
#clean word data
def word_clean(dframe, columnname):
    for w in range(len(dframe[columnname])):
        dframe[columnname][w] = dframe[columnname][w].lower()
        #remove punctuation
        dframe[columnname][w] = re.sub('[^a-zA-Z]', ' ', dframe[columnname][w])
        #remove tags
        dframe[columnname][w] = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", dframe[columnname][w])
        #remove special characters and digits
        dframe[columnname][w] = re.sub("(\\d|\\W)+"," ", dframe[columnname][w])
    return dframe

def nltk_tag_to_wordnet_tag(nltk_tag):
    #read in nltk tag and translate it to wordnet tag 
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence, use_syn_hyp = True):
    lem = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if not word in stop_words and len(word) >2:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lem.lemmatize(word, tag))
                if use_syn_hyp == True: #.part_meronyms(), .substance_meronyms() word heirarchy
                    lemmatized_sentence.append(wordnet.synsets(word+'.'+tag+'.01'))
                    try:
                        lemmatized_sentence.append(wordnet.synset(word+'.'+tag+'.01').hypernyms())
                    except:
                        pass
    #clean up the data again
    lem_sen = " ".join(str(v) for v in lemmatized_sentence)
    #remove punctuation
    lem_sen = re.sub('[^a-zA-Z]', ' ', lem_sen)
    #remove tags
    lem_sen = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", lem_sen)
    #remove special characters and digits
    lem_sen = re.sub("(\\d|\\W)+"," ", lem_sen)
    #remove synset
    lem_sen = re.sub("Synset"," ", lem_sen)
    
    return lem_sen

#def pruning(sentence):

def tfidf_lsa(descriptions):
    #TF-IDF vectorizer approach learned from https://towardsdatascience.com/k-means-clustering-
    #chardonnay-reviews-using-scikit-learn-nltk-9df3c59527f3. TF-IDF, TfidfVectorizer, uses a 
    #in-memory vocabulary (a python dict) to map the most frequent words to features indices 
    #and hence compute a word occurrence frequency (sparse) matrix. The word frequencies are 
    #then reweighted using the Inverse Document Frequency (IDF) vector collected feature-wise 
    #over the corpus. so this does not account for synonyms of the words
    #latent semantic analysis can also be used to reduce dimensionality and discover 
    #latent patterns in the data from https://scikit-learn.org/stable/auto_examples/
    #text/plot_document_clustering.html
    
    stop_words = stopwords.words('english')
    tfv = TfidfVectorizer(input='content', stop_words = stop_words, ngram_range = (1,1), 
                          max_df=0.5, min_df=2, use_idf=True)
    svd = TruncatedSVD(100) #For LSA, a value of 100 is recommended
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    #vectorize the data
    vec_desc = tfv.fit_transform(descriptions)
    #lsa the data
    vec_desc = lsa.fit_transform(vec_desc)
    
    return vec_desc

def kmeans(vectors, mini_batch=False, true_k = 20, predict = True):
    #setup kmeans clustering
    #k-means is optimizing a non-convex objective function, it will likely end up in a local optimum. Several runs 
    #with independent random init might be necessary to get a good convergence.
    if mini_batch == False:
        k_means = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=0)
    else:
        k_means = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, 
                                  batch_size=1000, verbose=0)
   
    #fit the data
    if predict == True:
        cluster = k_means.fit_predict(vectors)
    else:
        cluster = k_means.fit(vectors)
    
    return cluster

#not super worth using when a lot of the descriptors use the same words
def top_cluster_terms(descriptions, vec, true_k = 20, mini_batch=False):
    #build the model
    k_means = kmeans(vec, mini_batch=mini_batch, true_k = true_k, predict = False)
    stop_words = stopwords.words('english')
    tfv = TfidfVectorizer(input='content', stop_words = stop_words, ngram_range = (1,1), 
                          max_df=0.5, min_df=2, use_idf=True)
    order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
    
    tfv.fit_transform(descriptions)
    terms = tfv.get_feature_names()
    
    print("Top terms per cluster:")
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
    return

#create dataframe, set the column to work off of
df = pd.read_csv(r'C:\Users\fitzg\Documents\Python Scripts\NLP\wine_desc.csv')
column_name = 'description'

#basic nlp (clean + lematize + tfidf)
df = word_clean(df, column_name)
df[column_name] = df[column_name].apply(lambda sen: lemmatize_sentence(sen,use_syn_hyp = True))
desc = df[column_name].values
vec = tfidf_lsa(desc)

#kmeans on the tfidf data
cluster_list = kmeans(vec)
df['cluster_num'] = pd.Series(cluster_list)

############# to work on later
#comparing words based off their synonyms using wordnet
#learned from https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/

w1 = wordnet.synset('run.v.01') # v here denotes the tag verb 
w2 = wordnet.synset('sprint.v.01') 
print(w1.wup_similarity(w2)) 

#genism heard about from https://stackoverflow.com/questions/11798389/what-nlp-tools-to-use-to-match-phrases-having-similar-meaning-or-semantics
#genism further learned about here https://radimrehurek.com/gensim/models/word2vec.html
#not going to attempt gensim here

#https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
#set each word to it's wordnet word
print(train.path_similarity(horse))