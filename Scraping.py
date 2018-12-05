# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:40:58 2018

@author: asimon
"""

# import libraries
import gensim
from gensim import corpora, models
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import os
import nltk.stem as stemmer
from pprint import pprint
import itertools

import gensim
import pandas as pd
import os
import collections
import smart_open
from nltk.tokenize import RegexpTokenizer
import gensim.models.doc2vec
import multiprocessing
import numpy as np
from scipy.sparse import csr_matrix
import nltk.tokenize 
from gensim.models.doc2vec import TaggedDocument
import pickle
import seaborn as sns
from string import digits
from collections import namedtuple
import math
import matplotlib 
from datetime import timedelta
import matplotlib.pyplot as plt
import re
import pickle
import sys
import re
import csv
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from iwnlp.iwnlp_wrapper import IWNLPWrapper
import sys
import numpy as np
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
import csv
from sklearn.externals import joblib
from string import digits
import bz2
import pyLDAvis
import pyLDAvis.gensim
from collections import Counter
from googletrans import Translator
import plotly
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
plotly.tools.set_credentials_file(username='alphonse.m.simon', api_key='tL4FDjpWNTU1xUf5sXEl')

stemmer = PorterStemmer()
np.random.seed(2018)
nltk.download('wordnet')

path_link = "D:/Users/asimon/Desktop/Urban- Analysis on Urban"
os.chdir(path_link)


centers = tuple([ ("https://www.urban.org/urban-wire/policy-center/center-international-development-and-governance","IDG") , 
            ("https://www.urban.org/urban-wire/policy-center/center-labor-human-services-and-population", "LHP"), 
            ("https://www.urban.org/urban-wire/policy-center/center-nonprofits-and-philanthropy", "CNP"), 
            ("https://www.urban.org/urban-wire/policy-center/health-policy-center", "HPC"), 
            ("https://www.urban.org/urban-wire/policy-center/housing-finance-policy-center", "HFPC"), 
            ("https://www.urban.org/urban-wire/policy-center/income-and-benefits-policy-center", "IBP"), 
            ("https://www.urban.org/urban-wire/policy-center/justice-policy-center", "JPC"), 
            ("https://www.urban.org/urban-wire/policy-center/metropolitan-housing-and-communities-policy-center", "METRO"),
            ("https://www.urban.org/urban-wire/policy-center/research-action-lab", "RAL"), 
            ("https://www.urban.org/urban-wire/policy-center/urban-brookings-tax-policy-center", "TPC")])

df = pd.DataFrame(columns=['Center','Url', 'Texts'])


def b_scrape(url_link, center):
    global df
    url_list = []
    blog_posts = []
    urban = "https://www.urban.org"
    for i in range(1,100):
        url_list.append(url_link + "?page=" + str(i))
    for x in url_list: 
        page = requests.get(x)
        soup = BeautifulSoup(page.content, 'html.parser')
        for a in soup.find_all('a', href=True): 
            temp = a['href']
            if temp[0:12] == "/urban-wire/": 
                blog_posts.append(urban + a['href'])
            else: 
                continue
    blog_posts = list(set(blog_posts))
    for x in blog_posts: 
        if any( [x[0:39] == "https://www.urban.org/urban-wire/topic/", x[0:37] == "https://www.urban.org/urban-wire/rss?"] ):
             blog_posts.remove(x) 
        else: 
            continue
    for x in blog_posts:
        try: 
            page = requests.get(x)
            soup = BeautifulSoup(page.content, 'html.parser')
            body_text = ""
            for y in soup.find_all('p'):
                body_text = body_text + " " + y.text
            for strong_tag in soup.find_all('strong'):
                body_text = body_text + " " + strong_tag.text + " " + str(strong_tag.next_sibling.string)
            df2 = pd.DataFrame([[center, x, body_text]], columns=['Center','Url', 'Texts'])
            df = df.append(df2)
        except AttributeError:
            continue
    return df
    

for x in centers: 
    b_scrape(x[0], x[1])
    
df.to_pickle("./Text.pkl")
    

df = pd.read_pickle("./Text.pkl")
df = df.reset_index()
remove = "Your support helps Urban scholars continue to deliver evidence that can elevate debate, transform communities, and improve lives."
also_remove = "SIGN UP FOR OUR NEWSLETTERS"


df['Texts'] = df['Texts'].str.replace(remove, "", regex=True)
df['Texts'] = df['Texts'].str.replace(also_remove, "", regex=True)



def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result



df['clean_texts'] = df['Texts'].map(preprocess)



training_set = [[]]

for x in df['clean_texts']:
    training_set.append(x)
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
docs = []
for i, text in enumerate(training_set):
    tags = [i]
    docs.append(analyzedDocument(text, tags))

# train word2vec model using gensim
cores = multiprocessing.cpu_count()
model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=1, iter=200)


model.build_vocab(docs)


model_buyer = model.train(docs, total_examples=model.corpus_count, epochs=model.iter)


obj = {}
for x in ( df['clean_texts']): 
    for z in x:
        obj[str(z)] = model.wv[z] 
v_list_g = ["Word"]
v_cluster = []
for i in range(1,100): 
    v_list_g.append("v_" + str(i))
    v_cluster.append("v_" + str(i))

# Now we create a datafrmae that works like this: Word, Vector_ITEM_1, Vector_ITEM_2 ... Vector_ITEM_100
df_new = pd.DataFrame( columns=v_list_g)


cluster  = KMeans(n_clusters = 8)



for x in df['clean_texts']: 
    for z in x:
        word = z
        v_list = [word]
        for i in range(1,100):
            v = [obj[str(z)][i]][0]
            v_list.append(v)
        df2 = pd.DataFrame([v_list], columns=v_list_g)
        df = df.append(df2)


df['cluster'] = cluster.fit_predict(df[df.columns[1:101]])
vectors = df.columns[1:101]


centroids = cluster.cluster_centers_

df = df.reset_index(drop = True)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
yes = pca.fit_transform(df[v_cluster])
principalDf = pd.DataFrame(data = principalComponents
           , columns = ['principal component 1', 'principal component 2'])

df_word = df['Word']

finalDf = pd.concat([principalDf, df_word], axis = 1)

ax1 = finalDf.plot.scatter(x='principal component 1',
                       y='principal component 2',
                       c='DarkBlue')


import plotly.plotly as py
import plotly.graph_objs as go
import random
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go


c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
# Create a trace
trace = go.Scatter(
    x = finalDf['principal component 1'],
    y = finalDf['principal component 2'],
    mode = 'markers', 
    marker= dict(size= 14,
                 line= dict(width=1),
                 opacity= 0.3
                ),
        text= finalDf['Word'])



data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename ='basic-scatter')





def cosinesim(cosine, df):
    corpus = []
    corpus = [cosine['week1'], cosine['week2'], cosine['week3'],cosine['week4'], cosine['week5'],cosine['week6'], cosine['week7'],cosine['week8'], cosine['week9'],cosine['week10']]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    corpus_cosine= squareform(pdist(X.toarray(), 'cosine'))
    name = "corpus_cosine_" + str(df) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(df_doc, f) 


# Pre-Processing Main Steps:
#  
# 1. First we have code that elimiates special characters (with the exception of spaces), numbers and lowers all the letters
# 2. Then we need to get our data into four formats: 
#     1. To train our gensim we need a list of sentences (order does not matter for the items in the list)
#          = [ ["This", "is", "an", "example"], ["This", "is", "another", "example"]
#      2. To test the model aganist our sample we need a list for each week:
#          = list['week1'] = ["This is our example 1", "Then we need another example 1"]
#          = list['week2'] = ["This is our example 2", "This is our example 2"]
#      3. To run our cosine similarity we need one list that combines the list for every week
#          = ["This is every message in week1 this is every message in week1", "This is every message in week2"]
#      4. To our sentiment we need a list of words for each week
#          = list['week1'] = [ "this", "is", "our", "example"]
#      5. To train our LDA we need to have a list of sentances of our data
#          = list = ["This is the first sentance of week1", "This is first sentance of week2"]

    
    
    
def text_clean(df, label):
    tokenizer = RegexpTokenizer(r'\w+')
    obj = {}
    obj_list = []
    gensim_train = []
    newlines = {}
    texts = {}
    for i in range(1,11):
        obj['week' + str(i)] = df['message'][df['week'].isin([i])].tolist()
        for num in obj['week' + str(i)]:
            if type(num) != float:
                num = num.replace("_","")
        # Here I am deleting any numbers that appear
        newlines['week' + str(i)] = []
        for line in obj['week' + str(i)]:
            newlines['week' + str(i)].append(line.translate({ord(k): None for k in digits}))
        texts['week' + str(i)] = []
        for x in newlines['week' + str(i)]:
            tokens = []
            x = u"".join(x)
            raw = x.lower()
            tokens_temp = tokenizer.tokenize(raw)
            for a in tokens_temp:
                if a != 'der' or 'den':
                    try:
                        temp = lemmtizer.lemmatize_plain(a, ignore_case=True)[0]
                        temp = temp.lower()
                        tokens.append(temp)
                    except:
                        tokens.append(a)
                tokens = [z for z in tokens if not z in German]
                texts['week' + str(i)].append(tokens)
                gensim_train.append(tokens)
    texts_name = "texts_" + str(label) + ".pkl"
    gensim_name = "gensim_train_" + str(label) + ".pkl"
    with open(texts_name, 'wb') as f:
        pickle.dump(texts, f)
    with open(gensim_name, 'wb') as f:
        pickle.dump(gensim_train, f)
    gensim_test = {}
    for i in range(1,11):
        gensim_test['week' + str(i)] = []
        for x in texts['week' + str(i)]:
            x = "".join(x)
            gensim_test['week' + str(i)].append(x)
    name = "gensim_test_" + str(label) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(gensim_test, f)



def intertools(texts, label):
    cosine = {}
    sa = {}
    for i in range(1,11):
        sa['week' + str(i)] = list(itertools.chain.from_iterable(texts['week' + str(i)]))
        cosine['week' + str(i)] = u" ".join(sa['week' + str(i)])
        return sa, cosine
    name = "sa" + str(label) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(sa, f)
    name = "cosine" + str(label) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(cosine, f)
        
def doc2vec(gensim_train, gensim_test, df):
    train_file_buyer = gensim_train
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    gensim_train_buyer = []
    for i in range(1,11):
        gensim_train = gensim_test['week' + str(i)] + gensim_train
    for i, text in enumerate(gensim_train):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    cores = multiprocessing.cpu_count()
    model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=2, iter=200)
    model.build_vocab(docs)
    model_buyer = model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    names = []
    for i in range(1,101):
        names.append('v_' + str(i))
    average_model = {}
    modelrun = {}
    df_doc = {}
    for x in range(1,11):
        model1 = np.zeros((1,100))
        df_doc['week' + str(x)] = pd.DataFrame(model1, columns = names)
        df_doc['week' + str(x)]['message'] = " "
        for y in texts_buyer['week' + str(x)]:
            d = len(gensim_test['week' + str(x)])
            model2 = model.infer_vector(y)
            model2 = np.reshape(model2, (1,100))
            df2 = pd.DataFrame(model2, columns = names)
            df2['message'] = " ".join(y)
            df_doc['week' + str(x)] = pd.concat([df_doc['week' + str(x)], df2])
            model1 = np.add(model1, model2)
            average_model['week' + str(x)] = np.divide(model1,d)
    column = {}
    for i in range(1,11):
        column['column_' + str(i)] = []
        for num in range(1,11):
            dst = [distance.euclidean(average_model['week' + str(i)],average_model['week' + str(num)])]
            column['column_' + str(i)] = column['column_' + str(i)] + dst  
    Matrix = {}
    for i in range(1,11):
        Matrix['Matrix' + str(i)] = np.array(column['column_' + str(i)])
    EU = np.matrix((Matrix['Matrix1'], Matrix['Matrix2'],Matrix['Matrix3'], Matrix['Matrix4'], 
                    Matrix['Matrix5'], Matrix['Matrix6'], Matrix['Matrix7'],
                    Matrix['Matrix8'], Matrix['Matrix9'], Matrix['Matrix10']))
    name = "EU_" + str(df) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(sa, f)
    name = "df_doc_" + str(df) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(df_doc, f)
    del model



def cosinesim(cosine, df):
    corpus = []
    corpus = [cosine['week1'], cosine['week2'], cosine['week3'],cosine['week4'], cosine['week5'],cosine['week6'], cosine['week7'],cosine['week8'], cosine['week9'],cosine['week10']]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    corpus_cosine= squareform(pdist(X.toarray(), 'cosine'))
    name = "corpus_cosine_" + str(df) + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(df_doc, f) 




def sentiws(texts, df):
    xls = pd.ExcelFile("Simon_SENTIWS.xlsx")
    sentiws1 = pd.read_excel(xls, 'Sheet1')
    sentiws2 = pd.read_excel(xls, 'Sheet3')
    frames = [sentiws1, sentiws2]
    sentiws = pd.concat(frames)
    Polar = {}
    for i in range(1,38):
        Polar['inflection' + str(i)] = {}
        Polar['inflection' + str(i)] = sentiws.set_index('Inflection' + str(i))['Polairty'].to_dict()
    def merge_two_dicts(x, y):
        z = x.copy() # start with x's keys and values
        z.update(y) # modifies z with y's keys and values & returns None
        return z
    z = merge_two_dicts(Polar['inflection1'], Polar['inflection2'])
    for i in range(3,37):
        z.update(Polar['inflection' + str(i)])
    z_list = z.keys()
    Polar = {}
    for i in range(1,11):
        Polar['week' + str(i)] = []
        for x in texts['week' + str(i)]:
            messagep = 0
            for word in x:
                try:
                    messagep = messagep + (z[word])
                except:
                    messagep = messagep
            Polar['week' + str(i)].append(messagep)
    Polar_arr = {}
    for i in range(1,11):
        Polar_arr['week' + str(i)] = np.empty
        Polar_arr['week' + str(i)] = np.asarray(Polar_seller['week' + str(i)])
    Sentiment_Avg = []
    Sentiment_SD = []
    for i in range(1,11):
        Sentiment_Avg.append(np.mean(Polar_arr['week' + str(i)]))
        Sentiment_SD.append(np.std(Polar_arr['week' + str(i)])/ math.sqrt(len(texts['week' + str(i)])))
    x = [1,2,3,4,5,6,7,8,9,10]
    y = Sentiment_Avg
    e = Sentiment_SD
    return y, e



def cluster(df_docs, df):
    for i in range(1,11):
        df_doc['week' + str(i)]['week'] = i
    for i in range(2,11):
        df_cluster = pd.concat([df_doc['week1'], df_doc['week' + str(i)]])
    cluster  = KMeans(n_clusters = 3)
    df_cluster['cluster'] = cluster.fit_predict(df_cluster[df_cluster.columns[0:100]])
    centroids = cluster.cluster_centers_
    vlist = []
    for i in range(1,101):
        vlist.append('v_' + str(i))
    df_cluster = df_cluster.reset_index(drop = True)
    for index, row in df_cluster.iterrows():
        df_cluster.set_value(index, 'class', distance.euclidean(df_cluster[vlist].values[index], centroids[df_cluster['cluster'][index]]))
    df_cluster_c1 = df_cluster.loc[df_cluster['cluster'] == 0] 
    df_cluster_c2 = df_cluster.loc[df_cluster['cluster'] == 1] 
    df_cluster_c3 = df_cluster.loc[df_cluster['cluster'] == 2] 
    df_cluster_c1.sort_values('class', ascending=False)
    df_cluster_c2.sort_values('class', ascending=False)
    df_cluster_c3.sort_values('class', ascending=False)
    table1 =  df_cluster_c1['message'].head(20).tolist()
    table2 = df_cluster_c2['message'].head(20).tolist()
    table3 = df_cluster_c3['message'].head(20).tolist()
    translator = Translator()
    cluster1 = []
    for x in table1: 
        cluster1.append(translator.translate(x, dest = 'en').text)
    cluster2 = []
    for x in table2: 
        cluster2.append(translator.translate(x, dest = 'en').text)
    cluster3 = []
    for x in table3: 
        cluster3.append(translator.translate(x, dest = 'en').text)
    tab = tt.Texttable()
    headings = ['Cluster1','Cluster2', 'Cluster3']
    tab.header(headings)
    for row in zip(cluster1, cluster2, cluster3):
        tab.add_row(row)
    s = tab.draw()
    return s
    emp = pd.get_dummies(df_all_seller['cluster'])
    dates = pd.DataFrame(df_all_seller['week'])
    cleaned = dates.join(temp)
    frequencies = cleaned.groupby('week').mean()
    frequencies.plot()

