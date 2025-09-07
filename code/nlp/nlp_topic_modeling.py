# Databricks notebook source
#import necessary packages for analysis
import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pyspark.sql.types import IntegerType
from datetime import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

!pip install gensim
!pip install nltk
!pip install wordcloud

# COMMAND ----------

import string
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# COMMAND ----------

## create a directory called data/plots and data/csv to save generated data
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

df =  pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/cleaned_df.csv')
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## WordCloud

# COMMAND ----------

# WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#mask = np.array(Image.open('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/plots/mask.png'))
stopwords = ["think","come","val","always","length","much","content","seeing","today","object","talking","going","thing","dtype","now","wondered","know","seen","throughout"] + list(STOPWORDS)
#stopwords.add("think","come","val","always","length","much","content")
wc = WordCloud(background_color="white", 
               stopwords=stopwords, 
               contour_width=3, 
               contour_color='red')
# generate word cloud
wc.generate(str(df["cleaned_submission"]))
# show
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

# COMMAND ----------

# store to file
plot1_fpath = os.path.join(PLOT_DIR, "wc.png")
plt.savefig(plot1_fpath)

# COMMAND ----------

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# COMMAND ----------

text = pd.DataFrame(df["cleaned_submission"])

# COMMAND ----------

def txt_processing(content):
    stemmer = WordNetLemmatizer()
    data = []
    for a, sentence in content.iterrows():
        sentence = content['cleaned_submission'][a]
        sentence = str(sentence)
        #print(sentence)
        for i in sent_tokenize(sentence):
            tmp = []
            for j in word_tokenize(i):
                if j not in string.punctuation and nltk.corpus.stopwords.words('english'):
                    if len(j) >= 3 and j != "the":
                        tmp.append(stemmer.lemmatize(j.lower()))       
        data.append(tmp) 
    return data

# COMMAND ----------

input = txt_processing(text)
input

# COMMAND ----------

# MAGIC %md
# MAGIC ## LDA Topic Modeling

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import gensim
from gensim import corpora

dictionary = corpora.Dictionary(input) 
corpus = [dictionary.doc2bow(text) for text in input]
ldamodel = LdaModel(corpus, num_topics=30, id2word = dictionary, passes=30,random_state = 1) 
print(ldamodel.print_topics(num_topics=30, num_words=20)) 

# COMMAND ----------

!pip install pyLDAvis

# COMMAND ----------

def perplexity(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)

def umass_coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=10,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=input, dictionary=dictionary, coherence='u_mass')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()

# COMMAND ----------

x = range(1,20)
y = [umass_coherence(i) for i in x]
plt.plot(x, y)
plt.xlabel('topic number')
plt.ylabel('coherence size')
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.title('topic vs Umass coherence')
plot1_fpath = os.path.join(PLOT_DIR, "plt1.png")
plt.savefig(plot1_fpath)
plt.show()

# COMMAND ----------

x = range(1,20)
z = [perplexity(i) for i in x]
plt.plot(x, z)
plt.xlabel('topic number')
plt.ylabel('perplexity size')
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.title('topic vs perplexity')
plot3_fpath = os.path.join(PLOT_DIR, "plt3.png")
plt.savefig(plot3_fpath)
plt.show()

# COMMAND ----------

import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel
import gensim
from gensim import corpora

pyLDAvis.enable_notebook()
lda = LdaModel(corpus, id2word=dictionary, num_topics=11, passes = 30,random_state=1)
vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
pyLDAvis.display(vis_data)

# COMMAND ----------

pyLDAvis.save_html(vis_data, 'lda_viz.html')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering by Word2Vec

# COMMAND ----------

import string
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

model = Word2Vec(input, vector_size=3, window=5, min_count=1, workers=4)
train_model = model.train(input, total_examples=model.corpus_count, epochs=model.epochs)

# COMMAND ----------

vector_dict = {}
for word in model.wv.index_to_key: 
    vector_dict[word] = model.wv.get_vector(word)

vec_df = pd.DataFrame(vector_dict).T
vec_df = vec_df.iloc[1:, :]
vec_df

# COMMAND ----------

SS_dist = []

values_for_k=range(1,15)


for k_val in values_for_k:
    #print(k_val)
    k_means = KMeans(n_clusters=k_val)
    k_means = k_means.fit(vec_df)
    SS_dist.append(k_means.inertia_)
    


plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plot4_fpath = os.path.join(PLOT_DIR, "plt4.png")
plt.savefig(plot4_fpath)
plt.show()

# COMMAND ----------

from sklearn.decomposition import PCA
pca = PCA(2)
data = pca.fit_transform(vec_df)

# COMMAND ----------

k_model = KMeans(n_clusters = 6, init = "k-means++")
kmeans_model = KMeans(n_clusters=6).fit(vec_df)
centers = np.array(kmeans_model.cluster_centers_)
label = k_model.fit_predict(data)
uniq = np.unique(label)
for i in uniq:
   plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
plt.title("K-means Results 2D When k=6")
plt.legend()

plot5_fpath = os.path.join(PLOT_DIR, "plt5.png")
plt.savefig(plot5_fpath)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Topic Mapping with Characters

# COMMAND ----------

import re
import plotly.express as px

# COMMAND ----------

topic_seeds = ["avenger","spiderman","shang-chi","black widow","eternals","venom","wanda vision","deadpool"]

# COMMAND ----------

search_pattern = re.compile(r"\b(?:#\w*)?(" + "|".join(topic_seeds) + r")\b",re.IGNORECASE)


# COMMAND ----------

df_new = pd.DataFrame()
df_new['content'] = []
df_new['topic'] = []
df_new['id'] = []
res_list = []
topic_list =[]
id_list = []

# COMMAND ----------

for index,row in df.iterrows():
        txt = row['cleaned_submission']
        match = re.search(search_pattern,str(txt))
        if match:
            res_list.append(row['cleaned_submission'])
            topic_list.append(match.group(0))
            id_list.append(row['Unnamed: 0'])
df_new['content'] = res_list
df_new['topic'] = topic_list
df_new['id'] = id_list

# COMMAND ----------

df_new=df_new.rename(columns={"content": "cleaned_submission"})
df_new

# COMMAND ----------

df_sub = df[["Unnamed: 0","score","comment_score","cleaned_submission","submission_sentiment","comment_sentiment","comment_length"]]
df_sub = df_sub.rename(columns={"Unnamed: 0": "id"})

# COMMAND ----------

df_new=df_new.rename(columns={"content": "cleaned_submission"})
df_merge = df_new.merge(df_sub, on='id', how='left')
df_merge = df_merge[['id','topic','score','comment_score','submission_sentiment','comment_length']]
df_merge

# COMMAND ----------

import numpy as np
table = pd.pivot_table(df_merge, values=['score','comment_score','comment_length'], index=['topic', 'submission_sentiment'],
                             aggfunc={'score': [min, max, np.mean],
                                      'comment_score': [min, max, np.mean],
                                      'comment_length':[min, max, np.mean]})
table = table.rename(columns={'score':'submission_score'})
table

# COMMAND ----------

df_plt=df_merge.pivot_table(index=['topic','submission_sentiment'],aggfunc = 'count')
df_plt = df_plt[["id"]]
df_plt=df_plt.rename(columns={"id": "counts"})
df_pivot = df_plt.reset_index()
df_pivot

# COMMAND ----------

import plotly.express as px

long_df = px.data.medals_long()

fig = px.bar(df_pivot, x="topic", y="counts", color="submission_sentiment", title="Stack Plot for Character Topic Counts among Main Posts Sentiment")
plot6_fpath = os.path.join(PLOT_DIR, "fig6.html")
fig.write_html(plot6_fpath)
fig.show()

