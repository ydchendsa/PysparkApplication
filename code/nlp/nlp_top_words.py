# Databricks notebook source
!pip install nltk

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import nltk

# COMMAND ----------

#pip install -U kaleido

# COMMAND ----------

marvel_df = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/cleaned_df.csv')
marvel_df.head()

# COMMAND ----------

text = pd.DataFrame(marvel_df["cleaned_submission"])

# COMMAND ----------

text

# COMMAND ----------

!pip install nltk

# COMMAND ----------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# COMMAND ----------

import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


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
# MAGIC #### Count Vectorizer

# COMMAND ----------

import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# COMMAND ----------

## using the count vectorizer
stop_words = set(stopwords.words('english'))
count = CountVectorizer(analyzer=lambda x:[j for j in x if j not in stop_words])
word_count=count.fit_transform(input)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Word Frequency

# COMMAND ----------

import operator
word_count_name = count.get_feature_names()
feature_count = word_count.toarray().sum(axis=0)
word_count_dict = dict(zip(word_count_name,feature_count))
sorted_count = sorted(word_count_dict.items(), key=operator.itemgetter(1),reverse=True)
sorted_count

# COMMAND ----------

# MAGIC %md
# MAGIC ### TF-IDF

# COMMAND ----------

def dummy_fun(doc):
    return doc
tfidf = TfidfVectorizer(analyzer='word',tokenizer=dummy_fun,preprocessor=dummy_fun,token_pattern=None)
tfidf.fit(input)
tfidf.vocabulary_

# COMMAND ----------

## using the count vectorizer
stop_words = set(stopwords.words('english'))
count = CountVectorizer(analyzer=lambda x:[j for j in x if j not in stop_words])
word_count=count.fit_transform(input)

# COMMAND ----------

## use idf transformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count.get_feature_names(),columns=["idf"])

# COMMAND ----------

df_idf.sort_values(by=['idf'])

# COMMAND ----------

## act to TF-IDF transformation
tfidf_vec=tfidf_transformer.transform(word_count)
feature_names = count.get_feature_names()
first_document_vector=tfidf_vec[1]
df_tfidf= pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
sort_tfidf = df_tfidf.sort_values(by=["tfidf"],ascending=False)
sort_tfidf

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot

# COMMAND ----------

top10_df = pd.DataFrame(sorted_count[:10])
top10_df.columns = ['word','counts']
top10_df

# COMMAND ----------

import plotly.express as px

fig = px.bar(top10_df, x='word', y='counts',text='counts',title="Bar Plot for Top 10 Words")
fig.update_traces(marker_color='#a6cfe4')
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# COMMAND ----------

import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

top10_fpath = os.path.join(PLOT_DIR, "top10.html")
fig.write_html(top10_fpath)
fig.show()

# COMMAND ----------

top_tfidf = sort_tfidf[:10].reset_index()
top_tfidf = top_tfidf.rename(columns={'index':'word','tfidf':'score'})
top_tfidf

# COMMAND ----------

import plotly.express as px

fig = px.bar(top_tfidf, x='word', y='score',text='score',title="Bar Plot for Top 10 Words by TF-IDF Score" )
fig.update_traces(marker_color='#a6cfe4')
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# COMMAND ----------

tfidf_fpath = os.path.join(PLOT_DIR, "tfidf.html")
fig.write_html(tfidf_fpath)
fig.show()
