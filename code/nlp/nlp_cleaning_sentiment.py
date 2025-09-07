# Databricks notebook source
!pip install findspark

# COMMAND ----------

import findspark
findspark.init()

# COMMAND ----------

!pip install spark-nlp==4.2.1 --force
!pip install sparknlp

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

spark = SparkSession.builder \
        .appName("SparkNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.1") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read in dataset that collected for our topic (Marvel Universe)

# COMMAND ----------

marvel_df = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/all_marval_df.csv')
marvel_df.head()

# COMMAND ----------

sparkDF=spark.createDataFrame(marvel_df)
sparkDF.head()
sparkDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Clean data using Spark NLP + Adding Sentiment to text
# MAGIC clean title, content and comments, and add sentiment analysis to text

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("content")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setMinLength(2)\
    .setMaxLength(12)

stop_words = StopWordsCleaner.pretrained("stopwords_en", "en")\
    .setInputCols(["token"])\
    .setOutputCol("cleanTokens")

lemmatizer = LemmatizerModel.pretrained()\
    .setInputCols(["cleanTokens"])\
    .setOutputCol("lemma")

stop_words2 = StopWordsCleaner.pretrained("stopwords_en", "en")\
    .setInputCols(["lemma"])\
    .setOutputCol("cleanTokens2")

normalizer = Normalizer()\
    .setInputCols(["cleanTokens2"])\
    .setOutputCol("normalized")\
    .setLowercase(True)\

tokenAssembler = TokenAssembler()\
    .setInputCols(["document","normalized"])\
    .setOutputCol("cleaned")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document","cleaned"])\
 .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          sentenceDetector, #step 1, split documents into sentences
          tokenizer,        #step 2-3, tokenize sentences for further analysis, and set words to be within length 2-12
          stop_words,       #step 3, get rid of stop words
          lemmatizer,       #step 4, lemmatization
          stop_words2,      #step 5, make sure that there's no stop words left
          normalizer,       #step 6-8, get rid of punctuations, numbers and set to lowercases
          tokenAssembler,
          use,
          sentimentdl       #step 9, compute sentiment for the text
      ])



# COMMAND ----------

piplineModel = nlpPipeline.fit(sparkDF)
result = piplineModel.transform(sparkDF)

# COMMAND ----------

result_df = result.withColumn('cleaned_submission', F.explode('cleaned.result'))
result_df = result_df.withColumn('submission_words', F.col("normalized").result)
result_df = result_df.withColumn('submission_sentiment', F.explode('sentiment.result'))
result_df = result_df.drop('document','sentence','token','cleanTokens','lemma','cleanTokens2','normalized','cleaned','sentence_embeddings','sentiment')
result_df.show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC similar results are also applied to comments and title

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("comment_body")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setMinLength(2)\
    .setMaxLength(12)

stop_words = StopWordsCleaner.pretrained("stopwords_en", "en")\
    .setInputCols(["token"])\
    .setOutputCol("cleanTokens")

lemmatizer = LemmatizerModel.pretrained()\
    .setInputCols(["cleanTokens"])\
    .setOutputCol("lemma")

stop_words2 = StopWordsCleaner.pretrained("stopwords_en", "en")\
    .setInputCols(["lemma"])\
    .setOutputCol("cleanTokens2")

normalizer = Normalizer()\
    .setInputCols(["cleanTokens2"])\
    .setOutputCol("normalized")\
    .setLowercase(True)\

tokenAssembler = TokenAssembler()\
    .setInputCols(["document","normalized"])\
    .setOutputCol("cleaned")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document","cleaned"])\
 .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          sentenceDetector, #step 1, split documents into sentences
          tokenizer,        #step 2-3, tokenize sentences for further analysis, and set words to be within length 2-12
          stop_words,       #step 3, get rid of stop words
          lemmatizer,       #step 4, lemmatization
          stop_words2,      #step 5, make sure that there's no stop words left
          normalizer,       #step 6-8, get rid of punctuations, numbers and set to lowercases
          tokenAssembler,
          use,
          sentimentdl       #step 9, compute sentiment for the text
      ])

# COMMAND ----------

piplineModel = nlpPipeline.fit(result_df)
result_df1 = piplineModel.transform(result_df)

# COMMAND ----------

result_df1 = result_df1.withColumn('cleaned_comment', F.explode('cleaned.result'))
result_df1 = result_df1.withColumn('comment_words', F.col("normalized").result)
result_df1 = result_df1.withColumn('comment_sentiment', F.explode('sentiment.result'))
result_df1 = result_df1.drop('document','sentence','token','cleanTokens','lemma','cleanTokens2','normalized','cleaned','sentence_embeddings','sentiment')
result_df1.show(1)

# COMMAND ----------

final_df = result_df1.dropDuplicates()

# COMMAND ----------


