# Databricks notebook source
!pip install findspark

# COMMAND ----------

import findspark
from pyspark.sql import SparkSession
findspark.init()

# COMMAND ----------

spark = SparkSession.builder \
        .appName("SparkNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.1") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv("/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/cleaned_df.csv")

# COMMAND ----------

final_df=spark.createDataFrame(df) 

# COMMAND ----------

import os

CSV_DIR = os.path.join("data", "csv")
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "cleaned_df.parquet")
final_df.write.parquet(fpath)
