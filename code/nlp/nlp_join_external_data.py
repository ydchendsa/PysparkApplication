# Databricks notebook source
!pip install findspark

# COMMAND ----------

import findspark
findspark.init()


# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import col
from pyspark.sql.functions import *


# COMMAND ----------

marvel_df = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/cleaned_df.csv')
marvel_df = marvel_df.drop(marvel_df.columns[[0, 1]],axis = 1)
marvel_df.head()

# COMMAND ----------

Stock_price_df = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/DIS_Stock.csv')
Stock_price_df.head()

# COMMAND ----------

Marvel_spark=spark.createDataFrame(marvel_df)
Marvel_spark.printSchema()


# COMMAND ----------

# Removing time for the colums of timestamp_init

Marvel_spark = Marvel_spark.withColumn("new_date", to_date(col("timestamp_init")))
#Marvel_spark.show()
Marvel_spark.toPandas()

# COMMAND ----------

#Checking data types
Stock_spark=spark.createDataFrame(Stock_price_df)
Stock_spark.printSchema()

# COMMAND ----------

#Changing string to timestamp
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
#Stock_spark = Stock_spark.withColumn("Date", regexp_replace('Date',"/", "-"))
Stock_spark = Stock_spark.withColumn("new_date", to_date(col("Date"), "MM/dd/yyyy")).\
                withColumn('new_date', regexp_replace('new_date', '00', '20'))
Stock_spark.show()

# COMMAND ----------

#Changing columns names
Stock_spark = Stock_spark.selectExpr("Volume as volume", "Open as open", "new_date as date")
Stock_spark.show()

# COMMAND ----------

#Join table
Marvel_Stock = Marvel_spark.join(Stock_spark, Marvel_spark["new_date"]== Stock_spark['date'], how ='left')
Marvel_Stock = Marvel_Stock.dropDuplicates()
Marvel_Stock.count()


# COMMAND ----------

marvel_stock = Marvel_Stock.toPandas()

# COMMAND ----------

marvel_stock

# COMMAND ----------

import os

CSV_DIR = os.path.join("data", "csv")
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "marvel_stock.csv")
marvel_stock.to_csv(fpath)
