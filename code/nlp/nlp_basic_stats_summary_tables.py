# Databricks notebook source
pip install -U kaleido

# COMMAND ----------

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

data = pd.read_csv("/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/marvel_stock.csv")
data=spark.createDataFrame(data)
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adding Dummy 1: To see whether it talks about movie or not (is_Movie)

# COMMAND ----------

#to highlight posts contain information about if it is movie 
data=data.withColumn("is_movie",F.when((F.col("cleaned_submission").\
    rlike(r'(?i)movie|(?i)film|(?i)cinema|(?i)theater|(?i)ticket|(?i)screen')) ,'1').otherwise('0'))
data.take(1)

# COMMAND ----------

data=data.withColumn("in_topic",F.when((F.col("cleaned_comment").\
    rlike(r'(?i)avenger|(?i)spiderman|(?i)shang|(?i)widow|(?i)eternals|(?i)venom|(?i)wanda|(?i)vision|(?i)deadpool')) ,'1').otherwise('0'))
data.take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basic Statistic: Text Length Distribution

# COMMAND ----------

import os
import kaleido

submission_length = data.groupby('submission_length').count()
submission_length = submission_length.withColumn("submission_length",submission_length.submission_length.cast('int'))
submission_length = submission_length.orderBy(F.col('submission_length').asc())

#getting comment score
comment_length = data.groupby('comment_length').count()
comment_length = comment_length.withColumn("comment_length",comment_length.comment_length.cast('int'))
comment_length = comment_length.orderBy(F.col('comment_length').asc())

submission_length = submission_length.toPandas()
comment_length = comment_length.toPandas()

fig = make_subplots(subplot_titles=("Figure 2.Submission Length Distribution", "Comment Length Distribution"),rows=1, cols=2)

fig.add_trace(
   go.Histogram(name='submission_length',x=submission_length['submission_length'],y=submission_length['count']),
    row=1, col=1)
fig.add_trace(
    go.Histogram(name='comment_length',x=comment_length['comment_length'],y=comment_length['count']),
    row=1, col=2)

fig.update_xaxes(title_text="Length", row=1, col=1)
fig.update_xaxes(title_text="Length", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.show()

# COMMAND ----------

PLOT_DIR = os.path.join("data", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
plot2_fpath = os.path.join(PLOT_DIR, "fig2.html")
fig.write_html(plot2_fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA: External Data: Time Series of Stock Price

# COMMAND ----------

import plotly.graph_objects as go
data1 = data.toPandas()
data1 = data1[['open','date']]
data1 = data1.dropna()
data1 = data1.sort_values(by='date')
data1 = data1.drop_duplicates()
data1
fig = go.Figure([go.Scatter(x=data1['date'],y=data1['open'])])
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Stock Price ($)")
fig.update_layout(title='Time Series Plot of Disney Stock Price')
fig.show()

# COMMAND ----------

PLOT_DIR = os.path.join("data", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
plot7_fpath = os.path.join(PLOT_DIR, "fig7.html")
fig.write_html(plot7_fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Table 1: Sentiment Distribution by Day of Week

# COMMAND ----------

Sentiment_Day = data.groupBy("submission_weekday").pivot('submission_sentiment').count()
Sentiment_Day = Sentiment_Day.toPandas()
Sentiment_Day['total'] = Sentiment_Day[['negative','neutral','positive']].sum(axis=1)
Sentiment_Day = Sentiment_Day.reindex([0,1,6,4,2,5,3])
Sentiment_Day

# COMMAND ----------

data = data.withColumn("comment_weekday",F.date_format('timestamp_comment', 'E'))

Sentiment_Day = data.groupBy("comment_weekday").pivot('comment_sentiment').count()
Sentiment_Day = Sentiment_Day.toPandas()
Sentiment_Day['total'] = Sentiment_Day[['negative','neutral','positive']].sum(axis=1)
Sentiment_Day = Sentiment_Day.reindex([0,1,6,4,2,5,3])
Sentiment_Day

# COMMAND ----------

data = data.withColumn("comment_hour",F.date_format('timestamp_comment', 'H'))
sentiment_hour = data.groupBy("comment_hour").pivot('comment_sentiment').count()
sentiment_hour = sentiment_hour.toPandas()
sentiment_hour["pos_rate"] = sentiment_hour.positive / (sentiment_hour.negative + sentiment_hour.neutral + sentiment_hour.positive)
sentiment_hour["comment_hour"] = pd.to_numeric(sentiment_hour["comment_hour"])
sentiment_hour = sentiment_hour.sort_values("comment_hour", ascending=True).reset_index(drop=True)
sentiment_hour

# COMMAND ----------

# Values for the x axis
angles = np.linspace(0.0, 2 * np.pi - 0.05, len(sentiment_hour), endpoint=False)
length = sentiment_hour["pos_rate"].values
region = sentiment_hour["comment_hour"].values

# Group of Positive
sentiment_hour['n'] = pd.DataFrame([1 if i<=6 else 2 if i<=12 else 3 if i<=18 else 4 for i in sentiment_hour.index])
TRACKS_N = sentiment_hour["n"].values

import matplotlib as mpl

GREY12 = "#1f1f1f"
plt.rc("axes", unicode_minus=False)
COLORS = ["#5a336e","#FFD55F","#85D7FF","#c3727c"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("my colors", COLORS, N=256)
norm = mpl.colors.Normalize(vmin=TRACKS_N.min(), vmax=TRACKS_N.max())
COLORS = cmap(norm(TRACKS_N))

from textwrap import wrap

fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_theta_offset(1.2 * np.pi / 2)
ax.set_ylim(bottom=0.5, top=0.7)

ax.bar(x = angles, height=length-0.5, bottom=0.5, 
       color=COLORS, alpha=0.9, width=0.25, zorder=11)
ax.vlines(angles, 0.5, 0.7, color=GREY12, ls=(0, (4, 4)), zorder=12)
region = ["\n".join(wrap(r, 5, break_long_words=False)) for r in region.astype('str')]

ax.set_xticks(angles)
ax.set_xticklabels(region, size=13);
ax.set_title(label = "Radar Plot of Positive Sentiment Rate for Different Time of Day", fontsize = 20, fontweight = "bold")

PLOT_DIR = os.path.join("data", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
plot3_fpath = os.path.join(PLOT_DIR, "fig3.png")
fig.savefig(plot3_fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Table 2: Sentiment analysis and the positive rate on comments by different time interval of the day 

# COMMAND ----------

#data = Seq(("night",6),("morning",6),("afternoon",6),("evening",5))
#sentiment_hour.withColumn("time_interval",data)
sentiment_hour = spark.createDataFrame(sentiment_hour)
sentiment_interval= sentiment_hour.groupBy("n").agg(F.sum('negative').alias('negative'),
                                               F.sum('neutral').alias('neutral'),
                                               F.sum('positive').alias('positive'),
                                               F.avg('pos_rate').alias('pos_rate'))
sentiment_interval = sentiment_interval.toPandas()

# COMMAND ----------

sentiment_interval['time_interval'] = ['night','morning','afternoon','evening']
sentiment_interval = sentiment_interval.drop('n',axis=1)
sentiment_interval = sentiment_interval.set_index('time_interval')
sentiment_interval['total'] = sentiment_interval[['negative','neutral','positive']].sum(axis=1)
sentiment_interval

# COMMAND ----------

import os
marvel_stock = data.toPandas()
CSV_DIR = os.path.join("data", "csv")
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "marvel_stock_full.csv")
marvel_stock.to_csv(fpath)

# COMMAND ----------

marvel_stock
