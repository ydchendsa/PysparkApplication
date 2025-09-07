# Databricks notebook source
pip install findspark

# COMMAND ----------

import findspark
findspark.init()

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.functions import col, lit
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("popularity").getOrCreate()

# COMMAND ----------

spark

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

marvel = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/marvel_stock_full.csv')

# COMMAND ----------

marvel

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculating stock trend

# COMMAND ----------

stock = marvel[['open','date']]

# COMMAND ----------

stock = stock.drop_duplicates()
stock = stock.dropna()
stock = stock.sort_values(by='date')
stock

# COMMAND ----------

i_list = []
j_list = []
for i in stock['open']:
    i_list.append(i)
for j in stock['open'][1:]:
    j_list.append(j)

i_list = i_list[:-1]

# COMMAND ----------

num_list = []
for i in range(0,410):
    num = j_list[i] - i_list[i]
    num_list.append(num)

# COMMAND ----------

stock = stock[1:]
stock['stock_change'] = num_list

# COMMAND ----------

conditions = [
    (stock['stock_change'] < 0),
    (stock['stock_change'] >= 0)
]
     
values = ['down','up']

stock['stock_trend'] = np.select(conditions, values)

# COMMAND ----------

stock

# COMMAND ----------

stock = stock[['date','stock_trend']]

# COMMAND ----------

marvel = pd.merge(marvel, stock, how='inner')

# COMMAND ----------

marvel

# COMMAND ----------

marvel=spark.createDataFrame(marvel) 
marvel.printSchema()

# COMMAND ----------

marvel = marvel.drop("Unnamed: 0","Unnamed: 0.1")

# COMMAND ----------

# MAGIC %md
# MAGIC #### selecting variables and building the analytical dataset

# COMMAND ----------

#get the selected columns
col = ['content','score','comment_score','submission_weekday','submission_length','comment_length','submission_sentiment','comment_sentiment','open','volume','stock_trend','date']
analytical = marvel.select([i for i in col])
analytical.show(1)

# COMMAND ----------

analytical = analytical.na.drop()
analytical.count()

# COMMAND ----------

submission = analytical.groupBy('date').pivot('submission_sentiment').count()
submission = submission.fillna(0)

# COMMAND ----------

analytical = analytical.join(submission, 'date')

# COMMAND ----------

analytical.show()

# COMMAND ----------

ml = analytical.groupBy("date","comment_score","submission_weekday","volume","stock_trend","negative","neutral","positive","comment_sentiment").avg("submission_length","score")

# COMMAND ----------

ml.show()

# COMMAND ----------

ml=ml.withColumnRenamed("avg(submission_length)","average_submission_length").\
    withColumnRenamed("avg(score)","average_submission_score").\
    withColumnRenamed("negative","neg_sentiment_count").\
    withColumnRenamed("neutral","neu_sentiment_count").\
    withColumnRenamed("positive","pos_sentiment_count")

# COMMAND ----------

ml.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### spliting train and validation set

# COMMAND ----------

ml,validation_data = ml.randomSplit([0.8, 0.2], seed=529)

# COMMAND ----------

# MAGIC %md
# MAGIC #### utilizing *Pipeline* to conduct classfication

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, Model

# COMMAND ----------

stringIndexer_trend = StringIndexer(inputCol="stock_trend", outputCol="stock_trend_ix")
stringIndexer_weekday = StringIndexer(inputCol="submission_weekday", outputCol="submission_weekday_ix")
stringIndexer_sent = StringIndexer(inputCol="comment_sentiment", outputCol="comment_sentiment_ix")
onehot_weekday = OneHotEncoder(inputCol="submission_weekday_ix", outputCol="submission_weekday_vec")
onehot_sent = OneHotEncoder(inputCol="comment_sentiment_ix", outputCol="comment_sentiment_vec")

vectorAssembler_features = VectorAssembler(
    inputCols=['comment_score','submission_weekday_vec','volume','neg_sentiment_count','neu_sentiment_count','pos_sentiment_count','comment_sentiment_ix','average_submission_length','average_submission_score'], 
    outputCol= "features")

# COMMAND ----------

rf1 = RandomForestClassifier(labelCol="stock_trend_ix", featuresCol="features", numTrees=100, maxDepth=5, seed=529)
rf2 = RandomForestClassifier(labelCol="stock_trend_ix", featuresCol="features", numTrees=100, maxDepth=10, seed=529)
rf3 = RandomForestClassifier(labelCol="stock_trend_ix", featuresCol="features", numTrees=500, maxDepth=10, seed=529)

# COMMAND ----------

trend_fit = stringIndexer_trend.fit(ml)
print(trend_fit.labels)

# COMMAND ----------

labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedTrend", 
                               labels=trend_fit.labels)

# COMMAND ----------

pipeline_rf1 = Pipeline(stages=[stringIndexer_trend, 
                               stringIndexer_weekday, 
                               stringIndexer_sent, 
                               onehot_weekday,
                               onehot_sent,
                               vectorAssembler_features, 
                               rf1, 
                               labelConverter])

pipeline_rf2 = Pipeline(stages=[stringIndexer_trend, 
                               stringIndexer_weekday, 
                               stringIndexer_sent, 
                               onehot_weekday,
                               onehot_sent,
                               vectorAssembler_features, 
                               rf2, 
                               labelConverter])

pipeline_rf3 = Pipeline(stages=[stringIndexer_trend, 
                               stringIndexer_weekday, 
                               stringIndexer_sent, 
                               onehot_weekday,
                               onehot_sent,
                               vectorAssembler_features, 
                               rf3, 
                               labelConverter])

# COMMAND ----------

model_rf1 = pipeline_rf1.fit(ml)
train_predictions = model_rf1.transform(ml)

# COMMAND ----------

model_rf2 = pipeline_rf2.fit(ml)
train_predictions2 = model_rf2.transform(ml)

# COMMAND ----------

model_rf3 = pipeline_rf3.fit(ml)
train_predictions3 = model_rf3.transform(ml)

# COMMAND ----------

#validation_data = ml.sample(False, 0.3, seed=529)

# COMMAND ----------

# MAGIC %md
# MAGIC #### evaluating the mdoel

# COMMAND ----------

predictions1 = model_rf1.transform(validation_data)

# COMMAND ----------

predictions2 = model_rf2.transform(validation_data)

# COMMAND ----------

predictions3 = model_rf3.transform(validation_data)

# COMMAND ----------

evaluatorRF = MulticlassClassificationEvaluator(labelCol="stock_trend_ix", predictionCol="prediction", metricName="accuracy")
accuracy1 = evaluatorRF.evaluate(predictions1)
accuracy2 = evaluatorRF.evaluate(predictions2)
accuracy3 = evaluatorRF.evaluate(predictions3)

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

y_pred1=predictions1.select("predictedTrend").collect()
y_orig1=predictions1.select("stock_trend").collect()

y_pred2=predictions2.select("predictedTrend").collect()
y_orig2=predictions2.select("stock_trend").collect()

y_pred3=predictions3.select("predictedTrend").collect()
y_orig3=predictions3.select("stock_trend").collect()

# COMMAND ----------

cm1 = confusion_matrix(y_orig1, y_pred1)
print("Confusion Matrix:")
print(cm1)

cm2 = confusion_matrix(y_orig2, y_pred2)
print("Confusion Matrix:")
print(cm2)

cm3 = confusion_matrix(y_orig3, y_pred3)
print("Confusion Matrix:")
print(cm3)

# COMMAND ----------

cm1 = pd.DataFrame(cm1)
cm2 = pd.DataFrame(cm2)
cm3 = pd.DataFrame(cm3)

# COMMAND ----------

import seaborn as sn
import matplotlib.pyplot as plt

# COMMAND ----------

sn.heatmap(cm1, annot=True, xticklabels=trend_fit.labels,yticklabels=trend_fit.labels,annot_kws={"size": 18},fmt='d')

# COMMAND ----------

sn.heatmap(cm2, annot=True, xticklabels=trend_fit.labels,yticklabels=trend_fit.labels,annot_kws={"size": 18},fmt='d')

# COMMAND ----------

sn.heatmap(cm3, annot=True, xticklabels=trend_fit.labels,yticklabels=trend_fit.labels,annot_kws={"size": 18},fmt='d').set(xlabel='Predicted',ylabel='Actural',title='Confusion Matrix of Random Forest Classification for Stock Trend Prediction')
plt.savefig("ml_plot1.png")

# COMMAND ----------

import os

# COMMAND ----------

#PLOT_DIR = os.path.join("data", "plot")
#os.makedirs(PLOT_DIR, exist_ok=True)
#fpath = os.path.join(PLOT_DIR, "ml_plot1.png")
#plt.savefig(fpath)

# COMMAND ----------

evaluatorRF = BinaryClassificationEvaluator(labelCol="stock_trend_ix", rawPredictionCol="prediction", metricName="areaUnderROC")
roc_result1 = evaluatorRF.evaluate(predictions1)
roc_result2 = evaluatorRF.evaluate(predictions2)
roc_result3 = evaluatorRF.evaluate(predictions3)

# COMMAND ----------

evaluatorRF2 = BinaryClassificationEvaluator(labelCol="stock_trend_ix", rawPredictionCol="prediction", metricName="areaUnderPR")
auc_result1 = evaluatorRF2.evaluate(predictions1)
auc_result2 = evaluatorRF2.evaluate(predictions2)
auc_result3 = evaluatorRF2.evaluate(predictions3)

# COMMAND ----------

results = pd.DataFrame(columns = {'accuracy','ROC','AUC'})
results['accuracy'] = [accuracy1, accuracy2, accuracy3]
results['ROC'] = [roc_result1, roc_result2, roc_result3]
results['AUC'] = [auc_result1, auc_result2, auc_result3]

# COMMAND ----------

results['model'] = ['model1','model2','model3']
results.set_index('model',inplace=True)
results

# COMMAND ----------

model_rf1.write().overwrite().save('rf1')
model_rf2.write().overwrite().save('rf2')
model_rf3.write().overwrite().save('rf3')


# COMMAND ----------

# MAGIC %md
# MAGIC #### generating the feature importance graph

# COMMAND ----------

vif = model_rf3.stages[-2].featureImportances
#cvModel.bestModel.stages[2] 
cols = ['comment_score','submission_weekday','volume','stock_trend','neg_sentiment_count','neu_sentiment_count','pos_sentiment_count','comment_sentiment','average_submission_length','average_submission_score','submission_weekday_ix','comment_sentiment_ix']
vif = pd.DataFrame(vif.toArray())
vif['feature_name'] = cols
vif.columns = ['VIF','feature_names']
vif = vif.drop(3)
vif = vif.sort_values('VIF',ascending=False)[0:5]
vif.feature_names = ['average_submission_length','positive_submission_count','neural_submission_count','comment_sentiment','submission_weekday']

# COMMAND ----------

import plotly.graph_objects as go

fig = go.Figure(go.Bar(
            x=vif['VIF'],
            y=vif['feature_names'],
            orientation='h'))
fig.update_layout(title = 'Top 5 Feature Importance for Predicting Stock Trend')
fig.update_xaxes(title='feature importance')
fig.update_yaxes(title='feature name')
fig.show()

# COMMAND ----------

PLOT_DIR = os.path.join("data", "plot")
os.makedirs(PLOT_DIR, exist_ok=True)
fpath = os.path.join(PLOT_DIR, "ml_plot2.html")
fig.write_html(fpath)

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
cols = ['comment_score','volume','neg_sentiment_count','neu_sentiment_count','pos_sentiment_count','average_submission_length','average_submission_score','stock_trend_ix']
assembler = VectorAssembler(inputCols=cols, outputCol=vector_col)
df_vector = assembler.transform(train_predictions3).select(vector_col)
# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

# COMMAND ----------

matrix_array = matrix.collect()[0]["pearson({})".format(vector_col)].values
matrix_array = matrix_array.reshape((8, 8))

# COMMAND ----------

matrix_df = pd.DataFrame(matrix_array)
matrix_df

# COMMAND ----------

plt.figure(figsize=(16, 16))
x_axis_labels = cols
y_axis_labels = cols
sn.heatmap(matrix_df, annot=True,xticklabels=cols,yticklabels=cols,annot_kws={"size": 10})
