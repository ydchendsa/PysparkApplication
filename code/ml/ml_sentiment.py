# Databricks notebook source
# install packages
!pip install tensorflow 
!pip install transformers

# COMMAND ----------

# Calling packages
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer

keras.utils.set_random_seed(42)

# COMMAND ----------

#Read cleaned_df.csv where it contains sentiment
df = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/cleaned_df.csv')
df.head()

# COMMAND ----------

rec_neu  = df[df.submission_sentiment == 'neutral']
rec_pos  = df[df.submission_sentiment == 'positive']
rec_neg  = df[df.submission_sentiment == 'negative']

# COMMAND ----------

train_1, test_1 = train_test_split(rec_neu, test_size=0.2, random_state=42)
train_2, test_2 = train_test_split(rec_pos, test_size=0.2, random_state=42)
train_3, test_3 = train_test_split(rec_neg, test_size=0.2, random_state=42)

# COMMAND ----------

train = pd.concat([train_1,train_2,train_3])
test = pd.concat([test_1,test_2,test_3])

# COMMAND ----------

x_train = train['submission_words']
y_train = train['submission_sentiment']

# COMMAND ----------

x_train

# COMMAND ----------

x_test = test['submission_words']
y_test = test['submission_sentiment']

# COMMAND ----------

max_tokens = 6000
text_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="multi_hot",
    ngrams=2
)
text_vectorization.adapt(list(x_train))

# COMMAND ----------

text_vectorization_sentiment = keras.layers.TextVectorization()
text_vectorization_sentiment.adapt(y_train)

n_sentiments = text_vectorization_sentiment.vocabulary_size()

sentiment_data_train_sparse = text_vectorization_sentiment(y_train)
sentiment_data_test_sparse = text_vectorization_sentiment(y_test)


# COMMAND ----------

sentiment_data_test_sparse

# COMMAND ----------

units = 64

inputs = keras.Input(shape=(max_tokens, ))
x = keras.layers.Dense(units=128)(inputs)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(units=32)(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(units=8)(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

# COMMAND ----------

model.compile(optimizer="adam",
              #loss = 'mse',
              loss="sparse_categorical_crossentropy",
              #metrics=['rmse'])
              metrics=["sparse_categorical_accuracy"])

# COMMAND ----------

train_vec = text_vectorization(x_train)
test_vec = text_vectorization(x_test)

# COMMAND ----------

train_vec

# COMMAND ----------

history = model.fit(
          x = train_vec, 
          y = sentiment_data_train_sparse, 
          epochs=15,
          validation_data = (test_vec, sentiment_data_test_sparse))

# COMMAND ----------

model.evaluate(x=test_vec, y=sentiment_data_test_sparse)

# COMMAND ----------

pred = np.argmax(model.predict(x=test_vec),axis=1)

disp = ConfusionMatrixDisplay(confusion_matrix(tf.squeeze(sentiment_data_test_sparse), pred, labels=[2,3,4]),
                       display_labels=['pos','neg','neu'])
disp.plot()
disp.ax_.set_title("Confusion Matrix for Sentiment Prediction by Sparse Categorical Crossentropy")
plt.show()

# COMMAND ----------

plt.savefig("cmkeras.png")

# COMMAND ----------

f1_score(tf.squeeze(sentiment_data_test_sparse), pred, average='weighted')

# COMMAND ----------

pd.DataFrame(history.history)[['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot(figsize=(8,5))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.title("Accuracy for Train and Validation Plot")
plt.show()

# COMMAND ----------

plt.savefig("acckeras.png")

# COMMAND ----------

y_true = np.array(sentiment_data_test_sparse)
y_true = y_true.flatten()
y_score = pred


# COMMAND ----------

print(type(y_true))

# COMMAND ----------

import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

tpr, fpr, thresholds = roc_curve(y_true, y_score,pos_label=2)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()

# COMMAND ----------

fig.write_html("rockeras.html")

# COMMAND ----------

# save the model to databricks
#model.save('model.h5')

# COMMAND ----------

units = 64

inputs1 = keras.Input(shape=(max_tokens, ))
x1 = keras.layers.Dense(units=256)(inputs1)
x1 = keras.layers.Dropout(0.5)(x1)
x1 = keras.layers.Dense(units=128)(x1)
x1 = keras.layers.Dropout(0.5)(x)
x1 = keras.layers.Dense(units=64)(x1)
x1 = keras.layers.Dropout(0.5)(x1)
outputs = keras.layers.Dense(5, activation="softmax")(x)

model1 = keras.Model(inputs, outputs)
model1.summary()

# COMMAND ----------

model1.compile(optimizer="adam",
              #loss = 'mse',
              loss='mean_squared_error',
              metrics=[
                  'MeanSquaredError'
    ])

# COMMAND ----------

history1 = model1.fit(
          x = train_vec, 
          y = sentiment_data_train_sparse, 
          epochs=15,
          validation_data = (test_vec, sentiment_data_test_sparse))

# COMMAND ----------

model1.evaluate(x=test_vec, y=sentiment_data_test_sparse)

# COMMAND ----------

pred = np.argmax(model1.predict(x=test_vec),axis=1)

disp = ConfusionMatrixDisplay(confusion_matrix(tf.squeeze(sentiment_data_test_sparse), pred, labels=[2,3,4]),
                       display_labels=['pos','neg','neu'])
disp.plot()
disp.ax_.set_title("Confusion Matrix for Sentiment Prediction by MSE Loss")
plt.show()

# COMMAND ----------

plt.savefig("cmmse.png")

# COMMAND ----------

f1_score(tf.squeeze(sentiment_data_test_sparse), pred, average='weighted')

# COMMAND ----------

y_true = np.array(sentiment_data_test_sparse)
y_true = y_true.flatten()
y_score = pred
 

# COMMAND ----------


tpr, fpr, thresholds = roc_curve(y_true, y_score,pos_label=2)
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.update_xaxes(constrain='domain')
fig.show()

# COMMAND ----------

fig.write_html("losskeras.html")
