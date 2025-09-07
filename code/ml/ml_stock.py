# Databricks notebook source
pip install nltk

# COMMAND ----------

pip install mlxtend

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix

from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, MaxPool1D, Dropout, Dense, GlobalMaxPooling1D, Embedding, Activation
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences

# COMMAND ----------

marvel = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/marvel_stock_full.csv')#,index_col=0)
marvel = marvel.iloc[:,2:]


# COMMAND ----------

list(marvel.columns)

# COMMAND ----------

newdf = marvel.drop_duplicates('content')
newdf

# COMMAND ----------

not_movie = newdf[newdf.is_movie == 0]
print(len(not_movie))

# COMMAND ----------

# feeding the model with NETFLIX as training dataset
nflx = pd.read_csv('/Workspace/Repos/hs1063@georgetown.edu/fall-2022-reddit-big-data-project-project-group-5/data/csv/netflix.csv')
print(len(nflx))
nflx.head()

# COMMAND ----------

pred_traning_set = nflx[['Open']]
pred_traning_set = pred_traning_set.dropna()
pred_traning_set

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train = scaler.fit_transform(pred_traning_set)
scaled_train

# COMMAND ----------

x_train = []
y_train = []
for i in range(0,4881):
    x_train.append(scaled_train[i-0:i,0])
    y_train.append(scaled_train[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

# COMMAND ----------

print(x_train.shape)
print(y_train.shape)

# COMMAND ----------

x_train = np.reshape(scaled_train,(scaled_train.shape[0],scaled_train.shape[1],1))
x_train.shape

# COMMAND ----------

model = Sequential()
model.add(LSTM(units = 50,
         return_sequences = True,
         input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50,
         return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50,
         return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

# COMMAND ----------

model.compile(optimizer = 'adam', loss = 'huber_loss')
model.fit(x_train,
          y_train, 
          epochs = 32, 
          batch_size = 64)

# COMMAND ----------

actural_set = newdf[['timestamp_init','open']]
actural_set = actural_set.dropna()
actural_set['Date'] = pd.to_datetime(actural_set['timestamp_init'])
actural_set = actural_set.sort_values(by='Date') 
actural_set

# COMMAND ----------

scaler = MinMaxScaler(feature_range = (0,1))
pred_testing_set = actural_set[['open']]
scaled_set = scaler.fit_transform(pred_testing_set)
scaled_set

# COMMAND ----------

#x_test = np.array(scaled_set)
scaled_set.shape

# COMMAND ----------

x_test = np.reshape(scaled_set,(scaled_set.shape[0],scaled_set.shape[1],1))
x_test.shape

# COMMAND ----------

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# COMMAND ----------

len(predicted_price)

# COMMAND ----------

actural_set['predicted_price'] = predicted_price
actural_set

# COMMAND ----------

import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter(x=actural_set['timestamp_init'], 
                         y=actural_set['open'],
                        mode='lines',
                        name='actural'))

fig.add_trace(go.Scatter(x=actural_set['timestamp_init'], 
                         y=actural_set['predicted_price'],
                        mode='lines',
                        name='predicted'))
fig.update_layout(
    title="Disney Stock Price Prediction by Keras Model",
    xaxis_title="Time",
    yaxis_title="Stock Prics/USD",
    )
fig.show()

# COMMAND ----------

fig.write_html("stock.html")

# COMMAND ----------

y_true = np.array(actural_set['open'])
y_score = np.array(actural_set['predicted_price'])

# COMMAND ----------

from sklearn.metrics import r2_score,f1_score
coefficient_of_dermination = r2_score(y_true, y_score)
coefficient_of_dermination

# COMMAND ----------

# RMSE symbol is dollar 
import math
MSE = np.square(np.subtract(y_true,y_score)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
