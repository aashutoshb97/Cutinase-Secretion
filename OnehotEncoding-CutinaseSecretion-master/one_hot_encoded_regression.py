# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:36:53 2020

@author: IGB
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics


df = pd.read_csv('sp_bacillus.txt',header = None)
data = df.to_numpy()

train_data = []
test_data = []
train_seq = []
pre_y_train = []
test_seq = []
pre_y_test = []

for i in range(1,np.shape(data)[0]):
    if data[i,3] == 'TRUE':
        train_data.append(data[i,:])
        train_seq.append(data[i,1])
        pre_y_train.append(data[i,2])
    else:
        test_data.append(data[i,:])
        test_seq.append(data[i,1])
        pre_y_test.append(data[i,2])

y_train = [float(i) for i in pre_y_train]
y_test = [float(i) for i in pre_y_test]

    
seq = []
max_length = np.amax(data[1:,4].astype(int))
for i in range(np.size(train_seq)):
    #print(i)
    seq.append(np.array(pad_sequences([list(train_seq[i].lower())] ,maxlen = max_length, dtype='str',padding ='post')))

vocab = np.unique(seq)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(vocab)
X_mat_train = []

for i in range(np.shape(seq)[0]):
    enc = OneHotEncoder(categories = [(vocab)],sparse=False)
    X = np.matrix(enc.fit_transform(seq[0].reshape(-1,1)))
    X_conc = []
    X_conc = X[0]
    
    for j in range(1,np.shape(X)[0]):
        X_conc = np.vstack((X_conc,X[j]))
        
    X_mat_train.append(np.transpose(X_conc))

seq = []
for i in range(np.size(test_seq)):
    #print(i)
    seq.append(np.array(pad_sequences([list(test_seq[i].lower())] ,maxlen = max_length, dtype='str',padding ='post')[0]))

vocab = np.unique(seq)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(vocab)
X_mat_test = []

for i in range(np.shape(seq)[0]):
    enc = OneHotEncoder(categories = [(vocab)],sparse=False)
    X = np.matrix(enc.fit_transform(seq[0].reshape(-1,1)))
    X_conc = []
    X_conc = X[0]
    for j in range(1,np.shape(X)[0]):
        X_conc = np.vstack((X_conc,X[j]))
        
    X_mat_test.append(np.transpose(X_conc))

#print(X_mat_test)
#y_log = data[1:,2].astype(float)
#y = np.exp(y_log)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(np.reshape(X_mat_train,(np.shape(X_mat_train)[0],np.shape(X_mat_train)[1]*np.shape(X_mat_train)[2])), y_train)
y_pred = regressor.predict(np.reshape(X_mat_test,(np.shape(X_mat_test)[0],np.shape(X_mat_test)[1]*np.shape(X_mat_test)[2])))

#from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
