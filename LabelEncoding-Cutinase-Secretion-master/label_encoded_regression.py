# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:02:40 2020

@author: IGB
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:36:53 2020

@author: IGB
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
#from sklearn import preprocessing

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

seq_m = np.reshape(seq,(np.shape(seq)[0],np.shape(seq)[2]))
vocab = np.unique(seq)

X_mat_train = []

for i in range(np.shape(seq_m)[0]):
    X_conc = []
    curr_seq = seq_m[i]
    for j in range(np.shape(curr_seq)[0]):
        if j == 0 and curr_seq[j] == 'm':
            X_conc.append(np.shape(vocab)[0])
        else:
            for k in range(np.shape(vocab)[0]):
                if curr_seq[j] == vocab[k]:
                    X_conc.append(k)
                    
    X_mat_train.append(np.transpose(X_conc))
    
    
seq = []
for i in range(np.size(test_seq)):
    #print(i)
    seq.append(np.array(pad_sequences([list(test_seq[i].lower())] ,maxlen = max_length, dtype='str',padding ='post')[0]))

X_mat_test = []

for i in range(np.shape(seq)[0]):
    X_conc = []
    curr_seq = seq_m[i]
    for j in range(np.shape(curr_seq)[0]):
        if j == 0 and curr_seq[j] == 'm':
            X_conc.append(np.shape(vocab)[0])
        else:
            for k in range(np.shape(vocab)[0]):
                if curr_seq[j] == vocab[k]:
                    X_conc.append(k)
                    
    X_mat_test.append(np.transpose(X_conc))

#print(X_mat_test)
#y_log = data[1:,2].astype(float)
#y = np.exp(y_log)

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_mat_train, y_train)
y_pred = regressor.predict(X_mat_test)
#from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#for normalized data
normalized_y_train = y_train/np.amax(y_train)
normalized_y_test = y_test/np.amax(y_train)

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_mat_train, normalized_y_train)
normalized_y_pred = regressor.predict(X_mat_test)
#from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(normalized_y_test, normalized_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(normalized_y_test, normalized_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(normalized_y_test, normalized_y_pred)))


