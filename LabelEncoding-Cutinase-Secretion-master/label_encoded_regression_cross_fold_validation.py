# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:56:09 2020

@author: IGB
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:44:48 2020

@author: IGB
"""

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#from sklearn import preprocessing

df = pd.read_csv('sp_bacillus.txt',header = None)
unshuffled_data = df.to_numpy()
data = unshuffled_data[1:,:]

'''
train_seq, test_seq, pre_y_train, pre_y_test = train_test_split(data[:,1], data[:,2] , test_size=0.25, random_state=0)
y_train = [float(i) for i in pre_y_train]
y_test = [float(i) for i in pre_y_test]


seq = []
max_length = np.amax(data[0:,4].astype(int))
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
'''

seq = []
max_length = np.amax(data[0:,4].astype(int))
for i in range(np.size(data[:,1])):
    #print(i)
    seq.append(np.array(pad_sequences([list(data[i,1].lower())] ,maxlen = max_length, dtype='str',padding ='post')[0]))

X_mat = []

for i in range(np.shape(seq)[0]):
    X_conc = []
    curr_seq = seq[i]
    for j in range(np.shape(curr_seq)[0]):
        if j == 0 and curr_seq[j] == 'm':
            X_conc.append(np.shape(vocab)[0])
        else:
            for k in range(np.shape(vocab)[0]):
                if curr_seq[j] == vocab[k]:
                    X_conc.append(k)
                    
    X_mat.append(np.transpose(X_conc))

#print(X_mat_test)
y = data[0:,2].astype(float)

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
scores = cross_val_score(regressor, X_mat,y, cv=4)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
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
'''