from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, Dropout, merge
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2
from keras import backend as K
from collections import defaultdict
import numpy as np
import theano as T

import sys
import pandas as pd

T.config.exception_verbosity = 'high'


m = 5

def cost_function(y_true, y_pred):
    v = m - K.sum(y_pred, axis=0)
    z = T.tensor.scalar()
    if T.tensor.gt(v,z): return v
    return z
 
dropout = 0.2

# loading data
training_df = pd.read_csv('training_set.csv',header=0)
labels_df = pd.read_csv('training_labels.csv',header=0)

training_data = training_df.values
labels_data = labels_df.values

xTrain = training_data[0:1000]
yTrain = labels_data[0:1000]

testing_df = pd.read_csv('testing_set.csv', header=0)
tLabels_df = pd.read_csv('testing_labels.csv', header=0)

testing_data = testing_df.values

tLabels_data = tLabels_df.values

xTest = testing_data[0:200]
yTest = tLabels_data[0:200]

# standardization
scaler = preprocessing.StandardScaler().fit(xTrain)
scaler.transform(xTrain)
scaler.transform(xTest)

num_examples = xTrain.shape[0]
dims =  int( xTrain.shape[1] / 3 )
layer_nodes = dims

pos1 = Input( shape=(dims,) )
pos2 = Input( shape=(dims,) )
neg2 = Input( shape=(dims,) )

W_0 = Dense(layer_nodes, input_shape=(dims,), activation='linear', W_regularizer=l2(0.01), init='glorot_uniform')
W_St = Dense(layer_nodes, input_shape=(dims,), activation='linear', W_regularizer=l2(0.01), init='glorot_uniform')

##########################################
# FIRST SIAMESE NETWORK
# linear layer
W_01 = Dropout(dropout)( W_0( pos1 ) )
W_St1 = Dropout(dropout)( W_St( pos1 ) )
W_m1 = merge([W_01, W_St1], mode='sum')
# second layer
W_02 = Dropout(dropout)( W_0( pos2 ) )
W_St2 = Dropout(dropout)( W_St( pos2 ) ) 
W_m2 = merge([W_02, W_St2], mode='sum')
# third layer
W_mul1 = merge([W_m1, W_m2], mode='dot', dot_axes=1)
##########################################

##########################################
# SECOND SIAMESE NETWORK
# linear layer
W_03 = Dropout(dropout)( W_0( pos1 ) )
W_St3 = Dropout(dropout)( W_St( pos1 ) )
W_m3 = merge([W_03, W_St3], mode='sum')
# second layer
W_04 = Dropout(dropout)( W_0( neg2 ) )
W_St4 = Dropout(dropout)( W_St( neg2 ) ) 
W_m4 = merge([W_04, W_St4], mode='sum')
# third layer
W_mul2 = merge([W_m3, W_m4], mode='dot', dot_axes=1)
##########################################

##########################################
# LAST LAYER
W_sub = merge([W_mul1, W_mul2], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])

model = Model(input=[pos1, pos2, neg2], output=W_sub)

# adaptive SGD
adam=Adam(lr=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08)

model.compile(loss=cost_function, optimizer=adam)
model.fit([ xTrain[:,0:100], xTrain[:,100:200], xTrain[:,200:] ], yTrain, batch_size=10, nb_epoch=10, verbose=1,shuffle=True)

