# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:32:03 2017

@author: gtsror
"""
from keras.utils import np_utils

import time
import numpy as np
import matplotlib.pyplot as plt
import random

original_train_x = np.load('C:/Users/guytsror/Documents/Python Scripts/tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/guytsror/Documents/Python Scripts/tinyY.npy') 
#testX = np.load('C:/Users/guytsror/Documents/Python Scripts/tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]/255
#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
Y_train = np_utils.to_categorical(trainY, 40)
#Y_train = trainY
# choosing a smaller subset
Y_train1 = Y_train[0:100]
X_train1 = trainX[0:100]
#
#X_test = trainX[2500:2800]
#Y_test = Y_train[2500:2800]


