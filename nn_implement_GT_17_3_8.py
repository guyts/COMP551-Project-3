# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:41:54 2017

@author: gtsror
"""

from keras.utils import np_utils

import time
import numpy as np
import matplotlib.pyplot as plt
import random

original_train_x = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy')  # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy')
testX = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))


classes = 40
numLayers = 3

trainX = original_train_x[idx]/255
trainX = np.reshape(trainX, (26344,12288))


#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
#Y_train = np_utils.to_categorical(trainY, classes)
Y_train = trainY
# choosing a smaller subset
Y_train1 = Y_train[0:100]
X_train1 = trainX[0:100]
#
#X_test = trainX[2500:2800]
#Y_test = Y_train[2500:2800]
train_size = len(X_train1)
input_dim = np.size(X_train1,1)
# randomize initial weights
# we need to have numLayers-1 weights sets, by the number of inputs

#W = np.zeros([numLayers-1, input_dim])
W = np.random.randn(input_dim, numLayers-1)
#    W = np.append(w_tmp)

# Loss function definition
def loss_fn(model):
    W1 = model['W1']

def sigmoid(z):
    # sigmoid function defined as:
    return(1/(1+np.exp(-z)))
    
def output_calc(w,x):
    # a neuron output will be calculated by this, where w is the set of weights
    # in the entrance of a neuron, and x is the neuron input (previous neuron)
    # output. Both x and w are vectors
    return sigmoid(np.dot(w,x))
    
    
for img in trainX:
    # running on the input pictures
    x = np.transpose(np.asarray(trainX[img]))
    for layer in range(0,numLayers):
        w = W[:,layer]
        # Start by calculating the output of all the neurons in layer K
        if layer == 0:
            # in case we're in the input layer:
            o = output_calc(w,x)
        elif layer == numLayers-1:
            # if it's in the last layer, the output will be the predicted output
            o_last = output_calc(w,x)
            break #exiting the for loop, since it's the last layer
        else: 
            # if it's in one of the midlayers
            o = output_calc(w,xji)
        xji = o # copying the output into the input of the next layer