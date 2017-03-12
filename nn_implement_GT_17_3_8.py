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

original_train_x = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy')  # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy')
#testX = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))


classes = 40
numLayers = 4

trainX = original_train_x[idx]/255
trainX = np.reshape(trainX, (26344,12288))


#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
Y_train = np_utils.to_categorical(trainY, classes)
#Y_train = trainY
# choosing a smaller subset
Y_train1 = Y_train[0:100]
X_train1 = trainX[0:100]
#
#X_test = trainX[2500:2800]
#Y_test = Y_train[2500:2800]
train_size = len(X_train1)
input_dim = np.size(X_train1,1)
hlayer_size = 10 #size of each hidden layer
# randomize initial weights
# we need to have numLayers-1 weights sets, by the number of inputs

#W = np.zeros([numLayers-1, input_dim])
#W = np.random.randn(input_dim, train_size)
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
    return sigmoid(np.dot(x,w))
    
def derivative(output):
	return output * (1.0 - output)
# setting up forward propagation:
    # we shall run forward, with a random set of weights for each layer
all_xs = {}
all_ws = {}
x = np.copy(np.asarray(X_train1))
all_xs[0] = x

for layer in range(0,numLayers):
    # running on the layers to calculate the outputs
    
    # Start by calculating the output of all the neurons in layer K
    if layer == 0:
        # in case we're in the input layer, we'll use the input as x:
        # start by defining the set of weights, randomly
       
        W = np.random.randn(input_dim, hlayer_size)
        # calculate the output
        o = output_calc(W,x)
    elif layer == numLayers-1:
        W = np.random.randn(prev_dim, classes)
        # if it's in the last layer, the output will be the predicted output
        # we randomize W to have the appropriate size
        o_last = output_calc(W,xji)
        probs = np.exp(o_last) / np.sum(np.exp(o_last),axis=1,keepdims=True)
        all_ws[layer]=W
        all_xs[layer+1]=o_last
        break #exiting the for loop, since it's the last layer
    else: 
        # if it's in one of the midlayers
        W = np.random.randn(prev_dim, hlayer_size)
        o = output_calc(W,xji)
    xji = o # copying the output into the input of the next layer
    all_xs[layer+1]=xji
    all_ws[layer]=W
    del W
    prev_dim = np.size(xji,1)
    
# back propagation now - 
#err_output = (Y_train1-o_last)*transfer_derivative(o_last)
#err_hidden = 

# delta of the fourth layer out of 4:
#delta_final = all_xs[numLayers]-Y_train1
# delta of the third layer (out of 4 layers)
#delta_3 = np.multiply(np.dot(all_ws[numLayers-1],np.transpose(delta_final)),np.transpose(derivative(all_xs[numLayers-1])))
alpha = 0.1
deltas={}
def backprop(all_xs,all_ws,Y_train1,numLayers,alpha,deltas):
    for layer in numLayers:
        if layer == 0:
            #in case it's the last layer
            delta = all_xs[numLayers-layer]-Y_train1
        elif layer == numLayers:
            #in this case we got back to the first layer which doesnt have a dlta
            all_ws[numLayers-layer] = all_ws[numLayers-layer]+alpha*delta
            break
        else:
            delta = np.transpose(np.multiply(np.dot(all_ws[numLayers-layer],np.transpose(deltas[layer-1])),np.transpose(derivative(all_xs[numLayers-layer]))))
            all_ws[numLayers-layer] = all_ws[numLayers-layer]+np.transpose(alpha*np.dot(np.transpose(deltas[layer-1]),all_xs[numLayers-layer]))
        deltas[layer] = delta

    return all_ws, deltas

all_ws, deltas = backprop(all_xs,all_ws,Y_train1,numLayers,alpha,deltas)


