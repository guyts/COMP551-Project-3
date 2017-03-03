# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:32:03 2017

@author: gtsror
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import random

original_train_x = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc\McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc\McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy') 
testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]/255
#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
Y_train = np_utils.to_categorical(trainY, 40)

# choosing a smaller subset
Y_train1 = Y_train[0:1000]
X_train1 = trainX[0:1000]
X_test = trainX[2500:3400]
Y_test = Y_train[2500:3400]

# Creating the layers
model = Sequential()
#
                                
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3,64,64)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Convolution2D(64, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
#model.add(Convolution2D(128, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Convolution2D(128, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(40))
model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3,64,64)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(40, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train1, Y_train1, 
          batch_size=32, nb_epoch=5, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)