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

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
from keras.utils import np_utils

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import random

original_train_x = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy')  # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy')
testX = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]
#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
Y_train = np_utils.to_categorical(trainY, 40)
#Y_train = trainY
# choosing a smaller subset
Y_train1 = Y_train[0:21100]
X_train1 = trainX[0:21100]
#
X_test = trainX[22000:26000]
Y_test = Y_train[22000:26000]

# Creating the layers
model = Sequential()
# taking the shape
#model.add(Reshape((3,64,64),  input_shape=(12288,)))
#
#model.add(Conv2D(16, 3, 3, border_mode='same', subsample=(4,4), activation='relu'))
#model.add(Conv2D(16, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))

model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(BatchNormalization())

#model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))




model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dropout(0.5)) # Dropout is a good remedy to overfitting
model.add(Dense(40))
model.add(Activation('softmax')) # 10 classes again

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train1, Y_train1, batch_size=300,nb_epoch=150)


validation_metrics = model.evaluate(X_test, Y_test)
#print()
#print(validation_metrics)
#
Z = model.predict(testX)

Z = pd.DataFrame(Z)
Z = pd.DataFrame(Z.idxmax(axis=1))
Z.columns = ['class']
Z.to_csv("C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/cNN_4Conv_4Pool_21k_400batch_150epoch.csv", index=True, index_label="id")











                                
#model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3,64,64)))
#model.add(Activation('relu'))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Convolution2D(128, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Convolution2D(128, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(40))
#model.add(Activation('softmax'))

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

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
#
#model.fit(X_train1, Y_train1, 
#          batch_size=32, nb_epoch=5, verbose=1)
#
#score = model.evaluate(X_test, Y_test, verbose=0)
#
#testSet = testX/255
#classes = model.predict_classes(testSet, batch_size=32)
