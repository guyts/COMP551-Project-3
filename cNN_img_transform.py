# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:31:53 2017

@author: guyts
"""

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
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
from keras.utils import np_utils
from sklearn import linear_model, datasets
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.cross_validation  import train_test_split

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import random

original_train_x = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy')  # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy')
kaggle = np.load('C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/tinyX_test.npy') # (6600, 3, 64, 64)


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255,
        dim_ordering='th',
        fill_mode='nearest')


trainX, testX, trainY, testY = train_test_split(original_train_x, original_train_y, test_size=0.2, random_state=42)

del original_train_x, original_train_y

#i = 0
#tempX = temp.loc[temp.Y == i].head(140)
#i +=1
#while i < 40:
#    temp2 = temp.loc[temp.Y == i].head(140)
#    tempX = tempX.append(temp2)
#    i+=1

trainX2 = list(trainX)
trainY2 = list(trainY)

for j in range(0,40):
    a = np.array(np.where(trainY == j))
    b = np.array([trainX[z] for z in a])
    b = b[0, ...]
    print(len(b))
    if len(b) < 1000:
        print ("+")
        a = a.T
        i = 0
        for X, Y in datagen.flow(b,a,batch_size=1):
            trainX2.append(X[0])
            trainY2.append(j)
            i+=1
            if i > 1000-len(b):
                break
   

#np.save('trainX2_datagen', trainX2)
#np.save('trainY2_datagen', trainY2)

trainX3 = np.array(trainX2)
trainY3 = np.array(trainY2)

idx = random.sample(range(np.size(trainY)), np.size(trainY))

trainX = trainX3[idx]
#X_train = np.copy(trainX)
trainY = trainY3[idx] 
Y_train = np_utils.to_categorical(trainY, 40)
trainY = np.copy(Y_train)

del Y_train, trainX3, trainY3, a, b, X, Y, trainX2, trainY2
# choosing a smaller subset
#Y_train1 = Y_train[0:23000]
#X_train1 = trainX[0:23000]
#
#X_test = trainX[23001:26000]
#Y_test = Y_train[23001:26000]

# Creating the layers
model = Sequential()
# taking the shape
#model.add(Reshape((3,64,64),  input_shape=(12288,)))
#
#model.add(Conv2D(16, 3, 3, border_mode='same', subsample=(4,4), activation='relu'))
#model.add(Conv2D(16, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))

model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(BatchNormalization())
#

model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
#model.add(BatchNormalization())
#
#model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))
#model.add(MaxPooling2D((2,2), strides=(2,2)))
#model.add(BatchNormalization())

#model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(3,64,64)))

model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dropout(0.5)) # Dropout is a good remedy to overfitting
model.add(Dense(40))
model.add(Activation('softmax')) # 10 classes again

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=200,nb_epoch=50)

validation_metrics = model.evaluate(X_test, Y_test)
#print()
#print(validation_metrics)
#
Z = model.predict(testX)

Z = pd.DataFrame(Z)
Z = pd.DataFrame(Z.idxmax(axis=1))
Z.columns = ['class']
Z.to_csv("C:/Users/guyts/OneDrive/Important Docs/MSc/McGill/Semester B/COMP551/Projects/Project 3/cNN_4Conv_4Pool_23k_300batch_54epoch_various_filters_overfit.csv", index=True, index_label="id")
