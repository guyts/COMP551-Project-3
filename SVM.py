from sklearn import svm
%matplotlib inline
import cv2


import numpy as np
import random
import matplotlib.pyplot as plt

original_train_x = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('tinyY.npy') 
testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]
trainY = original_train_y[idx] 
img = plt.imshow(trainX[60].transpose(2,1,0))

trainX = original_train_x[idx]/255
#X_train = np.copy(trainX)
trainY = original_train_y[idx] 
#Y_train = np_utils.to_categorical(trainY, 40)

# choosing a smaller subset
Y_train1 = trainY[0:1000]
X_train1 = trainX[0:1000]
X_test = trainX[2500:3400]
Y_test = trainY[2500:3400]


clf = svm.SVC()
clf.fit(X_train1, Y_train1)  
svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.predict(testX)
