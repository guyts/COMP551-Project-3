%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv
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


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray
img_gray = to_gray(trainX[60].transpose(2,1,0))
plt.imshow(img_gray, cmap='gray');


# -------------------------------------------------------------------------------- #
# ----------- Changeing training data to gray scale ------------------------------ #
# -------------------------------------------------------------------------------- #


num = trainX.size
graytrainx = []
for i in range(0, 26344):
    print("changing training data to Gray" + " " + str(i))
    graytrainx.append(to_gray(trainX[i].transpose(2,1,0)))
graytrainxArr = np.asarray(graytrainx)
twoDtrainxArr = graytrainxArr.transpose(0,1,2).reshape(26344 ,-1)

# -------------------------------------------------------------------------------- #
# ----------- Changeing test data to gray scale ---------------------------------- #
# -------------------------------------------------------------------------------- #
graytestx = []
for i in range(0, 6600):
    print("changing test data to Gray" + " " + str(i))
    graytestx.append(to_gray(testX[i].transpose(2,1,0)))
graytestxArr = np.asarray(graytestx)
twoDtestxArr = graytestxArr.transpose(2,0,1).reshape(6600 ,-1)


# ------------------------------------------------------------#
# ----------- LBP-------------------------------------------- #
# ----------------------------------------------------------- #

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from __future__ import print_function

METHOD = 'uniform'

# settings for LBP
radius = 3
n_points = 8 * radius


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')
        
# ------------------------------------------------------------#
# ----------- Extracting LBP feature & histogram for trainx------ #
# ----------------------------------------------------------- #        

lbptrainX = []
for i in range(0, 26344):
  #print("LBP" + " " + str(i))
  lbptrainX.append(local_binary_pattern(graytrainxArr[i], n_points, radius, METHOD))

print ("Extracting LBP feature & histogram for training set...")
histtrainX = []
for i in range(0, 26344):
  print("hist" + " " + str(i))
  histtrainX.append(np.histogram(lbptrainX[i]))
histtrainXArr = np.asarray(histtrainX)

# ------------------------------------------------------------#
# ----------- Extracting LBP feature & histogram for testx------ #
# ----------------------------------------------------------- #        

lbptestX = []
for i in range(0, 6600):
  #print("LBP" + " " + str(i))
  lbptestX.append(local_binary_pattern(graytestxArr[i], n_points, radius, METHOD))
  
print ("Extracting LBP feature & histogram for Testing set...")
histtestX = []
for i in range(0, 6600):
  print("hist" + " " + str(i))
  histtestX.append(np.histogram(lbptestX[i]))
histtestXArr = np.asarray(histtestX)

# ------------------------------------------------------------#
# ----------- Classification step- SVM-------------------- #
# ----------------------------------------------------------- #   


# Instantiate and train a support vector machine
svm = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# Train the SVM
x_train, x_val, y_train, y_val = train_test_split(histtrainXArr , trainY, test_size = 0.3)


print("Training SVM...")
svm.fit(x_train, y_train)
print("Trained SVM.")

# Predict on training data
print("Predicting training set...")
pred_train = svm.predict(x_train)
pred_train_df = pd.DataFrame({
	'prediction': pred_train,
	'class': y_train
	})
pred_train_df.to_csv("trainprediction.csv",
	index_label = "id", header = ["prediction", "class"])
print(svm.score(x_train, y_train))

# Predict on validation data
print("Predicting validation set...")
pred_val = svm.predict(x_val)
pred_val_df = pd.DataFrame({
	'prediction': pred_val,
	'class': y_val
	})
pred_val_df.to_csv("valpredictions.csv",
	index_label = "id", header = ["prediction", "class"])
print(svm.score(x_val, y_val))
print(confusion_matrix(pred_val, y_val))

# Predict on test data
print("Predicting test set...")

pred_test = svm.predict(histtestXArr)
pred_test_df = pd.DataFrame({
	'class': pred_test,
	})
pred_test_df.to_csv("predictions.csv",
	index_label = "id", header = ["class"])


