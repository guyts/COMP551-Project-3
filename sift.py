%matplotlib inline
import cv2
help(cv2)
import numpy as np
import random
import matplotlib.pyplot as plt

original_train_x = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('tinyY.npy') 
#testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]
trainY = original_train_y[idx] 
img = plt.imshow(trainX[60].transpose(2,1,0))
print (cv2.__version__)
def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    img = plt.imshow(trainX[60].transpose(2,1,0))
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))
#show_rgb_img(img);
def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray
    
img_gray = to_gray(trainX[60].transpose(2,1,0))
plt.imshow(img_gray, cmap='gray');

#%%%%%%----------SIFT----------%%%%%%

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
   # sift = cv2.SIFT()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

# generate SIFT keypoints and descriptors
img_kp, img_desc = gen_sift_features(img_gray)


print ('Here are what our SIFT features look like for the image:')
show_sift_features(img_gray, trainX[60].transpose(2,1,0), img_kp);