

import numpy as np
import matplotlib.pyplot as plt
import random
import time

original_train_x = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc\McGill/Semester B/COMP551/Projects/Project 3/tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('C:/Users/gtsror/OneDrive/Important Docs/MSc\McGill/Semester B/COMP551/Projects/Project 3/tinyY.npy') 
#testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]
trainY = original_train_y[idx] 
# Feature extraction

# method 1+3:
# Flattening all data points; getting STDs 
pxlVals = np.zeros([64*64*3,1])
pxlSTD = np.zeros([1,1])

startTime = time.time()
for i in range(np.size(trainY)):
    tmp = np.copy(trainX[i].flatten())
    pxlVals = np.column_stack((pxlVals,tmp))
    pxlSTD = np.column_stack((pxlSTD,np.std(trainX[i])))
print("Extraction runtime is %f " % (time.time()-startTime))

# pxlVals needs to be transposed to be n pics x m features
#250 seconds for 3500 flattenings -> full thing ~1880 seconds (0.5hr)

# method 2:
# mean of pixels
tmp1 = np.mean(trainX,1)
tmp2 = np.mean(tmp1,1)
pxlMeans = np.mean(tmp2,1)



# to visualize only
plt.imshow(trainX[60].transpose(2,1,0))
