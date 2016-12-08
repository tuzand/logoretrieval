# -*- coding: utf-8 -*-
"""
Testing with data from memory, inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/tutorial/examples/01-learning-lenet.ipynb
Useful because of siamese setup: avoids unnecessary disk access.

@author: ch
"""

import numpy as np
from sklearn import metrics
import datetime
import caffe
from NetDataset import NetDataset
                   

### task: test pretrained siamese network in python
# parameters
dataFolderTest = './db/test'
modelDef = 'lrfacenet-definition-32-MaxMargin.prototxt'
modelParams = 'lrfacenet-32_iter_100-MaxMargin.caffemodel'

gpu = False 
imgDim = 32   # width and height of image

# Make settings
if gpu:
    caffe.set_mode_gpu()
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')


### step 1: load data
print('{:s} - Load test data'.format(str(datetime.datetime.now()).split('.')[0]))
testDataset = NetDataset()
testDataset.targetSize = imgDim
testDataset.flipAugmentation = False
testDataset.shiftAugmentation = False
testDataset.loadImageData(dataFolderTest)
testDataset.printStatus = False


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 3: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))

nTestBatches = testDataset.posPairs.shape[0] / (testDataset.batchSize/2)

labels = []
distances = []
testDataset.posPointer = 0
testDataset.negPointer = 0
for i in range(nTestBatches):
    testNetData, testNetLabels = testDataset.getNextVerficationBatch()
    net.set_input_arrays(testNetData, testNetLabels)
    net.forward()

    # accuracy test for verification
    ft1 = net.blobs['feat'].data
    ft2 = net.blobs['feat_p'].data
    curDist = np.sum((ft1 - ft2)**2, axis=1)     # euclidean distance between sample pairs
    
    labels = np.concatenate((labels, testNetLabels))
    distances = np.concatenate((distances, curDist))    

fpr, tpr, thresholds = metrics.roc_curve(labels, distances, pos_label=2)

print fpr
print tpr
print distances



# search for best threshold and use that accuracy
#bestAccuracy = -float('inf')
#for i in range(len(labels)):
#    curThresh = distances[i]
#    prediction = distances <= curThresh
#    accuracy = np.mean(prediction == labels)
#    if (accuracy > bestAccuracy):
#        bestAccuracy = accuracy
#        thresh = curThresh  

#print('{:s} - Finished with accuracy: {:.3f}'.format(str(datetime.datetime.now()).split('.')[0], bestAccuracy))   