# -*- coding: utf-8 -*-
"""
Retrieving with data from memory

"""

import numpy as np
from sklearn import metrics
import datetime
import caffe
from dataset import Dataset
import sys
                   

# parameters

MAINPATH = '/home/atuezkoe/CaffeData/data/'

TRAINPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/train/'
VALPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/val/'
TESTPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/test/'

LABELS_FILE = '../../preprocessedData/labels/labels.txt'

modelDef = MAINPATH + 'models/VGG/VGG_ILSVRC_16_layers_deploy.prototxt'
modelParams = MAINPATH + 'models/VGG/caffe_alexnet_train_iter_10000.caffemodel'
gpu = True

# Make settings
if gpu:
    caffe.set_mode_gpu()
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')


### step 1: load data
print('{:s} - Load test data'.format(str(datetime.datetime.now()).split('.')[0]))
testDataset = Dataset(TRAINPATH)
testDataset.addImages(VALPATH)

queryDataset = Dataset(TESTPATH)
sys.exit(0)


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 3: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))


labels = []
distances = []
while queryDataset.hasMoreImages():
    queryWindow = queryDataset.getNextWindow()
    net.set_input_arrays(queryWindow.data)
    net.forward()

    queryFeature = net.blobs['pool5'].data

    while testDataset.hasMoreImages():
        testWindow = testDataset.getNextWindow()        
        net.set_input_arrays(testWindow.data)
        net.forward()
        testFeature = net.blobs['pool5'].data
        distances.append(np.sum((queryFeature - testFeature)**2, axis=1))     # euclidean distance between sample pairs

        testDataset.loadNextImage()

    maxDistance = distances.max()
    distances /= maxDistance
    similarities = 1 / distances
    

    queryDataset.loadNextImage()
    labels = np.concatenate((labels, testNetLabels))
    distances = np.concatenate((distances, curDist))    


