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
import operator
                   

# parameters

MAINPATH = '/home/atuezkoe/CaffeData/data/'

TRAINPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/train/'
VALPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/val/'
TESTPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/test/'

RESULTPATH = './result/'
RESULTPOSTFIX = '.result2.txt'

modelDef = '../models/VGG_ILSVRC_16_layers_deploy.prototxt'
modelParams = MAINPATH + 'models/caffe_alexnet_train_iter_10000.caffemodel'
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
testDataset = Dataset(TRAINPATH, False)
#testDataset.addImages(VALPATH)

queryDataset = Dataset(TESTPATH, True)


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 3: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))


labels = []
distances = []
result = dict()
while queryDataset.hasMoreImages():
    queryWindow = queryDataset.getNextWindow()
    net.set_input_arrays(queryWindow.data)
    net.forward()

    queryFeature = net.blobs['pool5'].data

    testDataset.reset()
    minDistance = float("inf")
    while testDataset.hasMoreImages():
        testWindow = testDataset.getNextWindow()
        net.set_input_arrays(testWindow.data)
        net.forward()
        testFeature = net.blobs['pool5'].data
        distance = np.sum((queryFeature - testFeature)**2, axis=1)     # euclidean distance between sample pairs
        result[testDataset.getActFilename()] = distance
        if distance < minDistance:
            minDistance = distance

        testDataset.loadNextImage()


    for key in result:
        result[key] /= minDistance
        result[key] = 1 / result[key]

    sortedResult = sorted(result.items(), key=operator.itemgetter(1))

    if not os.path.exists(RESULTPATH):
        os.makedirs(RESULTPATH)
    with open(os.path.join(RESULTPATH, queryDatabase.getActFilename() + RESULTPOSTFIX), 'w') as res:
        for i in range(len(sortedResult)):
            res.write(sortedResult[i][0] + " " + sortedResult[i][1])


    

    queryDataset.loadNextImage()


