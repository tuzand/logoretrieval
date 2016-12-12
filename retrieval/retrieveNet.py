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
import os
                   

# parameters

#MAINPATH = '/home/atuezkoe/CaffeData/data/'
MAINPATH = '/home/andras/data/'


TRAINPATH = MAINPATH + 'datasets/FL32/FlickrLogos-v2/splitted/train/'
VALPATH = MAINPATH + 'datasets/FL32/FlickrLogos-v2/splitted/val/'
TESTPATH = MAINPATH + 'datasets/FL32/FlickrLogos-v2/splitted/test/'

RESULTPATH = './result/'
RESULTPOSTFIX = '.result2.txt'

IMAGESIZE = 224
modelDef = '../models/VGG_ILSVRC_16_layers_deploy.prototxt'
modelParams = MAINPATH + 'models/VGG_ILSVRC_16_layers.caffemodel'
gpu = True
GPUID = 5

# Make settings
if gpu:
    caffe.set_mode_gpu()
    caffe.set_device(GPUID);
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')


### step 1: load data
print('{:s} - Load test data'.format(str(datetime.datetime.now()).split('.')[0]))
testDataset = Dataset(TRAINPATH, IMAGESIZE, onlyLogos = False)
testDataset.addImages(VALPATH)

queryDataset = Dataset(TESTPATH, IMAGESIZE, onlyLogos = True)


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 3: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))


labels = []
distances = []
result = dict()
q = 0
while queryDataset.hasMoreImage():
    queryWindow = queryDataset.getNextWindow()
    label = np.array([[[[1]]]]).astype(np.float32)
    print('{:s} - Calculating similarities'.format(str(datetime.datetime.now()).split('.')[0]))
    print q
    q = q + 1

    net.set_input_arrays(queryWindow, label)
    net.forward()
    queryFeature = net.blobs['pool5'].data
    queryFeature = queryFeature.flatten()
    queryNorm = np.linalg.norm(queryFeature)
    normQueryFeature = queryFeature / queryNorm

    testDataset.reset()
    minDistance = float("inf")

    while testDataset.hasMoreImage():
        testWindow = testDataset.getNextWindow()
        net.set_input_arrays(testWindow, label)
        net.forward()
        testFeature = net.blobs['pool5'].data
        result[testDataset.getActFilePath()] = testFeature.flatten()
        testDataset.loadNextImage()

    for key, value in result.items():
        testNorm = np.linalg.norm(value)
        normTestFeature = value / testNorm
        similarity = np.dot(normQueryFeature, normTestFeature)
        result[key] = similarity

    sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    if not os.path.exists(RESULTPATH):
        os.makedirs(RESULTPATH)
    with open(os.path.join(RESULTPATH, queryDataset.getActFileName() + RESULTPOSTFIX), 'w') as res:
        for i in range(len(sortedResult)):
            out = str(sortedResult[i][0]) + " " + str(sortedResult[i][1]) + "\n"
            res.write(out)


    

    queryDataset.loadNextImage()


