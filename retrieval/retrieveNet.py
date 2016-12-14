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
MAINPATH = '/home/pp2015/pp2015_2/data/'
#MAINPATH = '/home/andras/data/datasets/'


TRAINPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/train/'
VALPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/val/'
TESTPATH = MAINPATH + 'FL32/FlickrLogos-v2/splitted/test/'

RESULTPATH = '../results/resnet2/'
RESULTPOSTFIX = '.result2.txt'

IMAGESIZE = 224
FEATURELAYER = 'pool5'
#modelDef = '../models/VGG_ILSVRC_16_layers.caffemodel'
#modelParams = MAINPATH + 'models/VGG_ILSVRC_16_layers.caffemodel'
modelDef = '../models/ResNet-152-deploy.prototxt'
modelParams = MAINPATH + '../models/resnet/ResNet-152-model.caffemodel'
#modelDef = '../models/imagenet_googlenet_train_val_googlenet.prototxt'
#modelParams = MAINPATH + 'models/imagenet_googlenet.caffemodel'
#modelDef = '../models/places_googlenet_train_val_googlenet.prototxt'
#modelParams = MAINPATH + 'models/places_googlenet.caffemodel'

gpu = True
GPUID = 0

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

print('{:s} - Load query data'.format(str(datetime.datetime.now()).split('.')[0]))
queryDataset = Dataset(TESTPATH, IMAGESIZE, onlyLogos = True)


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 3: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))

normedTestFeatures = dict()
result = dict()
q = 0

label = np.array([[[[1]]]]).astype(np.float32)


while testDataset.hasMoreImage():
        testWindow = testDataset.getNextWindow()
        net.set_input_arrays(testWindow, label)
        net.forward()
        testFeature = net.blobs[FEATURELAYER].data
        print np.shape(testFeature)
        testFeature = testFeature.flatten()
        testNorm = np.linalg.norm(testFeature)
        testFeature = testFeature / testNorm
        normedTestFeatures[testDataset.getActFilePath()] = testFeature
        testDataset.loadNextImage()

while queryDataset.hasMoreImage():
    queryWindow = queryDataset.getNextWindow()
    label = np.array([[[[1]]]]).astype(np.float32)
    print('{:s} - Calculating similarities'.format(str(datetime.datetime.now()).split('.')[0]))
    print q
    q = q + 1

    net.set_input_arrays(queryWindow, label)
    net.forward()
    queryFeature = net.blobs[FEATURELAYER].data
    queryFeature = queryFeature.flatten()
    queryNorm = np.linalg.norm(queryFeature)
    normedQueryFeature = queryFeature / queryNorm

    for filePath, normedTestFeature in normedTestFeatures.items():
        result[filePath] = np.dot(normedQueryFeature, normedTestFeature)

    sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    if not os.path.exists(RESULTPATH):
        os.makedirs(RESULTPATH)
    with open(os.path.join(RESULTPATH, queryDataset.getActFileName() + RESULTPOSTFIX), 'w') as res:
        res.write(queryDataset.getActFilePath() + " " + str(1.0))
        for i in range(len(sortedResult)):
            out = str(sortedResult[i][0]) + " " + str(sortedResult[i][1]) + "\n"
            res.write(out)
    queryDataset.loadNextImage()
