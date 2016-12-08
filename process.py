#!/usr/bin/env

#################################################################################
# -> Reads in generated raw pngs and generates sub windows of fixed size. Andras
#
# -> Iterate over all windows per image and compute nn output (feed caffe from outside) Matthias
# -> if one or more windows classified as positive, consider image as positive
# -> compare result with label and write result to modelState directory
#################################################################################

import cv2
import os
import numpy as np
import math
import caffe

PNG_PATH = '../../preprocessedData/data/'
LABELS_FILE = '../../preprocessedData/labels/labels.txt'

modelDef = './train_val.prototxt'
modelParams = './caffe_alexnet_train_iter_10000.caffemodel'
gpu = False

WINDOW_SIZE = 64
NONZERO_THRESHOLD = WINDOW_SIZE * WINDOW_SIZE * 0.9

LABELS_DICT = {}

class Window:
    data = None
    confidency = 0
    def __init__(self):
        self.data = []

class Image:
    windows = None
    filename = None
    def __init__(self, filename):
        self.windows = []
        self.filename = filename

class Dataset:
    path = ''
    data = None
    size = 10 # ~ 150 GB memory
    filePointer = 0
    readPointer = 0
    windowPointer = 0
    files = []
    image = None

    def __init__(self, path):
        self.path = path
        self.data = []
        self.files = os.listdir(self.path)
        self.readNextData()

    # read size pcs of images to the memory
    def readNextData(self):
        self.data = []
        self.readPointer = 0
        for i in range(self.size):
            if self.filePointer >= len(self.files):
                return False
            filename = self.files[self.filePointer]
            self.filePointer = self.filePointer + 1
            img = cv2.imread(self.path + '/' + filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            height, width = img.shape[:2]
            image = Image(filename.split('.')[0])
            nWidth = int(width / WINDOW_SIZE)
            nHeight = int(height / WINDOW_SIZE)
            yl = 0
            yu = WINDOW_SIZE
            for h in range(nHeight):
                if yu <= height:
                    yl = yl + WINDOW_SIZE
                    yu = yu + WINDOW_SIZE
                    xl = 0
                    xr = WINDOW_SIZE
                    for w in range(nWidth):
                        if xr <= width:
                            xl = xl + WINDOW_SIZE
                            xr = xr + WINDOW_SIZE
                            window = Window()
                            window.data = img[yl : yu, xl : xr]
                            if np.count_nonzero(window.data) > NONZERO_THRESHOLD and cv2.mean(window.data)[0] < 200:
                                image.windows.append(window)
            self.data.append(image)
        return True

    def loadNextImage(self):
        self.windowPointer = 0
        # if no more images in the memory to be read
        if self.readPointer >= len(self.data):
            # read the next size pcs. of images to memory
            if not self.readNextData():
                return None
        self.image = self.data[self.readPointer : self.readPointer+1][0]
        self.readPointer = self.readPointer + 1
        return self.image

    def getNextWindow(self):
        if self.image is None:
            return None
        if self.windowPointer >= len(self.image.windows):
            return None
        window = self.image.windows[self.windowPointer : self.windowPointer+1][0]
        self.windowPointer = self.windowPointer + 1
        return window

# load labels data
with open(LABELS_FILE) as f:
    content_labels = f.readlines()   
    for i in range(0,len(content_labels)):
        nm = content_labels[i]
        nm = nm[:-1]
        label = int(nm[-1:])
        nm = nm.split('.')[0]
        LABELS_DICT[nm] = label

# Make settings
if gpu:
    caffe.set_mode_gpu()
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')

# # load net 
net = caffe.Net(modelDef, modelParams, caffe.TEST)

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

dataSet = Dataset(path=PNG_PATH)
image = dataSet.loadNextImage()
markedPositive = False
numImages = 0
while image is not None:
    numImages += 1
    print('process image: ' + image.filename)
    markedPositive = False
    window = dataSet.getNextWindow()
    while window is not None:
        data = np.array([[window.data]]).astype(np.float32)
        label = np.array([[[[1]]]]).astype(np.float32)
        net.set_input_arrays(data, label)
        net.forward()
        ft1 = net.blobs['fc8'].data
        if ft1[0][1] > 4.0:
            # one window is classified as positive
            # -> image classified as positive
            if LABELS_DICT[image.filename] == 1:
                truePositive += 1
            else:
                falsePositive += 1
            window = None
            markedPositive = True
            continue

        # move to next window
        window = dataSet.getNextWindow()

    # no window is classified as positive
    # -> image classified as negative
    if not markedPositive:
        if LABELS_DICT[image.filename] == 1:
            falseNegative += 1
        else:
            trueNegative += 1

    # move to next image
    image = dataSet.loadNextImage()

print('========RESULTS=========')
print('numImages: ' + str(numImages))
print('========================')
print('truePositives: ' + str(truePositive))
print('falsePositives: ' + str(falsePositive))
print('trueNegatives: ' + str(trueNegative))
print('falseNegatives: ' + str(falseNegative))
print('========================')
