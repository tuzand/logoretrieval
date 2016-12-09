#################################################################################
# -> 
#################################################################################

import sys
import cv2
import os
import numpy as np
import math
import caffe
import skimage as ski

LABELS_DICT = {}

class Window:
    data = None
    def __init__(self):
        self.data = []

class Image:
    windows = None
    filePath = None
    def __init__(self, path):
        self.windows = []
        self.filePath = path

class Dataset:
    data = None
    imagePointer = 0
    windowPointer = 0
    files = []

    def __init__(self, path, imageSize, onlyLogos = True):
        self.data = []
        self.loadImages(path, onlyLogos, imageSize)

    def readWindows(self, filePath, windowSize):
        image = Image(filePath)
        #img = ski.io.imread(filename)
        img = cv2.imread(filePath)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = img.shape[:2]
        nWidth = int(width / windowSize)
        nHeight = int(height / windowSize)
        yl = 0
        yu = windowSize
        for h in range(nHeight):
            if yu <= height:
                yl = yl + windowSize
                yu = yu + windowSize
                xl = 0
                xr = windowSize
                for w in range(nWidth):
                    if xr <= width:
                        xl = xl + windowSize
                        xr = xr + windowSize
                        window = img[yl : yu, xl : xr]
                        image.windows.append(window)
        self.data.append(image)

    def readCompleteImage(self, filePath, imageSize):
        image = Image(filePath)
        img = cv2.imread(filePath)
        img = cv2.resize(img,(imageSize, imageSize), interpolation = cv2.INTER_CUBIC)
        img = np.array([img]).astype(np.float32)
        img = np.transpose(np.asarray(img), (0,3,1,2))
        img = np.ascontiguousarray(img, dtype=np.float32)
        image.windows.append(img)
        self.data.append(image)

    # reads images to the memory
    def loadImages(self, path, onlyLogos, imageSize):
        i = 0
        for subdir, dirs, files in os.walk(path):
            for f in files:
                filename = os.path.join(subdir, f)
                i = i + 1
                if i % 100 == 0:
                    print i
                if onlyLogos and subdir.split('/')[-1] == 'no-logo':
                    continue
                self.readCompleteImage(filename, imageSize)
        self.imagePointer = 0
        print i

    def addImages(self, path):
        self.loadImages(path)

    def loadNextImage(self):
        self.windowPointer = 0
        # if no more images in the memory to be read
        if self.imagePointer >= len(self.data):
            return None
        self.imagePointer = self.imagePointer + 1

    def hasMoreImage(self):
        return True if not (self.data[self.imagePointer] == None) else False

    def reset(self):
        self.imagePointer = 0
        self.windowPointer = 0

    def getNextWindow(self):
        image = self.data[self.imagePointer]
        if image is None:
            return None
        if self.windowPointer >= len(image.windows):
            return None
        window = image.windows[self.windowPointer]
        self.windowPointer = self.windowPointer + 1
        return window

    def getActFilePath(self):
        return self.data[self.imagePointer].filePath
    def getActFileName(self):
        filePath = self.getActFilePath()
        return filePath.split('/')[-1]

