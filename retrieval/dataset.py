#################################################################################
# -> 
#################################################################################

import cv2
import os
import numpy as np
import math
import caffe

LABELS_DICT = {}

class Window:
    data = None
    confidency = 0
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
    image = None

    def __init__(self, path):
        self.data = []
        self.loadImages(path)

    def readWindows(self, filePath, windowSize):
        image = Image(filePath)
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

    def readCompleteImage(self, filePath):
        image = Image(filePath)
        image.windows.append(cv2.imread(filePath))
        self.data.append(image)

    # reads images to the memory
    def loadImages(self, path):
        i = 0
        for subdir, dirs, files in os.walk(path):
            for f in files:
                filename = os.path.join(subdir, f)
                i = i + 1
                print filename
                self.readCompleteImage(filename)
        self.imagePointer = 0
        print i 

    def addImages(self, path):
        self.loadImages(path)

    def loadNextImage(self):
        self.windowPointer = 0
        # if no more images in the memory to be read
        if self.imagePointer >= len(self.data):
            return None
        self.image = self.data[self.imagePointer : self.imagePointer+1][0]
        print self.data[self.imagePointer : self.imagePointer+1][0]
#        print self.data[self.imagePointer]
        self.imagePointer = self.imagePointer + 1

    def hasMoreImage(self):
        return True if not (self.data[self.imagePointer]) else False

    def getNextWindow(self):
        if self.image is None:
            return None
        if self.windowPointer >= len(self.image.windows):
            return None
        window = self.image.windows[self.windowPointer : self.windowPointer+1][0]
        self.windowPointer = self.windowPointer + 1
        return window

