# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:31:20 2015

@author: c
"""
import numpy as np
import os
import datetime
import glob
import itertools
import skimage as ski
import cv2
    
# Class that handles image loading, batch preparation and stuff.
class NetDataset:    
    # parameters
    imgFormat = 'jpg'
    performMeanSubtraction = False
    #scalingFactor = 1.0/255.0    
    batchSize = 250
    flipAugmentation = True
    shiftAugmentation = True
    targetSize = None
    printStatus = True
    
    # data
    data = None         # list of image data
    dataMean = None     # image data mean
    labels = None       # label for each item in image data list
    posPairs = None     # all possible pairs of items from the same class without self-pairs (positives), random order
    negPairs = None     # set of pairs of items from different class (negatives), random order
    
    # internal attributes
    labelPointer = 0
    posPointer = 0
    negPointer = 0
    epoch = 0

    meanX = 0
    meanY = 0
    imgcount = 0
    
    
    ###################
    ### public methods
    ###################
    
    # Loads images from disk. Assumes a path/class1/img1.ext structure.
    # return: -
    def loadImageData(self, path):
        
        if (os.path.isfile(path + '/data.npy') and
            os.path.isfile(path + '/labels.npy') and 
            os.path.isfile(path + '/posPairs.npy') and
            os.path.isfile(path + '/negPairs.npy')):
                
            # found previously saved chunk files, use them because loading is significantly faster this way
            self.data = np.load(path + '/data.npy')
            self.labels = np.load(path + '/labels.npy')
            self.posPairs = np.load(path + '/posPairs.npy')
            self.negPairs = np.load(path + '/negPairs.npy') 
            
        else:
            # build dataset from scratch using the single images
            
            # get all class subfolders
            subdirs = next(os.walk(path))[1]
            
            # read all images and create positive pairs
            if (self.printStatus):
                print('{:s} - Read images and create positive pairs'.format(str(datetime.datetime.now()).split('.')[0]))
            posPairs = np.zeros((0,2), dtype=np.int32)
            data = []
            labels = np.zeros(0, dtype=np.int32)
            classId = 0
            for sd in subdirs:
                # get folder name and images
                curDir = path + '/' + sd    
                print curDir
                pData = [self.loadImage(imgName) for imgName in sorted(glob.glob(curDir + '/*.' + self.imgFormat))]
		pPairs = np.asarray([p for p in itertools.combinations(range(len(pData)),2)], dtype=np.int32)
                

                # collect pairs
                if (pPairs.shape[0] != 0):
                    posPairs = np.concatenate((posPairs, pPairs + len(data)))            
                
                # collect data and labels                
                data = data + pData
                labels = np.concatenate((labels, classId * np.ones((len(pData)))))
                            
                # move to next class
                classId += 1
            # shuffle positive pairs for more stable training
            print "Height: " + str(self.meanY / self.imgcount)
            print "W: " + str(self.meanX / self.imgcount)
            np.random.shuffle(posPairs)
                
            # create negative pairs
            if (self.printStatus):
                print('{:s} - Create negative pairs'.format(str(datetime.datetime.now()).split('.')[0]))
            negPairs = np.zeros((0,2), dtype=np.int32)
            N = 3.1     # create N times as many negative pairs than positive pairs
            while (negPairs.shape[0] < round(N*posPairs.shape[0])):  
                negPairs = np.concatenate((negPairs, np.random.randint(0,labels.shape[0], (round(N*posPairs.shape[0])-negPairs.shape[0],2))))
                sames = labels[negPairs[:,0]] == labels[negPairs[:,1]]
                negPairs = np.delete(negPairs, np.where(sames), axis=0)    
                           
            # save data to class attributes    
            self.data = np.transpose(np.asarray(data), (0,3,1,2))
            self.labels = labels
            self.posPairs = posPairs
            self.negPairs = negPairs 
                
            # save data to disk for faster load on the next call
            np.save('{:s}/data.npy'.format(path), self.data)
            np.save('{:s}/labels.npy'.format(path), self.labels)
            np.save('{:s}/posPairs.npy'.format(path), self.posPairs)
            np.save('{:s}/negPairs.npy'.format(path), self.negPairs)
        
    
    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextClassificationBatch(self):
        # currently still a dummy method     
        netData = np.zeros((32,32))
        netLabels = np.zeros((32,32))
        return (netData, netLabels)
        
    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextVerficationBatch(self):
        assert(self.batchSize % 2 == 0)
        if (self.targetSize is None):
            self.targetSize = self.data.shape[2]
            
        nPosPairsBatch = self.batchSize/2
        nNegPairsBatch = nPosPairsBatch
        
        # collect positive pairs
        posPairs = self.posPairs[self.posPointer:self.posPointer+nPosPairsBatch,:]                
        posData1 = self.data[posPairs[:,0],:,:,:]
        posData2 = self.data[posPairs[:,1],:,:,:]
        posData = np.concatenate((posData1,posData2), axis=1)
        posLabels = np.ones(nPosPairsBatch, dtype=np.float32)
        
        # collect negative pairs
        negPairs = self.negPairs[self.negPointer:self.negPointer+nNegPairsBatch,:]
        negData1 = self.data[negPairs[:,0],:,:,:]
        negData2 = self.data[negPairs[:,1],:,:,:]
        negData = np.concatenate((negData1,negData2), axis=1)
        negLabels = np.zeros(nNegPairsBatch, dtype=np.float32)
        
        # combine data
        imgData = np.concatenate((posData,negData), axis=0)
        netLabels = np.concatenate((posLabels,negLabels))
        
        # data augmentation
        netData = np.zeros((imgData.shape[0], imgData.shape[1], self.targetSize, self.targetSize), dtype=np.float32)
        for f in range(netData.shape[0]):
            for c in range(netData.shape[1]):
                netData[f,c,:,:] = self.augmentImage(imgData[f,c,:,:])

        # status print
        if self.posPointer < nPosPairsBatch:
            self.epoch += 1            
            if (self.printStatus):
                print('{:s} - Starting epoch {:d}'.format(str(datetime.datetime.now()).split('.')[0], self.epoch))
            
        # move to next batch        
        self.posPointer = (self.posPointer + nPosPairsBatch) % (self.posPairs.shape[0] - nPosPairsBatch)
        self.negPointer = (self.negPointer + nNegPairsBatch) % (self.negPairs.shape[0] - nNegPairsBatch)
         
        return (netData, netLabels)

        
    #####################    
    ### private methods
    #####################
        
    # load image, convert to grayscale and scale to [0,1]
    # return: img
    def loadImage(self, filename):        
        # read image
        #img = ski.io.imread(filename)
        img = cv2.imread(filename)
        print filename
        h, w = img.shape[:2]
        self.meanY += h
        self.meanX += w
        self.imgcount += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.resize(img, (32,32,3))
        # convert to color if needed
	if (img.ndim == 2):
            img = img[:,:,np.newaxis]
        if (img.shape[2] == 1):
            img = np.resize(img, (img.shape[0], img.shape[1], 3))
            
        # convert to float datatype
        img = ski.img_as_float(img).astype(np.float32)
        return img
        
    # Method for dataset augmentation. Performs random flipping and shifting
    # return: image 
    def augmentImage(self, image):
        assert((self.targetSize is not None) or not self.shiftAugmentation)
        # perform flipping
        if (self.flipAugmentation and np.random.randint(0,2) < 1):
            image = np.fliplr(image)
            
        # perform shifting
        padSize = image.shape[0] - self.targetSize;
        if (self.shiftAugmentation and padSize > 0):
            # random shift + cut
            offsetX = np.random.randint(0, padSize+1);    # random number between 0 and padSize
            offsetY = np.random.randint(0, padSize+1);    # random number between 0 and padSize
            image = image[offsetY:self.targetSize+offsetY,offsetX:self.targetSize+offsetX];
        elif (padSize > 0):
            # center cut
            offset = np.rint(padSize/2)
            image = image[offset:self.targetSize+offset,offset:self.targetSize+offset];
            
        return image        

