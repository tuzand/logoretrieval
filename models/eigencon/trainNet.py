# -*- coding: utf-8 -*-
"""
Training with data from memory, inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/tutorial/examples/01-learning-lenet.ipynb
Useful because of siamese setup: avoids unnecessary disk access.

Created on Wed Sep  2 17:22:21 2015

@author: ch
"""
import matplotlib
matplotlib.use('Agg')
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import caffe
import uuid
import os
from NetDataset import NetDataset


# plot training loss and test results
def plotStats(trainLoss, testAcc, figure=None):

    nLoss = trainLoss.shape[0]
    nAcc = testAcc.shape[0]    
    
    if (figure is None):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  
    else:        
        fig,ax1,ax2 = figure
    
    if (nAcc > 1):
        factor = round((nLoss-1)/(nAcc-1))
        
        ax1.plot(np.arange(nLoss), trainLoss)
        ax2.plot(factor * np.arange(nAcc), testAcc, 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')
        
    return [fig,ax1,ax2]
                   

### task: train siamese network in python
# parameters

startIteration = 0
nTrainIter = 50000 - startIteration
testInterval = 200
nTestBatches = 100

imgDim = 32  # width and height of image
identifier = 'eigenT'

dataFolder = '/home/andras/data/datasets/logorois'
solverDef = '/home/andras/data/models/eigencon/solver.prototxt'
#solverState = 'snapshots/' + str(identifier) + '_' + str(imgDim) + '_iter_' + str(startIteration) + '.solverstate'
#modelParams = 'snapshots/' + str(identifier) + '_' + str(imgDim) + '_iter_' + str(startIteration) + '.caffemodel'

uuId = uuid.uuid4()

gpu = True 

# make settings
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('logs/'+str(uuId)):
    os.makedirs('logs/'+str(uuId))

if not os.path.exists('plots/'):
    os.makedirs('plots/')
if not os.path.exists('plots/'+str(uuId)):
    os.makedirs('plots/'+str(uuId))

if not os.path.exists('snapshots/'):
    os.makedirs('snapshots/')

dataFolderTrain = dataFolder + '/train'
dataFolderTest = dataFolder + '/val'
if gpu:
    caffe.set_mode_gpu()
    caffe.set_device(5)
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')


### step 1: load data
print('{:s} - Load train data'.format(str(datetime.datetime.now()).split('.')[0]))
trainDataset = NetDataset()
trainDataset.targetSize = imgDim
trainDataset.loadImageData(dataFolderTrain)
trainDataset.printStatus = False

print('{:s} - Load test data'.format(str(datetime.datetime.now()).split('.')[0]))
testDataset = NetDataset()
testDataset.targetSize = imgDim
testDataset.flipAugmentation = False
testDataset.shiftAugmentation = False
testDataset.loadImageData(dataFolderTest)
testDataset.printStatus = False


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
solver = caffe.SGDSolver(solverDef)


# The next two line are for restoring training from a snapshot
if startIteration > 0:
    print('Loading solverstate')
    solver.restore(solverState)
    solver.net.copy_from(modelParams)
    solver.test_nets[0].share_with(solver.net)
    print('Solverstate loaded')

# print net structure
# each output is (batch size, feature dim, spatial dim)
print('                      Net data/blob sizes:')
for k, v in solver.net.blobs.items():
    print '                       ', k, v.data.shape
print('                      Net weight sizes:')
for k, v in solver.net.params.items():
    print '                       ', k, v[0].data.shape


### step 3: train net with dynamically created data
print('{:s} - Training'.format(str(datetime.datetime.now()).split('.')[0]))

# loss and accuracy will be stored in the log
trainLoss = np.zeros(nTrainIter+1)
testAcc = np.zeros(int(np.ceil(nTrainIter / testInterval))+1)

# the main solver loop
testNetData, testNetLabels = testDataset.getNextVerficationBatch()
solver.test_nets[0].set_input_arrays(testNetData, testNetLabels)
for trainIter in range(nTrainIter+1):
    netData, netLabels = trainDataset.getNextVerficationBatch()
    solver. net.set_input_arrays(netData, netLabels)
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    trainLoss[trainIter] = solver.net.blobs['loss'].data
    
    # abort if training diverged
    if (math.isnan(trainLoss[trainIter])):
        print('training diverged')
        break
        
    # run a full test every so often
    # Caffe can also do this for us and write to a log, but we show here
    # how to do it directly in Python, where more complicated things are easier.
    if (trainIter % testInterval == 0):
        print('{:s} - Iteration {:d} - Loss {:.4f} - Testing'.format(str(datetime.datetime.now()).split('.')[0], trainIter + startIteration, trainLoss[trainIter]))
        labels = []
        distances = []
                
        # testing with memory interface
        # collect test results
        testDataset.posPointer = 0
        testDataset.negPointer = 0
        for i in range(nTestBatches):
            testNetData, testNetLabels = testDataset.getNextVerficationBatch()
            solver.test_nets[0].set_input_arrays(testNetData, testNetLabels)
            solver.test_nets[0].forward()
        
            # accuracy test for verification
            ft1 = solver.test_nets[0].blobs['feat'].data
            ft2 = solver.test_nets[0].blobs['feat_p'].data
            curDist = np.sum((ft1 - ft2)**2, axis=1)     # euclidean distance between sample pairs
            
            labels = np.concatenate((labels, testNetLabels))
            distances = np.concatenate((distances, curDist))    
    
        # search for best threshold and use that accuracy
        bestAccuracy = -float('inf')
        for i in range(len(labels)):
            curThresh = distances[i]
            prediction = distances <= curThresh
            accuracy = np.mean(prediction == labels)
            if (accuracy > bestAccuracy):
                bestAccuracy = accuracy
                thresh = curThresh  
    
        testAcc[trainIter // testInterval] = bestAccuracy
        print('                      accuracy: {:.3f}'.format(bestAccuracy))       
	
        size = int(np.ceil(trainIter / testInterval))+1
        tempAcc = np.zeros(size)
        tempIter = np.zeros(size)
        for i in range(size):
            tempAcc[i] = testAcc[i] * 100
            tempIter[i] = i * testInterval
        plt.plot(tempIter, tempAcc)
        plt.savefig('plots/' + str(uuId) + '/after_iter_' + str(trainIter) + '.png')

        # write log
        with open ("logs/" + str(uuId) + "/log", "a") as valuesFile:
            valuesFile.write('RESULT id=' + str(uuId) + ' imgDim=' + str(imgDim) + ' loss=' + str(trainLoss[trainIter]) + ' acccuracy=' + str(bestAccuracy) + ' iterations=' + str(trainIter + 
startIteration) + '\n') 
        
    # end testing
# end train loop

## plot final stats
plotStats(trainLoss, testAcc)    

## save plot stats
np.save('temp/lrfacenet_{:s}_loss.npy'.format(str(datetime.datetime.now()).split('.')[0]).replace(':','-'), trainLoss)
np.save('temp/lrfacenet_{:s}_accuracy.npy'.format(str(datetime.datetime.now()).split('.')[0]).replace(':','-'), testAcc)
            
# done
print('{:s} - Done'.format(str(datetime.datetime.now()).split('.')[0]))

