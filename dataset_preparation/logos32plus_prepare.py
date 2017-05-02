import numpy as np
import os
import shutil
from shutil import copy2
import cv2
import scipy.io as sio

gt = sio.loadmat('/home/andras/data/datasets/L32P/groundtruth.mat')['groundtruth'][0]
outputpath = '/home/andras/data/datasets/L32P'

if os.path.exists(os.path.join(outputpath, 'classification')):
    shutil.rmtree(os.path.join(outputpath, 'classification'))
if os.path.exists(os.path.join(outputpath, 'detection')):
    shutil.rmtree(os.path.join(outputpath, 'detection'))

bboxes = dict()
imagelist = list()

for idx, values in enumerate(gt):
    brand = values[0][0].split('\\')[0]
    writebrand = brand.lower()
    if writebrand == 'guinness':
        writebrand = 'guiness'
    filename = values[0][0].split('\\')[1]
    filenamewoext = filename.split('.')[0]
    boxes = values[1]

    im = cv2.imread(os.path.join(outputpath, 'images', brand, filename))
    height, width = im.shape[:2]
    for box in boxes:
        x1 = max(0, int(round(float(box[0]))))
        y1 = max(0, int(round(float(box[1]))))
        w = int(round(float(box[2])))
        h = int(round(float(box[3])))
        x2 = min(width, x1 + w)
        y2 = min(height, y1 + h)
        if (x2 < x1) | (y2 < y1):
            print str(x1) + ' ' + str(x2)
            print str(y1) + ' ' + str(y2)
            print filename
            continue
        if not os.path.exists(os.path.join(outputpath, 'classification', 'data', 'Annotations', writebrand)):
            os.makedirs(os.path.join(outputpath, 'classification', 'data', 'Annotations', writebrand))
        if not os.path.exists(os.path.join(outputpath, 'detection', 'data', 'Annotations', writebrand)):
            os.makedirs(os.path.join(outputpath, 'detection', 'data', 'Annotations', writebrand))
        with open(os.path.join(outputpath, 'classification', 'data', 'Annotations', writebrand, filename + '.bboxes.txt'), 'a+') as bb:
            bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + writebrand + "\n")
        with open(os.path.join(outputpath, 'detection', 'data', 'Annotations', writebrand, filenamewoext + '_det.jpg.bboxes.txt'), 'a+') as bb:
            bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " logo" + "\n")
        imagelist.append(writebrand + '/' + filename.split('.')[0])
    if not os.path.exists(os.path.join(outputpath, 'detection', 'data', 'Images', writebrand)):
        os.makedirs(os.path.join(outputpath, 'detection', 'data', 'Images', writebrand))
    copy2(os.path.join(outputpath, 'images', brand, filename), os.path.join(outputpath, 'detection', 'data', 'Images', writebrand, filenamewoext + '_det.jpg'))
    #print str(idx) + '/' + str(len(gt))

if not os.path.exists(os.path.join(outputpath, 'classification', 'data', 'ImageSets')):
    os.makedirs(os.path.join(outputpath, 'classification', 'data', 'ImageSets'))
if not os.path.exists(os.path.join(outputpath, 'detection', 'data', 'ImageSets')):
    os.makedirs(os.path.join(outputpath, 'detection', 'data', 'ImageSets'))
with open(os.path.join(outputpath, 'classification', 'data', 'ImageSets', 'logos32plus.txt'), 'w') as clsfile:
    with open(os.path.join(outputpath, 'detection', 'data', 'ImageSets', 'logos32plus_detection.txt'), 'w') as detfile:
        for image in set(imagelist):
            clsfile.write(image + '\n')
            detfile.write(image + '_det\n')

