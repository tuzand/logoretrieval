from shutil import copy2
import os
import re
import shutil
from sets import Set

inputpath = '/home/andras/data/datasets/FL27'
inputfile = 'flickr_logos_27_dataset_training_set_annotation.txt'
imagespath = 'flickr_logos_27_dataset_images'
simpleoutputpath = '/home/andras/data/datasets/FL27/FL27/data'
detectionoutputpath = '/home/andras/data/datasets/FL27/FL27_detection/data'

detectionSuffix = '_det'


def prepare(outputpath, detection):
    if os.path.exists(os.path.join(outputpath, 'Annotations')):
        shutil.rmtree(os.path.join(outputpath, 'Annotations'))
    os.makedirs(os.path.join(outputpath, 'Annotations'))
    if os.path.exists(os.path.join(outputpath, 'ImageSets')):
        shutil.rmtree(os.path.join(outputpath, 'ImageSets'))
    os.makedirs(os.path.join(outputpath, 'ImageSets'))
    imageOutPath = os.path.join(outputpath, 'Images')
    if os.path.exists(imageOutPath):
        shutil.rmtree(os.path.join(outputpath, 'Images'))
    os.makedirs(imageOutPath)

    with open(os.path.join(inputpath, inputfile)) as f:
        lines = f.readlines()
    imageList = Set([])
    for ix, line in enumerate(lines):
        l = line.split()
        if detection:
            brand = 'logo\n'
        else:
            brand = l[1].lower() + '\n'
        image = l[0]
        x1 = l[-4]
        y1 = l[-3]
        x2 = l[-2]
        y2 = l[-1]
        if detection:
            image = image.split('.')[0] + detectionSuffix + '.jpg'
        with open(os.path.join(outputpath, 'Annotations', image + '.bboxes.txt'), 'a+') as bb:
            data = bb.read()
            if not re.findall(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2), data):
                bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + brand)
        imageList.add(image.split('.')[0])

    if detection:
        filelistname = 'train.txt'
    else:
        filelistname = 'fl27_detection_train.txt'
    with open(os.path.join(outputpath, 'ImageSets', filelistname), 'w') as imset:
        for im in imageList:
            imset.write(im + '\n')

            origfile = im
            if detection:
                origfile = origfile.split('_')[0]
            file = im + '.jpg'
            copy2(os.path.join(inputpath, imagespath, origfile + '.jpg'), os.path.join(imageOutPath, file))

prepare(simpleoutputpath, False)
prepare(detectionoutputpath, True)

