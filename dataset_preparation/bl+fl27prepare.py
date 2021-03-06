from shutil import copy2
import os
import re
import shutil
from sets import Set

blflIntersect = ["adidas", "adidas-text", "cocacola", "ferrari", "shell", "stellaartois"]
fl27fl32Intersect = ["adidas", "apple", "bmw", "cocacola", "dhl", "fedex", "ferrari", "ford", "google", "heineken", "hp", "pepsi", "starbucks", "texaco"]

bl_inputpath = '/home/andras/data/datasets/BL'
bl_imagespath = 'images'
bl_inputfile = 'qset3_internal_and_local.qry'
bl_simpleoutputpath = '/home/andras/data/datasets/BL/BL/data'
bl_detectionoutputpath = '/home/andras/data/datasets/BL/BL_detection/data'

flbl_inputpath = '/home/andras/data/datasets/FLBL'
flbl_imagespath = 'images'
flbl_inputfile = 'groundtruth.txt'
flbl_simpleoutputpath = '/home/andras/data/datasets/FLBL/FLBL/data'
flbl_detectionoutputpath = '/home/andras/data/datasets/FLBL/FLBL_detection/data'


fl27_inputpath = '/home/andras/data/datasets/FL27'
fl27_inputfile = 'flickr_logos_27_dataset_training_set_annotation.txt'
fl27_imagespath = 'flickr_logos_27_dataset_images'
fl27_simpleoutputpath = '/home/andras/data/datasets/FL27/FL27/data'
fl27_detectionoutputpath = '/home/andras/data/datasets/FL27/FL27_detection/data'

detectionSuffix = '_det'

def bl(detection, l):
    brand = l[1].lower() + '\n'
    image = l[2]
    x1 = int(l[-4])
    y1 = int(l[-3])
    x2 = int(l[-2])
    y2 = int(l[-1])
    #x2 = str(int(x1) + int(w))
    #y2 = str(int(y1) + int(h))
    return brand, image, x1, y1, x2, y2

def fl27(detection, l):
    brand = l[1].lower() + '\n'
    image = l[0]
    x1 = int(l[-4])
    y1 = int(l[-3])
    x2 = int(l[-2])
    y2 = int(l[-1])
    return brand, image, x1, y1, x2, y2


def prepare(inputpath, imagespath, inputfile, outputpath, detection, dataset):
    if os.path.exists(os.path.join(outputpath, 'Annotations')):
        shutil.rmtree(os.path.join(outputpath, 'Annotations'))
    os.makedirs(os.path.join(outputpath, 'Annotations'))
    if os.path.exists(os.path.join(outputpath, 'ImageSets')):
        shutil.rmtree(os.path.join(outputpath, 'ImageSets'))
    os.makedirs(os.path.join(outputpath, 'ImageSets'))
    imageOutPath = os.path.join(outputpath, 'Images')
    if os.path.exists(imageOutPath):
        shutil.rmtree(imageOutPath)
    os.makedirs(imageOutPath)

    with open(os.path.join(inputpath, inputfile)) as f:
        lines = f.readlines()
    imageList = Set([])
    invalid = 0
    valid = 0
    brands = list()
    imageBlackList = list()
    for ix, line in enumerate(lines):
        l = line.split()
        brand, image, x1, y1, x2, y2 = dataset(detection, l)
        b = brand.split()[0]
        if any(b in i for i in blflIntersect) or any(b in j for j in fl27fl32Intersect) or any(image in k for k in imageBlackList):
            imageBlackList.append(image)
            continue
        if detection:
            brand = 'logo\n'
        brands.append(brand)
        if x1 > x2 or y1 > y2:
            invalid += 1
            continue
        valid += 1
        if detection:
            image = image.split('.')[0] + detectionSuffix + '.jpg'
        with open(os.path.join(outputpath, 'Annotations', image + '.bboxes.txt'), 'a+') as bb:
            data = bb.read()
            if not re.findall(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2), data):
                bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + brand)
        imageList.add(image.split('.')[0])

    print 'Valid: ' + str(valid)
    print 'Invalid: ' + str(invalid)

    if detection:
        filelistname = dataset.__name__ + '_detection_train.txt'
    else:
        filelistname = dataset.__name__ + '_train.txt'
    with open(os.path.join(outputpath, 'brands.txt'), 'w') as bfile:
        brands = Set(brands)
        for br in brands:
            bfile.write(br)
    with open(os.path.join(outputpath, 'ImageSets', filelistname), 'w') as imset:
        for im in imageList:
            imset.write(im + '\n')

            origfile = im
            if detection:
                origfile = origfile.split('_')[0]
            file = im + '.jpg'
            if os.path.exists(os.path.join(inputpath, imagespath, origfile + '.jpg')):
                copy2(os.path.join(inputpath, imagespath, origfile + '.jpg'), os.path.join(imageOutPath, file))
            else:
               print os.path.join(inputpath, imagespath, origfile + '.jpg') + ' not found'

prepare(bl_inputpath, bl_imagespath, bl_inputfile, bl_simpleoutputpath, False, bl)
prepare(bl_inputpath, bl_imagespath, bl_inputfile, bl_detectionoutputpath, True, bl)

#prepare(flbl_inputpath, flbl_imagespath, flbl_inputfile, flbl_simpleoutputpath, False, bl)
#prepare(flbl_inputpath, flbl_imagespath, flbl_inputfile, flbl_detectionoutputpath, True, bl)

prepare(fl27_inputpath, fl27_imagespath, fl27_inputfile, fl27_simpleoutputpath, False, fl27)
prepare(fl27_inputpath, fl27_imagespath, fl27_inputfile, fl27_detectionoutputpath, True, fl27)
