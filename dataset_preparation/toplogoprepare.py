from shutil import copy2
import shutil
import os
import re

inputpath = '/home/andras/data/datasets/toplogo'
maskpath = 'masks'
imagepath = 'jpg'
simpleoutputpath = '/home/andras/data/datasets/toplogo/toplogo/data'
detectionoutputpath = '/home/andras/data/datasets/toplogo/toplogo_detection/data'

detectionSuffix = '_det'

def prepare(outputpath, detection):
    annotationspath = os.path.join(outputpath, 'Annotations')
    if os.path.exists(annotationspath):
        shutil.rmtree(annotationspath)
    os.makedirs(annotationspath)
    imagesetspath = os.path.join(outputpath, 'ImageSets')
    if os.path.exists(imagesetspath):
        shutil.rmtree(imagesetspath)
    os.makedirs(imagesetspath)
    imageOutPath = os.path.join(outputpath, 'Images')
    if os.path.exists(imageOutPath):
        shutil.rmtree(imageOutPath)
    os.makedirs(imageOutPath)

    for root, dirs, files in os.walk(os.path.join(inputpath, maskpath)):
        path = root.split(os.sep)
        if detection:
            brand = 'logo\n'
        else:
            
            brand = ('adidas' if (path[-1] == 'adidas0') else path[-1]) + '\n'
        for file in files:
            filepath = os.path.join(root, file)
            with open(filepath, "r") as bbfile:
                data = bbfile.read()
            lines = re.findall('\d+ \d+ \d+ \d+', data)
            if detection:
                dstfile = file.split('.')[0] + detectionSuffix + '.jpg.bboxes.txt'
            else:
                dstfile = file
            with open(os.path.join(annotationspath, dstfile), 'w') as bb:
                for ix, obj in enumerate(lines):
                    coor = re.findall('\d+', obj)
                    x1 = coor[0]
                    y1 = coor[1]
                    w = coor[2]
                    h = coor[3]
                    x2 = str(int(coor[0]) + int(w))
                    y2 = str(int(coor[1]) + int(h))
                    bb.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + brand)

    for root, dirs, files in os.walk(os.path.join(inputpath, imagepath)):
        path = root.split(os.sep)
        if detection:
            brand = 'logo\n'
        else:
            brand = path[-1] + '\n'
        for file in files:
            if not file.endswith('.jpg') and not file.endswith('.JPG'):
                continue
            filepath = os.path.join(root, file)
            
            if detection:
                dstfile = file.split('.')[0] + detectionSuffix + '.jpg'
            else:
                dstfile = file.split('.')[0] + '.jpg' # JPG -> jpg

            copy2(filepath, os.path.join(imageOutPath, dstfile))
            if detection:
                filelistname = 'toplogo_detection_train.txt'
            else:
                filelistname = 'toplogo_train.txt'
            with open(os.path.join(imagesetspath, filelistname), 'a+') as imset:
                imset.write(dstfile.split('.')[0] + '\n')

    

prepare(simpleoutputpath, False)
prepare(detectionoutputpath, True)
