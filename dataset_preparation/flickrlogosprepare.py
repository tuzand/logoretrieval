from shutil import copy2
import os.path
import re
import shutil

bboxes = 'Annotations'
images = 'Images'
imageSets = 'ImageSets'

detectionSuffix = '_det'

datasetPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2"
outputPathSimple = os.path.join(datasetPath, 'fl', 'fl', 'data')
outputPathDetection = os.path.join(datasetPath, 'fl', 'fl_detection', 'data')


def extractImages(fileListPath, phase, detection):
    if detection:
        outputPath = outputPathDetection
    else:
        outputPath = outputPathSimple
    with open(fileListPath, "r") as fileList:
        lines = fileList.read().splitlines()
        imageList = ''
        for f in lines:
            splitPath = f.split('/')
            brand = splitPath[-2]
            file = splitPath[-1]
            orig_file = file
            if detection:
                file = file.split('.')[0] + detectionSuffix + '.jpg'

            imageOutPath = os.path.join(outputPath, images)
            if not os.path.exists(imageOutPath):
                os.makedirs(imageOutPath)
            copy2(os.path.join(datasetPath, f), os.path.join(imageOutPath, file))

            imageList += file.split('.')[0] + '\n'

            bbOutputPath = os.path.join(outputPath, bboxes)
            if not os.path.exists(bbOutputPath):
                os.makedirs(bbOutputPath)
            if brand != 'no-logo':
                with open(os.path.join(datasetPath, 'classes', 'masks', brand.lower(), orig_file + '.bboxes.txt'), 'r') as bbFile:
                    data = bbFile.read()
                objs = re.findall('\d+ \d+ \d+ \d+', data)
                with open(os.path.join(bbOutputPath, file + '.bboxes.txt'), 'w') as bb:
                    if detection:
                        actBrand = 'logo\n'
                    else:
                        actBrand = brand.lower() + '\n'
                    for ix, obj in enumerate(objs):
                        coor = re.findall('\d+', obj)
                        x1 = coor[0]
                        y1 = coor[1]
                        w = coor[2]
                        h = coor[3]
                        x2 = str(int(coor[0]) + int(w))
                        y2 = str(int(coor[1]) + int(h))
                        bb.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + actBrand)
                        
        if not os.path.exists(os.path.join(outputPath, imageSets)):
            os.makedirs(os.path.join(outputPath, imageSets))
        with open(os.path.join(outputPath, imageSets, phase), 'w') as imset:
            imset.write(imageList)


if os.path.exists(outputPathSimple):
    shutil.rmtree(outputPathSimple)
if os.path.exists(outputPathDetection):
    shutil.rmtree(outputPathDetection)

extractImages(datasetPath + '/testset-logosonly.relpaths.txt', 'fl_test_logo.txt', detection = False)
extractImages(datasetPath + '/testset.relpaths.txt', 'fl_test.txt', detection = False)
extractImages(datasetPath + '/trainset.relpaths.txt', 'fl_train.txt', detection = False)
extractImages(datasetPath + '/trainvalset.relpaths.txt', 'fl_trainval.txt', detection = False)
extractImages(datasetPath + '/valset-logosonly.relpaths.txt', 'fl_val_logo.txt', detection = False)


extractImages(datasetPath + '/testset-logosonly.relpaths.txt', 'fl_detection_test_logo.txt', detection = True)
extractImages(datasetPath + '/testset.relpaths.txt', 'fl_detection_test.txt', detection = True)
extractImages(datasetPath + '/trainset.relpaths.txt', 'fl_detection_train.txt', detection = True)
extractImages(datasetPath + '/trainvalset.relpaths.txt', 'fl_detection_trainval.txt', detection = True)
extractImages(datasetPath + '/valset-logosonly.relpaths.txt', 'fl_detection_val_logo.txt', detection = True)

