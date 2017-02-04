from shutil import copy2
import os.path
import re

bboxes = 'Annotations'
images = 'Images'
imageSets = 'ImageSets'

detectionSuffix = '_det'

datasetPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2"
outputPathSimple = os.path.join(datasetPath, 'fl', 'fl_devkit', 'data')
outputPathDetection = os.path.join(datasetPath, 'fl', 'fl_devkit_detection', 'data')


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
                with open(os.path.join(datasetPath, 'classes', 'masks', brand, orig_file + '.bboxes.txt'), 'r') as bbFile:
                    data = bbFile.read()
                objs = re.findall('\d+ \d+ \d+ \d+', data)
                with open(os.path.join(bbOutputPath, file + '.bboxes.txt'), 'w') as bb:
                    for ix, obj in enumerate(objs):
                        coor = re.findall('\d+', obj)
                        x1 = coor[0]
                        y1 = coor[1]
                        w = coor[2]
                        h = coor[3]
                        x2 = str(int(coor[0]) + int(w))
                        y2 = str(int(coor[1]) + int(h))
                        bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")
                        
                    if detection:
                        bb.write('logo\n')
                    else:
                        bb.write(brand + '\n')
        if not os.path.exists(os.path.join(outputPath, imageSets)):
            os.makedirs(os.path.join(outputPath, imageSets))
        with open(os.path.join(outputPath, imageSets, phase), 'w') as imset:
            imset.write(imageList)



extractImages(datasetPath + '/testset-logosonly.relpaths.txt', 'test_logo.txt', detection = False)
extractImages(datasetPath + '/testset.relpaths.txt', 'test.txt', detection = False)
extractImages(datasetPath + '/trainset.relpaths.txt', 'train.txt', detection = False)
extractImages(datasetPath + '/trainvalset.relpaths.txt', 'trainval.txt', detection = False)
extractImages(datasetPath + '/valset-logosonly.relpaths.txt', 'val_logo.txt', detection = False)


extractImages(datasetPath + '/testset-logosonly.relpaths.txt', 'test_logo.txt', detection = True)
extractImages(datasetPath + '/testset.relpaths.txt', 'test.txt', detection = True)
extractImages(datasetPath + '/trainset.relpaths.txt', 'train.txt', detection = True)
extractImages(datasetPath + '/trainvalset.relpaths.txt', 'trainval.txt', detection = True)
extractImages(datasetPath + '/valset-logosonly.relpaths.txt', 'val_logo.txt', detection = True)

