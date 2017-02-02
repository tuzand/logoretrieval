from shutil import copy2
import os.path


dataSetPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2"
outputPathDetection = "/home/andras/data/datasets/FL32/FlickrLogos-v2/split_detection"
outputPathSimple = "/home/andras/data/datasets/FL32/FlickrLogos-v2/split"
bbOutputPath = os.path.join(dataSetPath, 'fl', 'fl_devkit', 'data', 'Annotations')

def extractImages(fileListPath, phase, nologo, detection, abs):
    if detection == True:
        outputPath = outputPathDetection
    else:
        outputPath = outputPathSimple
    with open(fileListPath, "r") as fileList:
        lines = fileList.read().splitlines()
        for f in lines:
            splittedPath = f.split('/')
            dirName = splittedPath[-2]
            if ( (dirName != 'no-logo') or (nologo == True) ):
                file = splittedPath[-1]
                if not os.path.exists(outputPath + "/" + phase + "/" + dirName):
                    os.makedirs(outputPath + "/" + phase + "/" + dirName)
                copy2(dataSetPath + "/" + f, outputPath + "/" + phase + "/" + dirName)
                if dirName != 'no-logo':
                    if not os.path.exists(bbOutputPath):
                        os.makedirs(bbOutputPath)                
                    with open(dataSetPath + "/classes/masks/"+ dirName + "/" + file + ".bboxes.txt", 'r') as bbFile:
                        data = bbFile.read()
                    import re
                    objs = re.findall('\d+ \d+ \d+ \d+', data)
                    with open(os.path.join(bbOutputPath, file + '.bboxes.txt'), 'w') as bb:
                        #bb.write("x y width height\n")
                        for ix, obj in enumerate(objs):
                            # Make pixel indexes 0-based
                            coor = re.findall('\d+', obj)
                            x1 = coor[0]
                            y1 = coor[1]
                            w = coor[2]
                            h = coor[3]
                            x2 = str(int(coor[0]) + int(w))
                            y2 = str(int(coor[1]) + int(h))
                        
                        if abs == True:
                            bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")
                        else:
                            bb.write(str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + "\n")
                        if detection == True:
                            bb.write('logo\n')
                        else:
                            bb.write(dirName + '\n')



extractImages(dataSetPath + "/testset.relpaths.txt", "test", nologo=True, detection = False, abs = False)
extractImages(dataSetPath + "/trainset.relpaths.txt", "train", nologo=False, detection = False, abs = False)
extractImages(dataSetPath + "/valset.relpaths.txt", "val", nologo=False, detection = False, abs = False)
