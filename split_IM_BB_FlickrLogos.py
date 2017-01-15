from shutil import copy2
import os.path


dataSetPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2"
outputPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2/splitted"
bbOutputPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2/splitted/bboxes"

def extractImages(fileListPath, phase, nologo):
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
                    if not os.path.exists(bbOutputPath + "/" + phase + "/" + dirName):
                        os.makedirs(bbOutputPath + "/" + phase + "/" + dirName)                
                    with open(dataSetPath + "/classes/masks/"+ dirName + "/" + file + ".bboxes.txt", 'r') as bbFile:
                        data = bbFile.read()
                    import re
                    objs = re.findall('\d+ \d+ \d+ \d+', data)
                    with open(bbOutputPath + "/" + phase + "/" + dirName + "/" + file + ".bboxes.txt", 'w') as bb:
                        bb.write("x y width height\n")
                        for ix, obj in enumerate(objs):
                            # Make pixel indexes 0-based
                            coor = re.findall('\d+', obj)
                            x1 = coor[0]
                            y1 = coor[1]
                            x2 = str(int(coor[0]) + int(coor[2]))
                            y2 = str(int(coor[1]) + int(coor[3]))
                        
                            bb.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")

                        bb.write(dirName)



extractImages(dataSetPath + "/testset.relpaths.txt", "test", nologo=True)
extractImages(dataSetPath + "/trainset.relpaths.txt", "train", nologo=False)
extractImages(dataSetPath + "/valset.relpaths.txt", "val", nologo=False)
