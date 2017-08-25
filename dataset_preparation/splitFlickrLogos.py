from shutil import copy2
import os.path
import cv2


dataSetPath = "/home/pp2015/pp2015_2/data/FL32/FlickrLogos-v2"
outputPath = "/home/pp2015/pp2015_2/data/FL32/FlickrLogos-v2/splitted"
croppedPath = "/home/pp2015/pp2015_2/data/FL32/FlickrLogos-v2/cropped"

def extractImages(fileListPath, cropLogos, phase):
    with open(fileListPath, "r") as fileList:
        lines = fileList.read().splitlines()
        for f in lines:
            splittedPath = f.split('/')
            dirName = splittedPath[-2]
            if False:
                if not os.path.exists(outputPath + "/" + phase + "/" + dirName):
                    os.makedirs(outputPath + "/" + phase + "/" + dirName)
                copy2(dataSetPath + "/" + f, outputPath + "/" + phase + "/" + dirName)
            if cropLogos:
                filename = splittedPath[-1]
                img = cv2.imread(dataSetPath + "/" + f)
                with open(dataSetPath + "/classes/masks/" + dirName + "/" + filename + ".bboxes.txt", "r") as bboxes:
                    bb = bboxes.read().splitlines()[1].split(" ")
                    print f
                    print bb[0]
                    print bb[1]
                    print bb[2]
                    print bb[3]
                    img = img[int(bb[1]) : int(bb[1]) + int(bb[3]), int(bb[0]) : int(bb[0]) + int(bb[2])]
                    if not os.path.exists(outputPath + "/" + phase + "/" + dirName):
                        os.makedirs(croppedPath + "/" + phase + "/" + dirName)
                    cv2.imwrite(croppedPath + "/" + phase + "/" + dirName + "/" + filename, img)


extractImages(dataSetPath + "/testset.relpaths.txt", True, "test")
extractImages(dataSetPath + "/trainset.relpaths.txt", True, "train")
extractImages(dataSetPath + "/valset.relpaths.txt", True, "val")
