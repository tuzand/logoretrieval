from shutil import copy2
import os.path


dataSetPath = "/home/atuezkoe/masterarbeit/FlickrLogos-32/FlickrLogos-v2"
outputPath = "/home/atuezkoe/masterarbeit/FlickrLogos-32/FlickrLogos-v2/splitted"

def extractImages(fileListPath, phase):
    with open(fileListPath, "r") as fileList:
        lines = fileList.read().splitlines()
        for f in lines:
            splittedPath = f.split('/')
            dirName = splittedPath[-2]
            if not os.path.exists(outputPath + "/" + phase + "/" + dirName):
                os.makedirs(outputPath + "/" + phase + "/" + dirName)
            copy2(dataSetPath + "/" + f, outputPath + "/" + phase + "/" + dirName)


extractImages(dataSetPath + "/testset.relpaths.txt", "test")
extractImages(dataSetPath + "/trainset.relpaths.txt", "train")
extractImages(dataSetPath + "/valset.relpaths.txt", "val")
