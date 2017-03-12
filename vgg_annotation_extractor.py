import os
import shutil
from enum import Enum

def createdirs(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

path = "/Users/andrastuzko/Pictures/datasets/SportTestbilder/"

vggfile = os.path.join(path, "eis_2nd.csv")
annotationspath = os.path.join(path, "data", "Annotations")
imagesetspath = os.path.join(path, "data", "ImageSets")
outpaths = ["good", "good_occlusion", "bad", "bad_occlusion"]
for p in outpaths:
    createdirs(os.path.join(path, "data", "Annotations", p))
createdirs(imagesetspath)

with open(vggfile, "r") as vgg:
    lines = vgg.readlines()

def getdistortion(part):
    prop = part.split("=")[0]
    if prop == "occluded":
        return 1
    elif prop == "bad":
        return 2

annotationsdicts = [dict() for x in range(4)]

for (idx, line) in list(enumerate(lines))[1:]:
    tokenized = line.split(';')
    filename = tokenized[0].split(',')[0]
    x1 = tokenized[1].split("=")[1]
    y1 = tokenized[2].split("=")[1]
    w = tokenized[3].split("=")[1]
    h_and_class = tokenized[4].split(",")
    h = h_and_class[0].split("=")[1].split("\"")[0]

    x2 = int(x1) + int(w)
    y2 = int(y1) + int(h)

    cls = h_and_class[1].split("=")[0].split("\"")[1]

    l = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + cls

    state = -1
    if len(tokenized) > 5:
        state = getdistortion(tokenized[5])
        if len(tokenized) > 6:
            state += getdistortion(tokenized[6])
    else:
        state = 0

    annotations = annotationsdicts[state]
    if not filename in annotations:
        annotations[filename] = list()
    annotations[filename].append(l)

for idx, annotations in enumerate(annotationsdicts):
    imagesetname = outpaths[idx]
    imagelist = list()
    for key, values in annotations.items():
        imagelist.append(os.path.join(imagesetname, key.split('.')[0]))
        with open(os.path.join(annotationspath, imagesetname, key + ".bboxes.txt"), "w") as annot:
            for value in values:
                annot.write(value + "\n")

    with open(os.path.join(imagesetspath, "srf_icehockey_" + imagesetname + ".txt"), "w") as f:
        for imagepath in set(imagelist):
            f.write(imagepath + "\n")
