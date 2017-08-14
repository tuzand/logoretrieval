import os
import shutil
from enum import Enum

def createdirs(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

#path = "/Users/andrastuzko/Pictures/datasets/SportTestbilder/"
path = '/Volumes/WD/datasets/logos/SportBilder'

dataset = "srf_ice"
outpaths = ["good", "good_occlusion", "bad", "bad_occlusion"]
vggfile = os.path.join(path, dataset + ".csv")

#dataset = "srf_ski"
#outpaths = ["good"]
#vggfile = os.path.join(path, dataset + ".csv")

#dataset = "srf_football"
#outpaths = ["good"]
#vggfile = os.path.join(path, dataset + ".csv")
postfix = '_det'

annotationspaths = list()
imagesetspaths = list()
for p in outpaths:
    annotationspaths.append(os.path.join(path, dataset, p, "data", "Annotations"))
    imagesetspaths.append(os.path.join(path, dataset, p, "data", "ImageSets"))

for i in range(len(outpaths)):
    createdirs(annotationspaths[i])
    createdirs(imagesetspaths[i])

with open(vggfile, "r") as vgg:
    lines = vgg.readlines()

def getdistortion(part):
    prop = part.split("=")[0]
    if prop == "occluded":
        return 1
    elif prop == "bad":
        return 2

annotationsdicts = [dict() for x in range(len(outpaths))]

brands = list()

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

    #brands.append(cls)
    brands.append('logo')

    #l = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + cls
    l = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + 'logo'

    state = -1
    if len(tokenized) > 5:
        print(tokenized)
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
        imagelist.append(key.split('.')[0])
        with open(os.path.join(annotationspaths[idx], key.split('.')[0] + postfix + ".jpg.bboxes.txt"), "w") as annot:
            for value in values:
                annot.write(value + "\n")

    with open(os.path.join(imagesetspaths[idx], dataset + "_" + imagesetname + ".txt"), "w") as f:
        for imagepath in set(imagelist):
            f.write(imagepath + postfix + "\n")

with open(os.path.join(path, dataset, 'brands.txt'), 'w') as bf:
    brands = sorted(set(brands))
    for brand in brands:
        bf.write(brand + '\n')
