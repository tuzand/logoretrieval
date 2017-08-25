import shutil
import os
from PIL import Image
from sets import Set
import random, string
import cv2


# Params
#path = '/home/andras/data/datasets/'

ext = '.jpg'
dstext = '.jpg'
skip_occluded = False
skip_small_rois = False
fuse_occluded = True
workingpath = '/home/atuezkoe/data/datasets'
outdataset = 'public_litw_wo_fl32'
datasets = ['BL/BL/data', 'FL27/FL27/data', 'toplogo/toplogo/data', 'logodata/data']



outpath = os.path.join(workingpath, outdataset + '_logorois')
if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath)


brands = list()
for dataset in datasets:
    with open(os.path.join(workingpath, dataset, 'brands.txt'), 'r') as f:
        brandlist = f.read().splitlines()
    for b in brandlist:
        if b == "":
            continue
        brands.append(b)

classes = list(Set(brands))

num_classes = len(classes)
print "Number of classes: " + str(num_classes)

class_to_ind = dict(zip(classes, xrange(num_classes)))
#for key, value in class_to_ind.iteritems():
#    print str(key) + " " + str(value)

i = 0

def extract(path):
    global trainlabels
    global vallabels
    global allbrands
    global i
    annotpath = os.path.join(path, 'Annotations')
    for root, subdirs, files in os.walk(annotpath):
        for filename in files:
            filewithpath = os.path.join(root, filename)
            with open(filewithpath, 'r') as f:
                lines = f.read().splitlines()
            imagename = filename.split('.')[0]

            rel = os.path.relpath(root, annotpath)
            im = cv2.imread(os.path.join(path, 'Images', rel, imagename + ext))
            for line in lines:
                line = line.split()
                brand = line[-1].lower()

                brand = brand.replace(".", "")
                brand = brand.replace(" ", "")

                x1 = int(line[0])
                y1 = int(line[1])
                x2 = int(line[2])
                y2 = int(line[3])

                #roi = im.crop((x1, y1, x2, y2))
                roi = im[y1:y2, x1:x2]
                trainfolder = os.path.join(outpath, 'train', brand)
                valfolder = os.path.join(outpath, 'val', brand)
                name = ''.join(random.choice(string.lowercase) for i in range(15)) + dstext
                if i%10 == 0 and brand in allbrands:
                    if not os.path.exists(valfolder):
                        os.makedirs(valfolder)
                    cv2.imwrite(os.path.join(valfolder, name), roi)
                    vallabels += brand + "/" + name + " " + str(class_to_ind[brand]) + "\n"
                else:
                    if not os.path.exists(trainfolder):
                        os.makedirs(trainfolder)
                    cv2.imwrite(os.path.join(trainfolder, name), roi)
                    trainlabels += brand + "/" + name + " " + str(class_to_ind[brand]) + "\n"
                allbrands.append(brand)
                allbrands = list(Set(allbrands))
                i += 1


trainlabels = ""
vallabels = ""
allbrands = []
for dataset in datasets:
    extract(os.path.join(workingpath, dataset))
print i

with open(os.path.join(outpath, 'train', 'labels.txt'), 'w') as f:
    f.write(trainlabels)

with open(os.path.join(outpath, 'val', 'labels.txt'), 'w') as f:
    f.write(vallabels)
