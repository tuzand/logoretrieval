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
outpath = '/home/andras/data/datasets/allnet_logorois'

if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath)


flickrlogo32 = ( 'adidas', 'apple', 'bmw', 'chimay', 'corona', 'erdinger', # FlickrLogos-32
                 'fedex', 'ford', 'google', 'heineken', 'milka', 'paulaner',
                 'rittersport', 'singha', 'stellaartois', 'tsingtao', 'aldi',
                 'becks', 'carlsberg', 'cocacola', 'dhl', 'esso', 'ferrari',
                 'fosters', 'guiness', 'hp', 'nvidia', 'pepsi', 'shell',
                 'starbucks', 'texaco', 'ups')

bl = (           'adidas', 'adidas-text', 'airness', 'base', 'bfgoodrich', 'bik', # BelgaLogos
                 'bouigues', 'bridgestone', 'bridgestone-text', 'carglass',
                 'citroen', 'citroen-text', 'cocacola', 'cofidis', 'dexia',
                 'eleclerc', 'ferrari', 'gucci', 'kia', 'mercedes', 'nike',
                 'peugeot', 'puma', 'puma-text', 'quick', 'reebok', 'roche',
                 'shell', 'sncf', 'standard_liege', 'stellaartois', 'tnt', 'total',
                 'us_president', 'umbro', 'veolia', 'vrt')

toplogo = (      'adidas', 'chanel', 'gucci', 'hh', 'lacoste', # TopLogo-10
                 'mk', 'nike', 'prada', 'puma', 'supreme')

fl27 = (         'adidas', 'apple', 'bmw', 'citroen', 'cocacola', 'dhl', # FlickrLogos-27
                 'fedex', 'ferrari', 'ford', 'google', 'heineken', 'hp',
                 'intel', 'mcdonalds', 'mini', 'nbc', 'nike', 'pepsi',
                 'porsche', 'puma', 'redbull', 'sprite', 'starbucks',
                 'texaco', 'unicef', 'vodafone', 'yahoo'
                 )

srf = list()
srfvids = ['srf_football', 'srf_ice', 'srf_ski']
for vid in srfvids:
    with open('/home/andras/data/datasets/' + vid + '/brands.txt', 'r') as f:
        brandlist = f.read().splitlines()

    for b in brandlist:
        if b == "":
            continue
        srf.append(b)

other_datasets = list()
other_datasets.extend(bl)
other_datasets.extend(toplogo)
other_datasets.extend(fl27)
other_datasets.extend(srf)
other_datasets = Set(other_datasets)
classes = list()
classes.append('__background__') # always index 0
classes.extend(flickrlogo32)
otherminusflickr = [brand for brand in other_datasets if brand not in flickrlogo32]
classes.extend(otherminusflickr)

num_classes = len(classes)

class_to_ind = dict(zip(classes, xrange(num_classes)))
#for key, value in class_to_ind.iteritems():
#    print str(key) + " " + str(value)

def extract(path):
    global trainlabels
    global vallabels
    annotpath = os.path.join(path, 'Annotations')
    i = 0
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
                if i%10 == 0:
                    if not os.path.exists(valfolder):
                        os.makedirs(valfolder)
                    cv2.imwrite(os.path.join(valfolder, name), roi)
                    vallabels += brand + "/" + name + " " + str(class_to_ind[brand]) + "\n"
                else:
                    if not os.path.exists(trainfolder):
                        os.makedirs(trainfolder)
                    cv2.imwrite(os.path.join(trainfolder, name), roi)
                    trainlabels += brand + "/" + name + " " + str(class_to_ind[brand]) + "\n"
                i += 1
    print i




paths = ["/home/andras/data/datasets/FL32/FlickrLogos-v2/fl/fl/data", "/home/andras/data/datasets/FL27/FL27/data", \
    "/home/andras/data/datasets/BL/BL/data", "/home/andras/data/datasets/L32P/classification/data", \
    "/home/andras/data/datasets/toplogo/toplogo/data", "/home/andras/data/datasets/srf_ice/good/data", \
    "/home/andras/data/datasets/srf_ice/good_occlusion/data", "/home/andras/data/datasets/srf_ice/bad/data", \
    "/home/andras/data/datasets/srf_ice/bad_occlusion/data", "/home/andras/data/datasets/srf_football/data", \
    "/home/andras/data/datasets/srf_ski/good/data"]

paths = ["/home/andras/data/datasets/FL32/FlickrLogos-v2/fl/fl/data", "/home/andras/data/datasets/FL27/FL27/data", \
    "/home/andras/data/datasets/BL/BL/data", "/home/andras/data/datasets/L32P/classification/data", \
    "/home/andras/data/datasets/toplogo/toplogo/data"]

trainlabels = ""
vallabels = ""
for p in paths:
    extract(p)

with open(os.path.join(outpath, 'train', 'labels.txt'), 'w') as f:
    f.write(trainlabels)

with open(os.path.join(outpath, 'val', 'labels.txt'), 'w') as f:
    f.write(vallabels)
