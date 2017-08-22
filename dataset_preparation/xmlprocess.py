from shutil import copy2
import shutil
import os
import xml.etree.ElementTree
import cv2

path = '/home/andras/data/datasets/LogoDataset_wo_fl32'
ext = '.jpg'
dstext = '.jpg'
postfix = '_det'

skip_occluded = True

fuse_occluded = True

outpath = os.path.join(path, '..', 'logodata_det')

annotationspath = os.path.join(outpath, 'Annotations')
imagespath = os.path.join(outpath, 'Images')
imagesetspath = os.path.join(outpath, 'ImageSets')
brandspath = os.path.join(outpath, 'brands')

if os.path.exists(outpath):
    shutil.rmtree(outpath)

os.makedirs(annotationspath)
os.makedirs(imagespath)
os.makedirs(imagesetspath)

imglist = ''
brandlist = list()
subdirID = 0
dirs = None
for r, subdirs, files in os.walk(path):
    if dirs == None:
        dirs = subdirs
    #parent = dirs[subdirID]
    subdirID += 1
    for filename in files:
        if not filename.endswith('.xml'):
            continue

        filewithpath = os.path.join(r, filename)
        parent = filewithpath.split('/')[-2]
        root = xml.etree.ElementTree.parse(filewithpath).getroot()

        imagename = filename.split('.')[0]

        imglist += parent + imagename + postfix + '\n'

        with open(os.path.join(annotationspath, parent + imagename + postfix + dstext + '.bboxes.txt'), 'w') as annotfile:
            #im = cv2.imread(os.path.join(path, imagename + ext))
            i = 0
            for obj in root.findall('object'):
                brand = obj.find('name').text.lower()

                bndbox = obj.find('bndbox')
                x1 = int(bndbox[0].text)
                y1 = int(bndbox[1].text)
                x2 = int(bndbox[2].text)
                y2 = int(bndbox[3].text)

                brand = 'logo'
                brandlist.append(brand)
                #roi = im[y1:y2, x1:x2]
                #folder = os.path.join(brandspath, brand)
                #if not os.path.exists(folder):
                #    os.makedirs(folder)
                #cv2.imwrite(os.path.join(folder, imagename + '_' + str(i) + '.jpg'), roi)

                annotfile.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + brand + '\n')
                i += 1

        copy2(os.path.join(r, imagename + ext), os.path.join(imagespath, parent + imagename + postfix + dstext))

with open(os.path.join(imagesetspath, 'ownlogos' + postfix + '.txt'), 'w') as f:
    f.write(imglist)

with open(os.path.join(outpath, '..', 'brands.txt'), 'w') as f:
    for brand in set(brandlist):
        f.write(brand + '\n')
