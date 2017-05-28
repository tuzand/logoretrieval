import shutil
import os
import xml.etree.ElementTree
import cv2
from PIL import Image

# Params
#path = '/home/andras/data/datasets/'

ext = '.png'
dstext = '.jpg'
skip_occluded = False
skip_small_rois = False
fuse_occluded = True
outpath = '/home/andras/data/datasets/logorois'


if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath)

for filename in os.listdir(path):
    if not filename.endswith('.xml'):
        continue

    filewithpath = os.path.join(path, filename)
    root = xml.etree.ElementTree.parse(filewithpath).getroot()

    imagename = filename.split('.')[0]

    im = Image.open(os.path.join(path, imagename + ext))
    i = 0
    for obj in root.findall('object'):
        brand = obj.find('name').text.lower()

        if skip_occluded and "teilsichtbar" in brand:
            continue

        brand = brand.replace(".", "")
        brand = brand.replace(" ", "")

        if fuse_occluded:
            brand = brand.replace("-teilsichtbar", "")

        bndbox = obj.find('bndbox')
        x1 = int(bndbox[0].text)
        y1 = int(bndbox[1].text)
        x2 = int(bndbox[2].text)
        y2 = int(bndbox[3].text)

        if (skip_small_rois and x2 - x1 < 26 and y2 - y1 < 26):
            continue

        roi = im.crop((x1, y1, x2, y2))
        folder = os.path.join(outpath, brand)
        if not os.path.exists(folder):
            os.makedirs(folder)
        roi.save(os.path.join(folder, imagename + '_' + str(i) + dstext))

        i += 1
