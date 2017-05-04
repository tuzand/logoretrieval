from shutil import copy2
import shutil
import os
import xml.etree.ElementTree

path = '/Volumes/WD/datasets/logos/SportBilder/Schalke/raw'
ext = '.png'
dstext = '.jpg'
postfix = '_det'

outpath = os.path.join(path, '..', 'data')

annotationspath = os.path.join(outpath, 'Annotations')
imagespath = os.path.join(outpath, 'Images')
imagesetspath = os.path.join(outpath, 'ImageSets')

if os.path.exists(outpath):
    shutil.rmtree(outpath)

os.makedirs(annotationspath)
os.makedirs(imagespath)
os.makedirs(imagesetspath)

imglist = ''
for filename in os.listdir(path):
    if not filename.endswith('.xml'):
        continue

    filewithpath = os.path.join(path, filename)
    root = xml.etree.ElementTree.parse(filewithpath).getroot()

    imagename = filename.split('.')[0]

    imglist += imagename + postfix + '\n'

    with open(os.path.join(annotationspath, imagename + postfix + dstext + '.bboxes.txt'), 'w') as annotfile:
        for obj in root.findall('object'):
            brand = obj.find('name').text.lower()

            brand = 'logo'

            bndbox = obj.find('bndbox')
            x1 = bndbox[0].text
            y1 = bndbox[1].text
            x2 = bndbox[2].text
            y2 = bndbox[3].text

            annotfile.write(x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + brand + '\n')

    copy2(os.path.join(path, imagename + ext), os.path.join(imagespath, imagename + postfix + dstext))

with open(os.path.join(imagesetspath, 'schalke.txt'), 'w') as f:
    f.write(imglist)
