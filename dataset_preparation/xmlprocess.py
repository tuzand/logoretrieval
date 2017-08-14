from shutil import copy2
import shutil
import os
import xml.etree.ElementTree
import cv2

path = '/Volumes/WD/datasets/logos/SportBilder/Schalke/raw'
ext = '.png'
dstext = '.jpg'
postfix = '_det'

skip_occluded = True

fuse_occluded = True

outpath = os.path.join(path, '..', 'data' + postfix)

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
for filename in os.listdir(path):
    if not filename.endswith('.xml'):
        continue

    filewithpath = os.path.join(path, filename)
    root = xml.etree.ElementTree.parse(filewithpath).getroot()

    imagename = filename.split('.')[0]

    imglist += imagename + postfix + '\n'

    with open(os.path.join(annotationspath, imagename + postfix + dstext + '.bboxes.txt'), 'w') as annotfile:
        im = cv2.imread(os.path.join(path, imagename + ext))
        i = 0
        for obj in root.findall('object'):
            brand = obj.find('name').text.lower()
            #brand = 'logo'

            if brand == 'anson`s':
                brand = 'ansons'
            elif brand == 'bauhaus-teilsichbar':
                brand = 'bauhaus-teilsichtbar'
            elif brand == "bayern münchen" or brand == "fc bayern münchen":
                brand = 'bayernmuenchen'
            elif brand == 'bayern münchen-teilsichtbar':
                brand = 'bayernmuenchen-teilsichtbar'
            elif brand == 'bet-at-hom2-teilsichtbar' or brand == 'bet-at-home2-teilisichtbar':
                brand = 'bet-at-home2-teilsichtbar'
            elif brand == 'cewe' or brand == 'mein cewe fotobuch':
                brand = 'meincewefotobuch'
            elif brand == 'coca cola':
                brand = 'coca-cola'
            elif brand == 'condor1':
                brand = 'condor'
            elif brand == 'ergo teilsichtbar':
                brand = 'ergo-teilsichtbar'
            elif brand == 'ergo-versichern heißt verstehen' or brand == 'offizieller' \
                or brand == 'thomasberg bennert' or brand == 'thomasberg und bennert' \
                or brand == 'vw-zusatz' or brand == 'vw-zusatz cup' \
                or brand == "wasenberg" or brand == "wasenberg-teilsichtbar":
                continue
            elif brand == 'grazprom-teilsichtbar':
                brand = 'gazprom-teilsichtbar'
            elif brand == 'mein cewe fotobuch-teilsichtbar' or brand == 'mein cewe-fotobuch-teilsichtbar':
                brand = 'meincewefotobuch-teilsichtbar'
            elif brand == 'preisboerse 24':
                brand = 'preisboerse24'
            elif brand == 'stada2':
                brand = 'stada'
            elif brand == 'tillman`s' or brand == 'tilman`s':
                brand = 'tillmans'
            elif brand == 'tillman`s-teilsichtbar':
                brand = 'tillmans-teilsichtbar'
            elif brand == "böklunder":
                brand = 'boeklunder'
            elif brand == "böklunder-teilsichtbar":
                brand = 'boeklunder-teilsichtbar'
            elif brand == "efteling.de":
                brand = "efteling"
            elif brand == "preisboerse24.de-teilsichtbar":
                brand = "preisboerse24-teilsichtbar"
            # 1 gt
            elif brand == "adidas2" or brand == "fcbayernmuenchen" or brand == "bvb" \
                or brand == "deutscher fussballbund" or brand == "nike"  or brand == "money gram" \
                or brand == "zoom":
                continue

            if "teilsichtbar" in brand:
                continue

            if brand == "bet-at-home2":
                brand = "bet-at-home"
            if brand == "bundesliga1":
                brand = "bundesliga"
            if brand == "zdf1":
                brand = "zdf"


            brand = brand.replace(".", "")
            brand = brand.replace(" ", "")

            if brand == "gazprom-footballcom":
                brand = "gazprom"
            if brand == "04schalke-schriftzug":
                brand = "04schalke"

            if fuse_occluded:
                brand = brand.replace("-teilsichtbar", "")




            bndbox = obj.find('bndbox')
            x1 = int(bndbox[0].text)
            y1 = int(bndbox[1].text)
            x2 = int(bndbox[2].text)
            y2 = int(bndbox[3].text)

            if (x2 - x1 < 26 and y2 - y1 < 26):
                continue
            brand = 'logo'
            brandlist.append(brand)
            roi = im[y1:y2, x1:x2]
            folder = os.path.join(brandspath, brand)
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(os.path.join(folder, imagename + '_' + str(i) + '.jpg'), roi)

            annotfile.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + brand + '\n')
            i += 1

    copy2(os.path.join(path, imagename + ext), os.path.join(imagespath, imagename + postfix + dstext))

with open(os.path.join(imagesetspath, 'schalke' + postfix + '.txt'), 'w') as f:
    f.write(imglist)

with open(os.path.join(outpath, '..', 'brands.txt'), 'w') as f:
    for brand in set(brandlist):
        f.write(brand + '\n')
