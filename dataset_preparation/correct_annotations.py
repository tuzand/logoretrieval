# -*- coding: iso-8859-1 -*-
from shutil import copy2
import shutil
import os
import xml.etree.ElementTree
import cv2

path = '/home/atuezkoe/data/datasets/LogoDataset_wo_fl32'
ext = '.jpg'
dstext = '.jpg'
postfix = ''

fuse_occluded = True

imglist = ''
brandlist = list()
for r, subdirs, files in os.walk(path):
    for filename in files:
        if not filename.endswith('.xml'):
            continue

        filewithpath = os.path.join(r, filename)
        parent = filewithpath.split('/')[-2]
        parser = xml.etree.ElementTree.XMLParser(encoding="utf-8")
        tree = xml.etree.ElementTree.parse(filewithpath, parser = parser)
        root = tree.getroot()

        imagename = filename.split('.')[0]

        imglist += parent + imagename + postfix + '\n'

        for obj in root.findall('object'):
            brand = obj.find('name').text.encode('utf-8').lower()
            if brand == "adidas-symbol":
                brand = "adidas"
            if brand == "aluratek":
                brand = "aluratek-symbol"
            if brand == "apecase":
                brand = "apecase-symbol"
            if brand == "apecase-teilsichtbar":
                brand = "apecase-symbol-teilsichtbar"
            if brand == "armitron1":
                brand = "armitron"
            if brand == "audi":
                brand = "audi-symbol"
            if brand == "b":
                brand = "bridgestone"
            if brand == "budweiser1" or brand == "budweiser2":
                brand = "budweiser"
            if brand == "budweiser-b":
                brand = "budweiser-b-symbol"
            if brand == "budweiser-teilsichtbar":
                brand = "budweiser-symbol-teilsichtbar"
            if brand == "bundweiser" in brand:
                brand = brand.replace("bundweiser", "budweiser")
            if brand == "burgerking":
                brand = "burgerking-symbol"
            if brand == "burgerking-teilsichtbar":
                brand = "burgerking-symbol-teilsichtbar"
            if brand == "canon1":
                brand = "canon"
            if "caterpillar1" in brand:
                brand = brand.replace("caterpillar1", "caterpillar")
            if brand == "citroen":
                brand == "citroen-symbol"
            if brand == "colgate1":
                brand = "colgate"
            if "dadone" in brand:
                brand = brand.replace("dadone", "danone")
            if brand == "danone1":
                brand = "danone"
            if brand == "google":
                brand = "google-symbol"
            if brand == "gucci1":
                brand = "gucci"
            if brand == "gucci logo":
                brand = "gucci-symbol"
            if "heineke" in brand:
                brand = brand.replace("heineke", "heineken")
            if brand == "hersheys1":
                brand = "hersheys"
            if brand == "hungry jacks logo":
                brand = "hungry jacks-symbol"
            if "hyundri" in brand:
                brand = brand.replace("hyundri", "hyundai")
            if "kellogg`s-k" in brand:
                brand = brand.replace("kellogg`s-k", "kellogg`s-symbol")
            if "kia-logo" in brand:
                brand = brand.replace("kia-logo", "kia")
            if brand == "lego":
                brand = "lego-symbol"
            if brand == "lego-teilsichtbar":
                brand = "lego-symbol-teilsichtbar"
            if brand == "mastercard1":
                brand = "mastercard"
            if brand == "mcdonalds":
                brand = "mcdonalds-symbol"
            if brand == "mcdonalds-teilsichtbar":
                brand = "mcdonalds-symbol-teilsichtbar"
            if brand == "mercedes" or brand == "mercedes-logo":
                brand = "mercedesbenz-symbol"
            if brand == "mercedes-schrift" or brand == "mercedes-schriftzug":
                brand = "mercedesbenz"
            if brand == "mercedes-teilsichtbar":
                brand = "mercedesbenz-symbol-teilsichtbar"
            if brand == "nestle1" or brand == "nestle2":
                brand = "nestle"
            if brand == "nike":
                brand = "nike-symbol"
            if "nikelogo" in brand:
                brand = brand.replace("nikelogo", "nike")
            if brand == "lego1":
                brand = "lego"
            if brand == "nivea1":
                brand = "nivea"
            if brand == "pizzahut-logo":
                brand = "pizzahut"
            if "ruffels" in brand:
                brand = brand.replace("ruffels", "ruffles")
            if brand == "the home depot1" or brand == "the home depot-logo":
                brand = "thehomedepot"
            if "vl" in brand:
                brand = brand.replace("vl", "louisvuitton")
            if brand == "volksbank":
                brand = "volksbank-symbol"
            if brand == "ströker":
                brand = "stroeker"
            if brand == "görtz":
                brand = "goertz"
            if "schriftzug" in brand:
                brand = brand.replace("-schriftzug", "")
            if "schrift" in brand:
                brand = brand.replace("-schrift", "")
            if "teilsichtbar" in brand:
                brand = brand.replace(u"-teilsichtbar", "")
                obj.find('truncated').text = str(1)
            if "logo" in brand:
                brand = brand.replace('logo', 'symbol')
            if "`" in brand:
                brand = brand.replace("`", "")
            brand = brand.replace(" ", "")
            obj.find('name').text = brand
        tree.write(filewithpath, encoding="UTF-8")

