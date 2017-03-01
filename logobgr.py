import os
import cv2
import colorsys
import numpy as np
from math import sqrt
import random

imagepath = '/Users/andrastuzko/Pictures/datasets/new_queryset_allinone'


for f in os.listdir(imagepath):

    if f.endswith('.jpg'):
        print f


        im = cv2.imread(os.path.join(imagepath, f))
        width = im.shape[1]
        height = im.shape[0]


        vertical = random.randint(int(width/2), int(1.5 * width))
        horizontal = random.randint(int(-height/2), int(height/2))
        shear = random.randint(int(-width/4), int(width/4))
        perspective = float(random.randint(7, 10)) / 10




        '''notwhiteim = im[(im < 250).all(axis=2)]

        meanr = np.mean(notwhiteim[:, 2])
        meang = np.mean(notwhiteim[:, 1])
        meanb = np.mean(notwhiteim[:, 0])

        hsv = colorsys.rgb_to_hsv(meanr, meang, meanb)

        hsv = list(hsv)



        hsv[0] += 1.5

        hsv[2] += (255 - hsv[2]) * 0.5

        rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])

        im2 = im
        whiteim = (im > 200).all(axis=2)
        CV_BGR2BGRA
        im2[whiteim] = rgb'''
        im2 = im




        #inp = np.float32([[0,0], [width+200,+150], [width+200, height+50], [0, height-1]])
        #out = np.float32([[0,50], [width-1, 50], [width-1, height-1], [0,height-1]])

        inp = np.float32([[0,0], [width-1,0], [width-1, height-1], [0, height-1]])
        if horizontal < 0:
            lefthorizontal = -horizontal
            righthorizontal = 0
        else:
            lefthorizontal = 0
            righthorizontal = horizontal

        if shear < 0:
            bottomshear = -shear
            topshear = 0
        else:
            bottomshear = 0
            topshear = shear
        out = np.float32([[topshear,lefthorizontal], [(vertical + topshear), righthorizontal], [(vertical+bottomshear), (height-1 + righthorizontal)], [bottomshear,(height-1+lefthorizontal)]])



        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(inp, out)
        print 'size'
        print vertical
        print max(height-1 + righthorizontal, height-1+lefthorizontal)
        warp = cv2.warpPerspective(im2, M, (max(vertical + topshear, vertical+bottomshear), max(height-1 + righthorizontal, height-1+lefthorizontal)))

        cv2.imwrite(os.path.join(imagepath, 'results', f), warp)
