import os
import cv2
import colorsys
import numpy as np

im = cv2.imread('/home/atuezkoe/datasets/1-2.jpg')
whiteim = (im > 220).all(axis=2)
notwhiteim = im[(im < 220).all(axis=2)]

meanr = np.mean(notwhiteim[:, 2])
meang = np.mean(notwhiteim[:, 1])
meanb = np.mean(notwhiteim[:, 0])

hsv = colorsys.rgb_to_hsv(meanr, meang, meanb)

hsv = list(hsv)

hsv[0] += 0.5

rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])

im2 = im
im2[whiteim] = rgb

cv2.imwrite('/home/atuezkoe/datasets/logo.jpg', im2)


