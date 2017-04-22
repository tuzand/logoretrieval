import os
import cv2

datasetpath = '/home/andras/data/datasets/METU/metu/data'

imagelist = ''
i = 0
for subdir, dirs, files in os.walk(os.path.join(datasetpath, 'Images')):
    print 'Processing directory: ' + subdir
    dir = subdir.split('/')[-1]
    for file in files:
        if file.endswith('.jpg'):
            im = cv2.imread(os.path.join(subdir, file))
            ratio = float(im.shape[0])/float(im.shape[1])
            if ratio < 5 and ratio > 0.2:
                file = ('').join(file.split('.')[:-1])
                imagelist += os.path.join(dir, file) + '\n'
            if i % 1000 == 0:
                print i
            i += 1

print 'Writing to file'
with open(os.path.join(datasetpath, 'ImageSets', 'metu_sample.txt'), 'w') as f:
    f.write(imagelist)

