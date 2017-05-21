import os

blpath = '/home/andras/data/datasets/BL/BL/data/ImageSets'
blinput = 'bl_train_all.txt'
bltrain = 'bl_train.txt'
bltest = 'bl_test.txt'
blgt = '/home/andras/data/datasets/BL/qset3_internal_and_local.qry'

flblpath = '/home/andras/data/datasets/FLBL/FLBL/data/ImageSets'
flblinput = 'flbl_train_all.txt'
flbltrain = 'flbl_train.txt'
flbltest = 'flbl_test.txt'

flblgt = '/home/andras/data/datasets/FLBL/groundtruth.txt'


traintestratio = 5

with open(blgt, 'r') as all:
    bllines_all = all.readlines()

with open(flblgt, 'r') as all:
    flbllines_all = all.readlines()

with open(os.path.join(blpath, blinput), 'r') as all:
    lines = all.readlines()
with open(os.path.join(blpath, bltrain), 'w') as bltrain:
    with open(os.path.join(blpath, bltest), 'w') as bltest:
        with open(os.path.join(flblpath, flbltrain), 'w') as flbltrain:
            with open(os.path.join(flblpath, flbltest), 'w') as flbltest:
                for idx, line in enumerate(lines):
                    if idx % traintestratio == 0:
                        bltest.write(line)
                    else:
                        bltrain.write(line)
                    bllines = [s for s in bllines_all if line[:-1] in s]
                    '''for blline in bllines:
                        blimg = blline.split()[0]
                        for flblline in flbllines_all:
                            print flblline
                            print blimg
                            dd
                        flblline = [s for s in flbllines_all if blimg in s]
                        print flblline
                        dd
                        flblimg = flblline.split()[2].split('.')[0]
                        if idx % traintestratio == 0:
                            flbltest.write(flblimg + '\n')
                        else:
                            flbltrain.write(flblimg + '\n')'''
