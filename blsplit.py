import os

blpath = '/home/andras/data/datasets/BL/BL/data/ImageSets'
blinput = 'bl_train_all.txt'
bltrain = 'bl_train.txt'
bltest = 'bl_test.txt'

flblpath = '/home/andras/data/datasets/FLBL/FLBL/data/ImageSets'
flblinput = 'flbl_train_all.txt'
flbltrain = 'flbl_train.txt'
flbltest = 'flbl_test.txt'


traintestratio = 5

def split(path, input, train, test):
    with open(os.path.join(path, input), 'r') as all:
        lines = all.readlines()

    with open(os.path.join(path, train), 'w') as train:
        with open(os.path.join(path, test), 'w') as test:
            for idx, line in enumerate(lines):
                if idx % traintestratio == 0:
                    test.write(line)
                else:
                    train.write(line)


split(blpath, blinput, bltrain, bltest)

split(flblpath, flblinput, flbltrain, flbltest)

