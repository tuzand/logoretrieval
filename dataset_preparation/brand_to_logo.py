import os
import shutil
from shutil import copyfile


ext = '.png'
indir = '/home/andras/data/datasets/srf_ski/good/data/'
outdir = '/home/andras/data/datasets/srf_ski_logo/good/data/'

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(os.path.join(outdir, 'Annotations'))
os.makedirs(os.path.join(outdir, 'Images'))
os.makedirs(os.path.join(outdir, 'ImageSets'))

i = 0
filelist = ""
for filename in os.listdir(os.path.join(indir, 'Annotations')):
    with open(os.path.join(indir, 'Annotations', filename), 'r') as f:
        lines = f.readlines()
    t = ''
    for line in lines:
        line = line.split(' ')
        t += line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' logo\n'
    im = filename.split('.')[0]
    filelist += im + '_det\n'
    with open(os.path.join(outdir, 'Annotations', im + '_det.jpg.bboxes.txt'), 'w') as f:
        f.write(t)
    copyfile(os.path.join(indir, 'Images', im + ext), os.path.join(outdir, 'Images', im + '_det.jpg'))

with open(os.path.join(outdir, 'ImageSets', 'srf_ski_logo_good.txt'), 'w') as f:
    f.write(filelist)

