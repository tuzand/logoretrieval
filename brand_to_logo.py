import os
import shutil

indir = '/home/atuezkoe/datasets/SYN_METU_TA/Annotations/'
outdir = '/home/atuezkoe/datasets/SYN_METU_TA/Annotations_logo/'

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)

i = 0
for filename in os.listdir(indir):
    with open(os.path.join(indir, filename), 'r') as f:
        lines = f.readlines()
    t = ''
    for line in lines:
        line = line.split(' ')
        t += line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' logo\n'
    with open(os.path.join(outdir, filename), 'w') as f:
        f.write(t)
    if i % 1000 == 0:
        print i
    i += 1
