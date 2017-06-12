caffe_root = './'
import caffe
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
#from py_faster_rcnn.lib.fast_rcnn.train import SolverWrapper

caffe.set_mode_gpu()
caffe.set_device(2)

cfg_from_file('./py_faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml')
cfg.TEST.HAS_RPN = True
cfg.TRAIN.HAS_RPN = True
cfg.TRAIN.IMS_PER_BATCH = 1
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
cfg.TRAIN.RPN_BATCHSIZE = 256
cfg.TRAIN.PROPOSAL_METHOD = 'gt'
cfg.TRAIN.BG_THRESH_LO = 0.0

PROTO = './py_faster_rcnn/models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/train.prototxt'
MODEL = './py_faster_rcnn/data/imagenet/VGG_CNN_M_1024.v2.caffemodel'

#PROTO = './py_faster_rcnn/models/logo/VGG16_219_sharedconv/train.prototxt'
#MODEL = './py_faster_rcnn/data/imagenet/VGG16.v2.caffemodel'

net = caffe.Net(PROTO, MODEL, caffe.TRAIN)

srclayers = ['fc6', 'fc7']
dstlayers = ['fc6_det', 'fc7_det']

# List all layers
#all_names = [n for n in net._layer_names]
#for n in all_names:
#    print n

for i in range(len(srclayers)):
    net.params[dstlayers[i]][0].data[...] = net.params[srclayers[i]][0].data[...]
    net.params[dstlayers[i]][1].data[...] = net.params[srclayers[i]][1].data[...]

#net.save('./py_faster_rcnn/data/imagenet_models/ext81_VGG16.v2.caffemodel')
#net.save('./py_faster_rcnn/data/imagenet_models/ext219_VGG16.v2.caffemodel')


