caffe_root = './'
import caffe
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
#from py_faster_rcnn.lib.fast_rcnn.train import SolverWrapper

caffe.set_mode_gpu()
caffe.set_device(2)

#sw = SolverWrapper(solver_prototxt, roidb, roidb_det, output_dir,
#                       pretrained_model=pretrained_model)

cfg_from_file('./py_faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml')
net = caffe.Net('./py_faster_rcnn/models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/train.prototxt', 
                './py_faster_rcnn/data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel', 
                caffe.TRAIN)

srclayers = ['fc6', 'fc7']
dstlayers = ['fc6_det', 'fc7_det']

for i in range(len(srclayers)):
    net.params[dstlayers[i]][0] = net.params[srclayers[i]][0]
    net.params[dstlayers[i]][1] = net.params[srclayers[i]][1]

net.save('./py_faster_rcnn/data/imagenet_models/ext_VGG_CNN_M_1024.v2.caffemodel')

