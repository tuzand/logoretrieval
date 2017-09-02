#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_det import im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
from utils.timer import Timer
import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt
from custom_imdb import get_custom_imdb
from fast_rcnn.nms_wrapper import nms
import cPickle
import sys

##### Parameters #####

vis = True
rpndet = False
threshold = 0.75

FRCNN = 'py_faster_rcnn'

PROTO = os.path.join(FRCNN, 'models/logo_detection/VGG16/test.prototxt')
# Master Thesis Best Detector
#MODEL = os.path.join(FRCNN, 'output/final/allnet_srf_det_cl_reducedlr/vgg16_faster_rcnn_detection_iter_4000.caffemodel')

# Paper Best Detector
MODEL = os.path.join(FRCNN, \
'output/final/publicNonFlickr_ownlogo_detection_vgg16/vgg16_faster_rcnn_detection_iter_30000.caffemodel')

SEARCH = 'fl_detection_test_logo'

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_simple/test.prototxt')
#MODEL = os.path.join(FRCNN, 'output/final/allnet_vgg_cnn_m_single_all_for_fl_test/vgg_cnn_m_1024_faster_rcnn_allnet_simple_iter_120000.caffemodel')

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/test.prototxt')
#MODEL = os.path.join(FRCNN, 'output/final/fl_vgg_cnn_m/vgg_cnn_m_1024_faster_rcnn_simple_fl_iter_80000.caffemodel')

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/test.prototxt')


#PROTO = os.path.join(FRCNN, 'models/logo/VGG16_219/test.prototxt')
#MODEL = os.path.join(FRCNN, 'output/final/alllogo_simple_vgg16/vgg16_faster_rcnn_alllogo_iter_120000.caffemodel')

#PROTO = os.path.join(FRCNN, 'models/logo/VGG16_219_sharedconv/test.prototxt')
#MODEL = os.path.join(FRCNN, 'output/final/alllogo_vgg16_sharedconv/vgg16_faster_rcnn_alllogo_sharedconv_iter_30000.caffemodel')

#SEARCH = 'schalke_det'
SEARCH = 'fl_detection_test'
# Custom dataset: only for bounding box printing
customdataset = False

#####################

def draw_boxes(img, dets, imagename):
    if dets.shape[0] == 0:
        return
    im = img.copy()
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),6)
    cv2.imwrite('/home/atuezkoe/github/LogoRetrieval/resultimages/' + imagename.split('.')[0] + '.jpg', im)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='./py_faster_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def search(net, score_list, box_list):
    if customdataset:
        imdb = get_custom_imdb(SEARCH)
    else:
        imdb = get_imdb(SEARCH)

    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    print threshold

    i = 0
    for i in xrange(num_images):
        imagepath = imdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        _t['im_detect'].tic()
        if len(score_list) < i+1:
            scores, boxes = im_detect(net, im, rpndet, None)
            score_list.append(np.copy(scores))
            box_list.append(np.copy(boxes))
        else:
            scores = np.copy(score_list[i])
            boxes = np.copy(box_list[i])
        _t['im_detect'].toc()

        _t['misc'].tic()

        for j in xrange(1, imdb.num_classes):
            if rpndet:
                k = j-1
            else:
                k = j
            inds = np.where(scores[:, k] > threshold)[0]
            cls_scores = scores[inds, k]
            cls_boxes = boxes[inds, k*4:(k+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                draw_boxes(im, cls_dets, imagepath.split('/')[-1].split('.')[0])
            all_boxes[j][i] = cls_dets

        _t['misc'].toc()


        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
        i += 1

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if not customdataset:
        print 'Evaluating detections'
        rec, prec, map, tp, fp = imdb.evaluate_detections(all_boxes, output_dir)
        return rec, prec, map, tp, fp, num_images


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TEST.HAS_RPN = True
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BG_THRESH_LO = 0.0

    cfg.TEST.BBOX_REG = False
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(PROTO, MODEL, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(MODEL))[0]

    score_list = list()
    box_list = list()
    if rpndet:
        dettext = 'rpndet'
    else:
        dettext = 'det'
    if customdataset:
        search(net, score_list, box_list)
    else:
        with open('vgg_pub_' + dettext + '_' + SEARCH + '_results.txt', 'w') as f:
            for t in np.arange(0.99, 0.009, -0.01):
                threshold = t
                rec, prec, map, tp, fp, num_images = search(net, score_list, box_list)
                f.write(str(threshold) + '\t' + str(fp/float(num_images)) + '\t' + str(rec[-1]) + '\t' + str(map) + '\n')
        
