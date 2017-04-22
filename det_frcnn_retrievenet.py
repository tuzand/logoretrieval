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

max_per_image = 0
vis = False
visualize_logo_detection = False
thresh = 0.2
RESULTPATH = './results/'
RESULTPOSTFIX = '.result2.txt'

FRCNN = 'py_faster_rcnn'

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/test.prototxt')
PROTO = os.path.join(FRCNN, 'models/logo_detection/VGG_CNN_M_1024/faster_rcnn_end2end/simple/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/sharedconv/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/sharedconv_srf_ice/test.prototxt')
PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/simple/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/faster_rcnn_alt_opt_simple/faster_rcnn_test.pt')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_ignorelabel_iter_80000.caffemodel')
MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/synmetu_ta_detection/vgg_cnn_m_1024_faster_rcnn_synmetuta_detection_iter_600000.caffemodel')
MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_logos32plus_sharedconv/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/fl_train+fl_val_logo/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/synmetu_ta_detection/vgg_cnn_m_1024_faster_rcnn_synmetuta_detection_iter_50000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/srf_ice_sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_sharedconv_srf_ski_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/default/train/fl_faster_rcnn_final.caffemodel')
EXAMPLEPATH = '/home/andras/logoexamples'
#SEARCHPATH = '/home/andras/data/datasets/fussi'

def write_bboxes(im, imagename, bboxArray, scoreArray, classArray):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(bboxArray)):
        bbox = bboxArray[i][4:8]
        score = scoreArray[i]
        class_name = classArray[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(str(class_name), score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #ax.set_title(('Detections with '
    #              'p(obj | box) >= {:.1f}').format(logo_threshold),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('/home/andras/github/logoretrieval/resultimages/' + imagename.split('.')[0] + '.jpg')
    plt.close()

def vis_detections(im, class_name, dets, thresh=0.3, imagename='im'):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.axis('off')   
            plt.tight_layout()
            plt.draw()
            plt.savefig('/home/andras/github/logoretrieval/resultimages/' + imagename)
            plt.close()


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


    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(PROTO, MODEL, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(MODEL))[0]

    imdb = get_imdb('srf_ski_logo_good')
    imdb.competition_mode(args.comp_mode)

    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    _t = {'im_detect' : Timer(), 'misc' : Timer()}


    i = 0
    for i in xrange(num_images):
        imagepath = imdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, False, None)
        #print scores
        #boxes = boxes_det
        if visualize_logo_detection:
            s_det = scores[:, 1]
            inds = np.array(s_det).argsort()[::-1][:10]
            roi_classes = ['logo' for j in range(len(inds))]
            write_bboxes(im, imagename, boxes[inds], s_det[inds], roi_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()

        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets, 0.3, imagepath.split('/')[-1].split('.')[0])
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()


        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
        i += 1

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

