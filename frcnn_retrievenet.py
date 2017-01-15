#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
from utils.timer import Timer
import cv2
import numpy as np

logo_threshold = 0.7

def test_net(net, imdb, max_per_image=100, thresh=0.05):
	num_images = len(imdb.image_index)

	# timers
	_t = {'im_detect' : Timer(), 'misc' : Timer()}
	image_scores = []
	for i in xrange(num_images):
		im = cv2.imread(imdb.image_path_at(i))
		_t['im_detect'].tic()
		scores, boxes = im_detect(net, im, None)
		_t['im_detect'].toc()

		_t['misc'].tic()
		max_score = 0
		cls = None
		logo_scores = None
		for j in xrange(1, imdb.num_classes):
			inds = np.where(scores[:, j] > thresh)[0]
			cls_scores = scores[inds, j]
			if len(scores[inds, j]) > 0:
				max_class_score = scores[inds, j].max()
				if max_class_score > logo_threshold and \
					max_class_score > max_score:
					max_score = max_class_score
					cls = j
					bbox_index = inds[scores[inds, j].argmax()]
					logo_scores = scores[bbox_index, :]
		image_scores.append(logo_scores)
		_t['misc'].toc()

		print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
	          .format(i + 1, num_images, _t['im_detect'].average_time,
	                  _t['misc'].average_time)

	
	return image_scores

def get_features(net, args, dataset):
	imdb = get_imdb(dataset)
	imdb.competition_mode(args.comp_mode)
	if not cfg.TEST.HAS_RPN:
		imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
	return test_net(net, imdb, max_per_image=args.max_per_image)

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
                        help='optional config file', default=None, type=str)
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

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]


    train_scores = get_features(net, args, 'fl_train')
    print "Train scores: " + str(len(train_scores))
    test_scores = get_features(net, args, args.imdb_name)
    print "Test scores: " + str(len(test_scores))

    
