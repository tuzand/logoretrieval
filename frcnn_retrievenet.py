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
import operator
import matplotlib.pyplot as plt
from custom_imdb import get_custom_imdb

logo_threshold = 0.1
similaritythreshold = 0.1
thresh = 0.0
RESULTPATH = './results/'
RESULTPOSTFIX = '.result2.txt'

FRCNN = 'py_faster_rcnn'

PROTO = os.path.join(FRCNN, 'models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/simple/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/faster_rcnn_alt_opt_simple/faster_rcnn_test.pt')
MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/default/train/fl_faster_rcnn_final.caffemodel')
EXAMPLEPATH = '/home/andras/data/datasets/fussi/example'
SEARCHPATH = '/home/andras/data/datasets/fussi'

def write_bboxes(im, imagename, bboxArray, scoreArray, classArray):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(bboxArray)):
        bbox = bboxArray[i]
        score = scoreArray[i]
        class_name = classArray[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #ax.set_title(('Detections with '
    #              'p(obj | box) >= {:.1f}').format(logo_threshold),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('/home/andras/github/logoretrieval/resultimages/' + imagename)
    plt.close()


def test_net(net, imdb, onlymax, max_per_image=100):
	num_images = len(imdb.image_index)

	# timers
	_t = {'im_detect' : Timer(), 'misc' : Timer()}
	normed_features = dict()
	outputbboxes = dict()
	for i in xrange(num_images):
		imagepath = imdb.image_path_at(i)
		imagename = imagepath.split('/')[-1]
		im = cv2.imread(imagepath)
		_t['im_detect'].tic()
		scores, boxes, features, scores_det = im_detect(net, im, True, None)
		_t['im_detect'].toc()

		_t['misc'].tic()
		max_score = 0
                roi_bboxes = list()
		roi_scores = list()
		roi_classes = list()
		roi_features = list()
		logo_inds = list()
		if False:
			logo_inds.append(scores_det[:, 1].argmax())
			print scores_det[:, 1].max()
		else:
			#print scores
			for i in range(len(scores)):
				m = scores[i, 1:].max()
				if m > logo_threshold:
					logo_inds.append(i)
				#logo_inds.extend(np.where(scores_det[:, 1:] >= logo_threshold)[0])
		for idx in logo_inds:
			score = scores[idx, :]
			print score
			max_score = scores[idx, 1:].max()
			brand_idx = score.argmax()
			bbox = boxes[idx, 4 * brand_idx : 4 * (brand_idx + 1)]
                        feature = features[idx, :]
                	feature = feature.flatten()
	                norm = np.linalg.norm(feature)
        	        feature = feature / norm
			roi_bboxes.append(bbox)
			roi_scores.append(max_score)
			roi_classes.append('logo')
                        roi_features.append(feature)
		#write_bboxes(im, imagename, roi_bboxes, roi_scores, roi_classes)
		normed_features[imagename] = [imagepath, roi_features, roi_bboxes, roi_scores]

		#for j in xrange(1, imdb.num_classes):
		#	inds = np.where(scores[:, j] > thresh)[0]
		#	if len(scores[inds, j]) > 0:
		#		max_class_score = scores[inds, j].max()
		#		if max_class_score > logo_threshold and \
		#				max_class_score > max_score:
		#			max_score = max_class_score
		#			bbox_index = scores[inds, j].argmax()
		#			bbox = boxes[bbox_index, 4 * j : 4 * (j + 1)]
		#			feature = scores[bbox_index, :]
		#			feature = feature.flatten()
		#			norm = np.linalg.norm(feature)
		#			feature = feature / norm
		#normed_features[imagename] = [imagepath, feature]
		#outputbboxes[imagename] = np.asarray([bbox])
		#print outputbboxes[imagename]
		#print type(outputbboxes[imagename])
		#print type(outputbboxes[imagename][0])
		
		_t['misc'].toc()

		print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
	          .format(i + 1, num_images, _t['im_detect'].average_time,
	                  _t['misc'].average_time)

	
	return normed_features, outputbboxes

def get_features(net, args, dataset, custom, onlymax):
    if custom:
        imdb = get_custom_imdb(dataset)
    else:
        imdb = get_imdb(dataset)
        imdb.competition_mode(args.comp_mode)
        #if not cfg.TEST.HAS_RPN:
            #imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    return test_net(net, imdb, onlymax, max_per_image=args.max_per_image)

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

	#while not os.path.exists(args.caffemodel) and args.wait:
	#	print('Waiting for {} to exist...'.format(args.caffemodel))
	#	time.sleep(10)
	

	caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        net = caffe.Net(PROTO, MODEL, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(MODEL))[0]

        #all_test_features, all_test_bboxes = get_features(net, args, 'fl_test_logo', custom=False, onlymax=True)
        all_test_features, all_test_bboxes = get_features(net, args, EXAMPLEPATH, custom=True, onlymax=True)


	#caffe.set_mode_gpu()
	#caffe.set_device(args.gpu_id)
	#net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	#net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

	#cfg.TEST.HAS_RPN = False
	print "Test scores: " + str(len(all_test_features))
        
	#train_features = test_features

	#cfg.TEST.HAS_RPN = True
	#train_features, train_bboxes = get_features(net, args, 'fl_trainval', onlymax=False)
        all_train_features, all_train_bboxes = get_features(net, args, SEARCHPATH, custom=True, onlymax=False)

	print "Train scores: " + str(len(all_train_features))



	for testfilename, test_value in all_test_features.items():
            testfilepath = test_value[0]
            test_features = test_value[1]
            test_bboxes = test_value[2]
            sortedResult = []
            for test_feature in test_features:
                for trainfilename, train_value in all_train_features.items():
                    roi_bboxes = list()
                    roi_scores = list()
                    roi_classes = list()
                    trainfilepath = train_value[0]
                    train_features = train_value[1]
                    train_bboxes = train_value[2]
                    for i in range(len(train_features)):
                        train_feature = train_features[i]
                        #distance = np.dot(test_feature, train_feature)
			#print (test_feature-train_feature)
			distance = np.sum((test_feature - train_feature)**2, axis=0)
			print distance
                        #if distance >= similaritythreshold:
			if True:
                            roi_bboxes.append(train_bboxes[i])
                            roi_scores.append(1/distance)
                            roi_classes.append('srf')
                    im = cv2.imread(trainfilepath)
                    write_bboxes(im, trainfilename, roi_bboxes, roi_scores, roi_classes)


                '''for testfilename, test_value in test_features.items():
		testfilepath = test_value[0]
		test_feature = test_value[1]
		sortedResult = []
		if test_feature != None:
			result = dict()
			for trainfilename, train_value in train_features.items():
				trainfilepath = train_value[0]
				train_feature = train_value[1]
				result[trainfilepath] = np.dot(test_feature, train_feature)

			sortedResult = sorted(result.items(), \
				key=operator.itemgetter(1), reverse=True)
		else:
			print testfilename

		if not os.path.exists(RESULTPATH):
			os.makedirs(RESULTPATH)
		with open(os.path.join(RESULTPATH, testfilename \
						+ RESULTPOSTFIX), 'w') as res:
			res.write(testfilepath + " " + str(1.0) + "\n")
			for i in range(len(sortedResult)):
				out = str(sortedResult[i][0]) + " "  \
						+ str(sortedResult[i][1]) + "\n"
				res.write(out)'''
