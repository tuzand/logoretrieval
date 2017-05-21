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
import Image

max_per_image = 0
logo_threshold = 0.1
similarity_threshold = 0.9
RESULTPATH = './results/'
RESULTPOSTFIX = '.result2.txt'

visualize_logo_recognition = False
visualize_logo_detection = True
np.set_printoptions(threshold=np.nan)


FRCNN = 'py_faster_rcnn'

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/sharedexceptlast/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/test.prototxt')
PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/simple/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/faster_rcnn_alt_opt_simple/faster_rcnn_test.pt')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_ignorelabel_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_logos32plus_sharedconv/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')
MODEL = os.path.join(FRCNN, 'output/final/allnet_allnet_det_sharedconv_vgg_cnn_m/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/fl_train+fl_val_logo/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/sharedexceptlast_v2/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/default/train/fl_faster_rcnn_final.caffemodel')
QUERYPATH = '/home/andras/audi'
#SEARCHPATH = '/home/andras/audi'
#SEARCHPATH = '/home/andras/data/datasets/LegalSamples'
SEARCHPATH = 'srf_ski_good'

def write_bboxes(im, imagename, bboxArray, scoreArray, classArray):
    if len(bboxArray) == 0:
        return
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
                '{:s} {:.3f}'.format(str(class_name), score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('/home/andras/github/logoretrieval/resultimages/' + imagename.split('.')[0] + '.jpg')
    plt.close()

def vis_detections(im, class_name, dets, imagename, thresh=0.3):
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
            plt.savefig('/home/andras/github/logoretrieval/resultimages/' + imagename.split('.')[0] + '.jpg')
            plt.close()

def draw_boxes(im, dets, imagename):
    if dets.shape[0] == 0:
        return
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite('/home/andras/github/logoretrieval/resultimages/' + imagename.split('.')[0] + '.jpg', im)

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
        scores, boxes, features, scores_det, boxes_det = im_detect(net, im, None, True)
        if visualize_logo_detection:
            s_det = scores_det[:, 1]
            inds = np.array(s_det).argsort()[::-1][:1]
            roi_classes = ['logo' for j in range(len(inds))]
            draw_boxes(im, boxes[inds], imagename)
            #write_bboxes(im, imagename, boxes[inds], s_det[inds], roi_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        max_score = 0
        roi_bboxes = list()
        roi_scores = list()
        roi_classes = list()
        roi_features = list()
        logo_inds = list()
        if onlymax:
            logo_inds.append(scores_det[:, 1].argmax())
        else:
            logo_inds = np.where(scores_det[:, 1] > logo_threshold)[0]
        for idx in logo_inds:
            s = scores[idx, 1:]
            max_score_idx = s.argmax()
            max_score = s.max()
            feature = features[idx]
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            feature = feature / norm
            print feature
            roi_bboxes.append(boxes[idx, 4*max_score_idx : 4*(max_score_idx + 1)])
            roi_scores.append(max_score)
            roi_classes.append('logo')
            roi_features.append(feature)

        #normed_features[imagename] = [imagepath, roi_features, boxes[logo_inds, :]]
        print roi_bboxes
        normed_features[imagename] = [imagepath, roi_features, roi_bboxes]

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return normed_features

def get_features(net, args, dataset, custom, onlymax):
    if custom:
        imdb = get_custom_imdb(dataset)
    else:
        imdb = get_imdb(dataset)
        imdb.competition_mode(args.comp_mode)
        #if not cfg.TEST.HAS_RPN:
            #imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    return test_net(net, imdb, onlymax, max_per_image=args.max_per_image), imdb

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
    parser.add_argument('--fps', dest='fps', help='fps of the video',
                        default=30, type=float)

    args = parser.parse_args()
    return args

def updateFeatures(net, boxes):
    i = 0
    for filename, value in boxes.items():
        roi_bboxes = list()
        roi_scores = list()
        roi_classes = list()
        filepath = value[0]
        features = value[1]
        bboxes = value[2]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(bboxes):
            features[idx] = getClassificatorFeatures(net, img, box)
        i += 1
        print str(i) + "/" + str(len(boxes))

def getClassificatorFeatures(net, im, box):
    dummy = np.array([[[[1]]]]).astype(np.float32)
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    roi = im[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))
    roi = roi.swapaxes(0,2).swapaxes(1,2)
    roi = np.array([roi]).astype(np.float32)
    net.set_input_arrays(roi, dummy)
    net.forward()

    feature = net.blobs['fc1000'].data
    #feature = net.blobs['prob'].data
    feature = feature.flatten()
    norm = np.linalg.norm(feature)
    return feature / norm


def search(net, fps):

    all_query_features, imdb = get_features(net, args, QUERYPATH, custom=True, onlymax=True)
    d
    all_search_features, imdb = get_features(net, args, SEARCHPATH, custom=False, onlymax=False)

    #PROTO = '/home/andras/data/models/resnet/ResNet-50-deploy.prototxt'
    #MODEL = '/home/andras/data/models/resnet/ResNet-50-model.caffemodel'

    #PROTO = '/home/andras/data/models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt'
    #MODEL = '/home/andras/data/models/vgg_16/VGG_ILSVRC_16_layers.caffemodel'

    #PROTO = '/home/andras/data/models/resnext/deploy.prototxt'
    #MODEL = '/home/andras/data/models/resnext/resnext50.caffemodel'

    PROTO = '/home/andras/data/models/densenet201/deploy.prototxt'
    MODEL = '/home/andras/data/models/densenet201/DenseNet_201.caffemodel'

    PROTO = '/home/andras/data/models/voc/deploy.prototxt'
    MODEL = '/home/andras/data/models/voc/pascalvoc2012_train_simple2_iter_30000.caffemodel'


    net = caffe.Net(PROTO, MODEL, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(MODEL))[0]
    all_query_features = dict()
    queryimdb = get_custom_imdb(QUERYPATH)
    num_images = len(queryimdb.image_index)
    for i in xrange(num_images):
        imagepath = queryimdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        height, width, channels = im.shape
        all_query_features[imagename] = getClassificatorFeatures(net, im, (0,0,width,height))

    print 'Update search features'
    updateFeatures(net, all_search_features)
    
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    _t = {'misc' : Timer()}

    logotimes = list()
    [logotimes.append(0) for i in range(imdb.num_classes)]
    logoareas = list()
    [logoareas.append(0) for i in range(imdb.num_classes)]
    i = 0
    for i in xrange(num_images):
        searchfilepath = imdb.image_path_at(i)
        searchfilename = searchfilepath.split('/')[-1]
        _t['misc'].tic()
        search_value = all_search_features[searchfilename]
        searchfilepath = search_value[0]
        search_features = search_value[1]
        search_bboxes = search_value[2]
                
        scores = np.zeros((len(search_bboxes), imdb.num_classes))
        boxes = np.zeros((len(search_bboxes), imdb.num_classes * 4))
        for queryfilename, query_value in all_query_features.items():
            #query_feature = query_value[1]
            query_feature = query_value

            # Calculate similarity between logos
            for j in range(len(search_features)):
                search_feature = search_features[j]
                similarity = np.dot(query_feature, search_feature)
                # get the index of the classname
                classindex = imdb.classes.index(queryfilename.split('.')[0])
                scores[j, classindex] = similarity

                max_score_index = search_feature.argmax()
                boxes[j, classindex*4 : classindex*4+4] = search_bboxes[j]

        img_dets = None
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > similarity_threshold)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.2)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets
            if cls_dets != None:
                classes = np.zeros(cls_dets.shape[0])
                classes.fill(j)
                cls_dets = np.hstack((cls_dets, classes[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                if img_dets == None:
                    img_dets = cls_dets
                else:
                    img_dets = np.concatenate((img_dets,cls_dets))
        keep = nms(img_dets, 0.2)
        img_dets = img_dets[keep]
        if visualize_logo_recognition:
            im = cv2.imread(searchfilepath)
            classArray = [imdb.classes[int(img_dets[l, 5])] for l in range(len(img_dets))]
            write_bboxes(im, searchfilename, img_dets[:, :4], img_dets[:, 4], classArray)
        for j in xrange(1, imdb.num_classes):
            cls_dets = img_dets[img_dets[:, 5] == j]
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

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images,
                      _t['misc'].average_time)
        
        im=Image.open(searchfilepath)
        width, height = im.size
        imagearea = width * height
        for j in xrange(1, imdb.num_classes):
            if all_boxes[j][i].size:
                logotimes[j] += 1.0/float(fps)
                logos = all_boxes[j][i]
                for logo in logos:
                    w = logo[2] - logo[0]
                    h = logo[3] - logo[1]
                    logoareas[j] += float(w*h) / float(imagearea)

        i += 1


    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    i = 0
    print logoareas
    print logotimes
    for area, time in zip(logoareas, logotimes):
        print imdb.classes[i] + ": " + "{:.2f}".format(logotimes[i]) + "s, " + "{:.2f}".format(logoareas[i]) + " full frames"
        i += 1

    print 'Evaluating detections'
    return imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TEST.HAS_RPN = True
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BG_THRESH_LO = 0.0


    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)


    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(PROTO, MODEL, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(MODEL))[0]
    

    search(net, args.fps)
    sys.exit(0)

    maxMAP = 0.0
    t_max = 0.0
    lt_max = 0.0
    for lt in np.arange(0,1,0.1):
        for t in np.arange(0,1,0.1):
            logo_threshold = lt
            similarity_threshold = t
            print "Act Thresh: " + str(thresh)
            print "Act Logo Thresh: " + str(logo_threshold)
            print "MAP: " + str(maxMAP)
            map = search(net)
            print map
            if map > maxMAP:
                lt_max = logo_threshold
                t_max = thresh
                maxMAP = map
    print 'Thresh: ' + str(t_max)
    print 'Logo threshold: ' + str(lt_max)
    print 'MAP: ' + str(maxMAP)
