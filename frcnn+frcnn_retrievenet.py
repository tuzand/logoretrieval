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
from fast_rcnn.test_det import im_detect as im_detect_det
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import os
os.environ['GLOG_minloglevel'] = '2'
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
import skimage as ski
from skimage.transform import rescale, resize, downscale_local_mean

max_per_image = 0
logo_threshold = 0.1
similarity_threshold = 0.7

recognition_score = True
cosinesimilarity = True


#VGG_CNN_M & VGG16
#featurelayer = 'fc7'

# ResNet
featurelayer = 'fc1000'

# SqeezeNet
#featurelayer = 'pool10'

# Inception BN
#featurelayer = 'fc1'

# eigenT
#featurelayer = 'feat'

visualize_logo_detection = True
visualize_logo_recognition = False
np.set_printoptions(threshold=np.nan)


FRCNN = 'py_faster_rcnn'

# PROPOSAL NETWORK MASTER THESIS
PROPOSALPROTO = os.path.join(FRCNN, 'models/logo_detection/VGG16/test.prototxt')
PROPOSALMODEL = os.path.join(FRCNN, 'output/final/allnet_srf_det_cl_reducedlr/vgg16_faster_rcnn_detection_iter_4000.caffemodel')

# PROPOSAL NETWORK PAPER
PROPOSALPROTO = os.path.join(FRCNN, 'models/logo_detection/VGG16/test.prototxt')
PROPOSALMODEL = os.path.join(FRCNN, '..', 'train/publicNonFlickr_ownlogo_detection_vgg16/vgg16_faster_rcnn_detection_iter_30000.caffemodel')

#PUBLIC Multiclass VGG16
FASTPROTO = os.path.join(FRCNN, 'models/logo/VGG16_48/fast_test.prototxt')
FASTERPROTO = os.path.join(FRCNN, 'models/logo/VGG16_48/test.prototxt')
MODEL = os.path.join(FRCNN, '..', 'train/publicNonFlickr_vgg16/vgg16_faster_rcnn_publiclogo_iter_50000.caffemodel')


FASTMODEL = MODEL
FASTERMODEL = MODEL


####### CLASSIFIERS MASTER THESIS
# VGG_CNN_M
#CLASSIFIERPROTO = ('/home/andras/data/models/vgg_cnn_m_1024/test.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, 'data/imagenet/VGG_CNN_M_1024.v2.caffemodel')

# AlexBN
#CLASSIFIERPROTO = ('/home/andras/data/models/alexbn/test.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/alexbn/alexbn.caffemodel')

# VGG-16
#CLASSIFIERPROTO = ('/home/andras/data/models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, 'data/imagenet/VGG16.v2.caffemodel')

# ResNet50
#CLASSIFIERPROTO = ('/home/andras/data/models/resnet/ResNet-50-deploy.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/resnet/ResNet-50-model.caffemodel')

#CLASSIFIERPROTO = ('/home/andras/data/models/resnet/ResNet-101-deploy.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/resnet/ResNet-101-model.caffemodel')

# SqueezeNet
#CLASSIFIERPROTO = ('/home/andras/data/models/squeezenet/test.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/squeezenet/squeezenet.caffemodel')

# Inception BN
#CLASSIFIERPROTO = ('/home/andras/data/models/inception_bn/deploy.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/inception_bn/Inception21k.caffemodel')

#CLASSIFIERPROTO = ('/home/andras/data/models/eigenT/deploy.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/eigenT/eigenT_32_iter_5000.caffemodel')


#CLASSIFIERPROTO = ('/home/andras/data/models/resnet_112/deploy.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/resnet_112/10k/resnet_50_iter_10000.caffemodel')

#CLASSIFIERPROTO = ('/home/andras/data/models/vgg_cnn_m_1024/test_219.prototxt')
#CLASSIFIERMODEL = ('/home/andras/data/models/vgg_cnn_m_1024/vgg_cnn_m_train_iter_7000.caffemodel')


########## CLASSIFIERS PAPER
# VGG_16 PUBLIC
CLASSIFIERPROTO = ('train/vgg16_public/deploy.prototxt')
CLASSIFIERMODEL = ('train/vgg16_public/VGG16_public_iter_300.caffemodel')

# VGG_16 PUBLIC + LitW
#CLASSIFIERPROTO = ('train/vgg16_public/deploy.prototxt')
#CLASSIFIERMODEL = ('train/vgg16_public_litw/VGG16_public_litw_iter_2975.caffemodel')

# RESNET-101 PUBLIC
#CLASSIFIERPROTO = os.path.join(FRCNN, '..', 'train/resnet_101_public/deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, '..', 'train/resnet_101_public/resnet_101_224_public_iter_800.caffemodel')


# RESNET-101 PUBLIC + LitW
#CLASSIFIERPROTO = os.path.join(FRCNN, '..', 'train/resnet_101_public_litw/deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, '..', 'train/resnet_101_public_litw/resnet_101_224_public_litw_iter_4740.caffemodel')


# DENSENET PUBLIC
#CLASSIFIERPROTO = os.path.join(FRCNN, '..', 'train/densenet_161_public/deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, '..', 'train/densenet_161_public/densenet_161_224_public_iter_900.caffemodel')

# DENSENET PUBLIC + LitW
#CLASSIFIERPROTO = os.path.join(FRCNN, '..', 'train/densenet_161_public_litw/deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, '..', 'train/densenet_161_public_litw/densenet_161_224_public_iter_8130.caffemodel')


#QUERYPATH = '/home/atuezkoe/data/datasets/schalke_query_crop/'
QUERYPATH = '/home/atuezkoe/data/datasets/fl_32_train_queries'

SEARCHPATH = 'fl_test'
#SEARCHPATH = 'schalke'

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
    plt.savefig('/home/atuezkoe/github/LogoRetrieval/resultimages/' + imagename.split('.')[0] + '.jpg')
    plt.close()

def detect_net(net, imdb, onlymax, max_per_image=100):
    if not net:
        net = caffe.Net(PROPOSALPROTO, PROPOSALMODEL, caffe.TEST)
        print "PROPOSALPROTO: " + PROPOSALPROTO
        print "PROPOSALMODEL: "+ PROPOSALMODEL
        net.name = os.path.splitext(os.path.basename(PROPOSALMODEL))[0]
        cfg.TEST.HAS_RPN = True

    num_images = len(imdb.image_index)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    detections = dict()
    allscores = dict()
    for i in xrange(num_images):
        imagepath = imdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        _t['im_detect'].tic()
        scores, boxes = im_detect_det(net, im, False, None)

        _t['misc'].tic()
        logo_inds = list()
        if onlymax:
            logo_inds.append(scores[:, 1].argmax())
        else:
            logo_inds = np.where(scores[:, 1] > logo_threshold)[0]
        if visualize_logo_detection:
            roi_classes = ['logo' for j in range(len(logo_inds))]
            write_bboxes(im, imagename, boxes[logo_inds], scores[logo_inds, 1], roi_classes)
        _t['im_detect'].toc()

        roi_bboxes = np.zeros((len(logo_inds),4))
        scores = scores[logo_inds, 1]
        for j, idx in enumerate(logo_inds):
            roi_bboxes[j] = boxes[idx, 4:8]

        detections[imagename] = roi_bboxes
        allscores[imagename] = scores


        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return detections, allscores

def cls_net(imdb, faster = False, onlymax = False, rpndetection = False, max_per_image=100):
    
    num_images = len(imdb.image_index)
    if not faster:
        cfg.TEST.HAS_RPN = False
        net = caffe.Net(FASTPROTO, FASTMODEL, caffe.TEST)
        print "FastProto: " + FASTPROTO
        print "FastModel: " + FASTMODEL
        net.name = os.path.splitext(os.path.basename(MODEL))[0]
    else:
        cfg.TEST.HAS_RPN = True
        net = caffe.Net(FASTERPROTO, FASTERMODEL, caffe.TEST)
        print "FasterProto: " + FASTERPROTO
        print "FasterModel: " + FASTERMODEL
        net.name = os.path.splitext(os.path.basename(MODEL))[0]

	# timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    normed_features = dict()
    detections = dict()
    allscores = dict()
    if not faster and not onlymax:
        dets, allscores = detect_net(net=None, imdb=imdb, onlymax=False)
        cfg.TEST.HAS_RPN = False
        detections = dets

    for i in xrange(num_images):
        imagepath = imdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        _t['im_detect'].tic()
        if not faster:
            if onlymax:
                box = np.zeros((1,4))
                box[0] = [0,0,im.shape[1], im.shape[0]]
                scores, boxes, features = im_detect(net, im, box, detection=False, customfeatures=True)
            else:
                scores, boxes, features = im_detect(net, im, boxes=dets[imagename], detection=False, customfeatures=True)
            logo_inds = list(range(0, len(scores)))
        else:
            scores, boxes, features, scores_det, boxes_det = im_detect(net, im, boxes=None, customfeatures=True, detection = True, rpndet=rpndetection)
            # using the bb of the class with the bigest score is better
            b = np.zeros((len(boxes), 4))
            for indx, s in enumerate(scores):
                max_indx = s[1:].argmax()
                b[indx] = boxes[indx, max_indx*4: (max_indx+1)*4]
            boxes = b
            if rpndetection:
                k = 0
            else:
                k = 1
            logo_inds = []
            if onlymax:
                logo_inds.append(scores_det[:, k].argmax())
            else:
                logo_inds = np.where(scores_det[:, k] > logo_threshold)[0]
            boxes = boxes[logo_inds, :]
            allscores[imagename] = scores_det[logo_inds, k]
            detections[imagename] = boxes


        _t['im_detect'].toc()
        _t['misc'].tic()
        roi_features = list()
        for idx in logo_inds:
            feature = features[idx]
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            feature = feature / norm
            roi_features.append(feature)

        normed_features[imagename] = roi_features

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return normed_features, detections, allscores, net

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

def classify(net, imdb, boxes):
    num_images = len(imdb.image_index)
    features = dict()
    j = 0
    for i in xrange(num_images):
        if j % 10 == 9:
            net = caffe.Net(CLASSIFIERPROTO, CLASSIFIERMODEL, caffe.TEST)
            net.name = os.path.splitext(os.path.basename(CLASSIFIERMODEL))[0]
        j += 1
        filepath = imdb.image_path_at(i)
        imagename = filepath.split('/')[-1]
        img = cv2.imread(filepath)
        roi_features = list()
        if not boxes:
            box = np.zeros((1,4))
            box = [0,0,img.shape[1], img.shape[0]]
            feat = getClassificatorFeatures(net, img, box)
            roi_features.append(feat)
        else:
            imgboxes = boxes[imagename]
            for idx in range(len(imgboxes)):
                feature = getClassificatorFeatures(net, img, imgboxes[idx])
                roi_features.append(feature)
        features[imagename] = roi_features
        print str(i) + "/" + str(num_images)

    return features


def getClassificatorFeatures(net, im, box):
    dummy = np.array([[[[1]]]]).astype(np.float32)
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    roi = im[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))
    #roi = roi - np.array([103.939, 116.779, 123.68])
    roi = roi.swapaxes(0,2).swapaxes(1,2)
    roi = np.array([roi]).astype(np.float32)
    net.set_input_arrays(roi, dummy)
    net.forward()

    feature = net.blobs[featurelayer].data
    feature = feature.flatten()
    norm = np.linalg.norm(feature)
    ret = feature / norm
    return ret

scoresarray = []
boxesarray = []
def process(net, imdb, query_features, all_search_features, dets, boxscores, fps, thres):
    global scoresarray
    global boxesarray
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    net = None
    _t = {'misc' : Timer(), 'cd' : Timer(), 'th' : Timer()}
    logotimes = list()
    [logotimes.append(0) for i in range(imdb.num_classes)]
    logoareas = list()
    [logoareas.append(0) for i in range(imdb.num_classes)]
    i = 0
    print thres
    for i in xrange(num_images):
        searchfilepath = imdb.image_path_at(i)
        searchfilename = searchfilepath.split('/')[-1]
        search_features = all_search_features[searchfilename]
        search_bboxes = dets[searchfilename]
        bscores = boxscores[searchfilename]

        if len(scoresarray) > i:
            scores = np.copy(scoresarray[i])
            boxes = np.copy(boxesarray[i])
        else:
            print "calc"
            scores = np.zeros((len(search_bboxes), imdb.num_classes))
            boxes = np.zeros((len(search_bboxes), imdb.num_classes * 4))
            for queryfilename, query_feature in query_features.items():

                # Calculate similarity between logos
                for j in range(len(search_features)):
                    search_feature = search_features[j]
                    if cosinesimilarity:
                        similarity = np.dot(query_feature, search_feature)
                    else:
                        curDist = np.sum((query_feature - search_feature)**2, axis=1)
                        similarity = 1.0 / (1.0 + curDist)
                    # get the index of the classname
                    classindex = imdb.classes.index(queryfilename.split('.')[0].split('_')[0])
                    if recognition_score:
                        s = similarity * bscores[j]
                        if scores[j, classindex] < s:
                            scores[j, classindex] = s
                            boxes[j, classindex*4 : classindex*4+4] = search_bboxes[j]
                    elif scores[j, classindex] < similarity:
                        scores[j, classindex] = similarity
                        boxes[j, classindex*4 : classindex*4+4] = search_bboxes[j]                

            scoresarray.append(np.copy(scores))
            boxesarray.append(np.copy(boxes))

        img_dets = None

        all_classes = np.array(())
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thres)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets
            if cls_dets != None:
                classes = np.zeros(cls_dets.shape[0])
                classes.fill(j)
                all_classes = np.append(all_classes, classes)
                #cls_dets = np.hstack((cls_dets, classes[:, np.newaxis])) \
                #    .astype(np.float32, copy=False)
                if img_dets == None:
                    img_dets = cls_dets
                else:
                    img_dets = np.concatenate((img_dets,cls_dets))
        keep = nms(img_dets, cfg.TEST.NMS)
        img_dets = img_dets[keep]
        all_classes = all_classes[keep]
        img_dets = np.hstack((img_dets, all_classes[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
        if visualize_logo_recognition:
            im = cv2.imread(searchfilepath)
            classArray = [imdb.classes[int(img_dets[l, 5])] for l in range(len(img_dets))]
            write_bboxes(im, searchfilename, img_dets[:, :4], img_dets[:, 4], classArray)
        for j in xrange(1, imdb.num_classes):
            cls_dets = img_dets[img_dets[:, 5] == j][:, :5]
            all_boxes[j][i] = cls_dets

        # Calculating cumulated area
        '''im=Image.open(searchfilepath)
        width, height = im.size
        imagearea = width * height
        for j in xrange(1, imdb.num_classes):
            if all_boxes[j][i].size:
                logotimes[j] += 1.0/float(fps)
                logos = all_boxes[j][i]
                for logo in logos:
                    w = logo[2] - logo[0]
                    h = logo[3] - logo[1]
                    logoareas[j] += float(w*h) / float(imagearea)'''

        i += 1

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    i = 0
    #print logoareas
    #print logotimes
    #for area, time in zip(logoareas, logotimes):
    #    print imdb.classes[i] + ": " + "{:.2f}".format(logotimes[i]) + "s, " + "{:.2f}".format(logoareas[i]) + " full frames"
    #    i += 1

    print 'Evaluating detections'
    rec, prec, map, tp, fp, npos = imdb.evaluate_detections(all_boxes, output_dir)
    return rec, prec, map, tp, fp, num_images, npos

g_all_search_features = None
g_dets = None
g_boxscores = None
g_net = None
def search(fps, fold):
    global g_all_search_features, g_dets, g_boxscores, g_net
    solution = 4
    imdb = get_imdb(SEARCHPATH)
    if SEARCHPATH == 'schalke':
        queryimdb = get_custom_imdb(QUERYPATH)
    else:
        queryimdb = get_custom_imdb(os.path.join(QUERYPATH, str(fold)))
    timer = Timer()
    if solution == 1:
        timer.tic()
        query_features, b, boxscores,  net = cls_net(queryimdb, faster=True, onlymax=True, rpndetection=True)
        all_search_features, dets, boxscores, net = cls_net(imdb, faster=True, onlymax=False, rpndetection=True)
        timer.toc()
    elif solution == 2:
        timer.tic()
        query_features, b, boxscores, net = cls_net(queryimdb, faster=False, onlymax=True, rpndetection=False)
        if g_all_search_features == None:
            all_search_features, dets, boxscores, net = cls_net(imdb, faster=True, onlymax=False, rpndetection=True)
            g_all_search_features = all_search_features
            g_dets = dets
            g_boxscores = boxscores
            g_net = net
        else:
            all_search_features = g_all_search_features
            dets = g_dets
            boxscores = g_boxscores
            net = g_net
        resfilename = 'multiclass' + FASTERMODEL.split('/')[-1].split('.')[0] + '_' + str(fold) + '_' + SEARCHPATH + '_results.txt'
        timer.toc()
    elif solution == 3:
        timer.tic()
        query_features, b, boxscores, net = cls_net(queryimdb, faster=False, onlymax=True, rpndetection=False)
        all_search_features, dets, boxscores, net = cls_net(imdb, faster=False, onlymax=False, rpndetection=False)
        timer.toc()
    elif solution == 4:
        timer.tic()
        if g_all_search_features == None:
            net = caffe.Net(CLASSIFIERPROTO, CLASSIFIERMODEL, caffe.TEST)
            net.name = os.path.splitext(os.path.basename(CLASSIFIERMODEL))[0]
            dets, boxscores = detect_net(net=None, imdb=imdb, onlymax=False)
            print "CLASSIFIERPROTO: " + CLASSIFIERPROTO
            print "CLASSIFIERMODEL: " + CLASSIFIERMODEL
            all_search_features = classify(net, imdb, dets)
            g_all_search_features = all_search_features
            g_dets = dets
            g_boxscores = boxscores
            g_net = net
        else:
            all_search_features = g_all_search_features
            dets = g_dets
            boxscores = g_boxscores
            net = g_net
        query_features = classify(net, queryimdb, None)
        timer.toc()
        resfilename = 'det+cls_' + CLASSIFIERMODEL.split('/')[-1].split('.')[0] + '_' + str(fold) + '_' + SEARCHPATH + '_results.txt'
    elif solution == 5:
        timer.tic()
        query_features, b, boxscores, net = cls_net(queryimdb, faster=False, onlymax=True, rpndetection=False)
        all_search_features, dets, boxscores, net = cls_net(imdb, faster=True, onlymax=False, rpndetection=False)
        timer.toc()

    if os.path.isfile(resfilename):
        os.remove(resfilename)

    with open(resfilename, 'a') as f:
        for t in np.arange(0.99, 0.009, -0.01):
            thres = t #similarity_threshold
            rec, prec, map, tp, fp, num_images, npos = process(net, imdb, query_features, all_search_features, dets, boxscores, fps, thres)
            if len(rec) == 0:
                rec = [0.0]
            f.write(str(thres) + '\t' + str(fp/float(num_images)) + '\t' + str(100.0*tp/npos) + '\t' + str(map) + '\n')
    print 'Feature extraction time: {:.3f}s'.format(timer.average_time)


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


    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()

    global scoresarray
    global boxesarray
    
    if SEARCHPATH == 'schalke':
        for i in range(0,10):
            scoresarray = []
            boxesarray = []
            search(args.fps, 0)
    else:
        for i in range(0,10):
            scoresarray = []
            boxesarray = []
            search(args.fps, i)
