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
similarity_threshold = 0.6
RESULTPATH = './results/'
RESULTPOSTFIX = '.result2.txt'

detectionpathexists = False

classifier = False
featurelayer = 'fc1000'

visualize_logo_recognition = False
visualize_logo_detection = False
np.set_printoptions(threshold=np.nan)


FRCNN = 'py_faster_rcnn'

# PROPOSAL NETWORK
PROPOSALPROTO = os.path.join(FRCNN, 'models/logo_detection/VGG16/test.prototxt')
PROPOSALMODEL = os.path.join(FRCNN, 'output/final/allnet_srf_det_cl_reducedlr/vgg16_faster_rcnn_detection_iter_4000.caffemodel')

#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/sharedexceptlast/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/test.prototxt')

PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_sharedconv/test.prototxt')
MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_logos32plus_fl_test_sharedconv/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')

#PROTO = os.path.join(FRCNN, 'models/fl/VGG_CNN_M_1024/faster_rcnn_end2end/simple/test.prototxt')
#PROTO = os.path.join(FRCNN, 'models/fl/faster_rcnn_alt_opt_simple/faster_rcnn_test.pt')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_ignorelabel/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_ignorelabel_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_sharedconv_v2/vgg_cnn_m_1024_faster_rcnn_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_logos32plus_sharedconv/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')
MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/allnet_logos32plus_fl_test_sharedconv/vgg_cnn_m_1024_faster_rcnn_allnet_sharedconv_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/fl_train+fl_val_logo/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/faster_rcnn_end2end/sharedexceptlast_v2/vgg_cnn_m_1024_faster_rcnn_fl_iter_80000.caffemodel')
#MODEL = os.path.join(FRCNN, 'output/default/train/fl_faster_rcnn_final.caffemodel')

#FL32 simple
FASTPROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/fast_test.prototxt')
PROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/simple_fl/test.prototxt')
MODEL = os.path.join(FRCNN, 'output/final/fl_vgg_cnn_m/vgg_cnn_m_1024_faster_rcnn_simple_fl_iter_80000.caffemodel')

#ALLNET simple
FASTPROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_simple/fast_test.prototxt')
FASTERPROTO = os.path.join(FRCNN, 'models/logo/VGG_CNN_M_1024/faster_rcnn_end2end/allnet_simple/test.prototxt')
MODEL = os.path.join(FRCNN, 'output/final/allnet_vgg_cnn_m_single_all_for_fl_test/vgg_cnn_m_1024_faster_rcnn_allnet_simple_iter_120000.caffemodel')

#ALLLOGO simple
#FASTPROTO = os.path.join(FRCNN, 'models/logo/VGG16_219/fast_test.prototxt')
#FASTERPROTO = os.path.join(FRCNN, 'models/logo/VGG16_219/test.prototxt')
#MODEL = os.path.join(FRCNN, 'output/final/alllogo_simple_vgg16/vgg16_faster_rcnn_alllogo_iter_120000.caffemodel')
FASTMODEL = MODEL
FASTERMODEL = MODEL


# CLASSIFIERS
# VGG_CNN_M
CLASSIFIERPROTO = ('/home/andras/data/models/vgg_cnn_m_1024/test.prototxt')
CLASSIFIERMODEL = os.path.join(FRCNN, 'data/imagenet/VGG_CNN_M_1024.v2.caffemodel')

# VGG-16
#CLASSIFIERPROTO = ('/home/andras/data/models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt')
#CLASSIFIERMODEL = os.path.join(FRCNN, 'data/imagenet/VGG16.v2.caffemodel')

QUERYPATH = '/home/andras/data/misc/schalke_query_crop/'

SEARCHPATH = 'schalke'

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
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2]) - x
        h = int(bbox[3]) - y
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite('/home/andras/github/logoretrieval/resultimages/' + imagename.split('.')[0] + '.jpg', im)

def detect_net(net, imdb, onlymax, max_per_image=100):
    if not net:
        net = caffe.Net(PROPOSALPROTO, PROPOSALMODEL, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(PROPOSALMODEL))[0]
        cfg.TEST.HAS_RPN = True

    num_images = len(imdb.image_index)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    detections = dict()
    for i in xrange(num_images):
        imagepath = imdb.image_path_at(i)
        imagename = imagepath.split('/')[-1]
        im = cv2.imread(imagepath)
        _t['im_detect'].tic()
        #scores, boxes, features, scores_det, boxes_det = im_detect(net, im, None, detectionpathexists)
        scores, boxes = im_detect_det(net, im, False, None) #detectionpathexists)

        if visualize_logo_detection:
            s_det = scores[:, 1]
            inds = np.array(s_det).argsort()[::-1][:10]
            roi_classes = ['logo' for j in range(len(inds))]
            write_bboxes(im, imagename, boxes[inds], s_det[inds], roi_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        logo_inds = list()
        if onlymax:
            logo_inds.append(scores[:, 1].argmax())
        else:
            logo_inds = np.where(scores[:, 1] > logo_threshold)[0]
        roi_bboxes = np.zeros((len(logo_inds),4))
        
        for j, idx in enumerate(logo_inds):
            roi_bboxes[j] = boxes[idx, 4:8]

        detections[imagename] = roi_bboxes


        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return detections

def cls_net(imdb, faster = False, onlymax = False, rpndetection = False, max_per_image=100):
    
    num_images = len(imdb.image_index)
    if not faster:
        cfg.TEST.HAS_RPN = False
        net = caffe.Net(FASTPROTO, FASTMODEL, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(MODEL))[0]
    else:
        cfg.TEST.HAS_RPN = True
        net = caffe.Net(FASTERPROTO, FASTERMODEL, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(MODEL))[0]

	# timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    normed_features = dict()
    detections = dict()
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
                boxes = box
            else:
                dets = detect_net(net=None, imdb=imdb, onlymax=False)
                scores, boxes, features = im_detect(net, im, boxes=dets, detection=False, customfeatures=True)
            logo_inds = list(range(0, len(scores)))
        else:
            scores, boxes, features, scores_det, boxes_det = im_detect(net, im, boxes=None, customfeatures=True, detection = True, rpndet=rpndetection)
            
            boxes = boxes_det[:, 1:]
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
            draw_boxes(im, boxes, imagepath.split('/')[-1].split('.')[0])

        _t['im_detect'].toc()
        _t['misc'].tic()
        roi_features = list()
        #logo_inds = list(range(0, len(scores)))
        for idx in logo_inds:
            feature = features[idx]
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            feature = feature / norm
            roi_features.append(feature)

        detections[imagename] = boxes
        normed_features[imagename] = roi_features

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return normed_features, detections, net

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
        value[1] = list()
        features = value[1]
        bboxes = value[2]
        img = cv2.imread(filepath)
        bs = np.array(bboxes)
        
        scores, bs, fs = im_detect(net, img, False, bs)
        value[2] = list()
        bboxes = value[2]
        for idx, feature in enumerate(fs):
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            feature = feature / norm
            features.append(feature)
            s = scores[idx, 1:]
            max_score_idx = s.argmax()
            bboxes.append(bs[idx, 4*max_score_idx : 4*(max_score_idx + 1)])
        i += 1
        print str(i) + "/" + str(len(boxes))

def classify(net, imdb, boxes):
    num_images = len(imdb.image_index)
    features = dict()
    for i in xrange(num_images):
        filepath = imdb.image_path_at(i)
        imagename = filepath.split('/')[-1]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi_features = list()
        if not boxes:
            box = np.zeros((1,4))
            box = [0,0,img.shape[1], img.shape[0]]
            roi_features.append(getClassificatorFeatures(net, img, box))
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
    roi = roi.swapaxes(0,2).swapaxes(1,2)
    roi = np.array([roi]).astype(np.float32)
    net.set_input_arrays(roi, dummy)
    net.forward()

    feature = net.blobs[featurelayer].data
    #feature = net.blobs['prob'].data
    feature = feature.flatten()
    norm = np.linalg.norm(feature)
    return feature / norm

def process(net, imdb, query_features, all_search_features, dets, fps, thres):
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
    print thres
    for i in xrange(num_images):
        searchfilepath = imdb.image_path_at(i)
        searchfilename = searchfilepath.split('/')[-1]
        _t['misc'].tic()
        search_features = all_search_features[searchfilename]
        search_bboxes = dets[searchfilename]

        scores = np.zeros((len(search_bboxes), imdb.num_classes))
        boxes = np.zeros((len(search_bboxes), imdb.num_classes * 4))
        for queryfilename, query_feature in query_features.items():

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
    #print logoareas
    #print logotimes
    #for area, time in zip(logoareas, logotimes):
    #    print imdb.classes[i] + ": " + "{:.2f}".format(logotimes[i]) + "s, " + "{:.2f}".format(logoareas[i]) + " full frames"
    #    i += 1

    print 'Evaluating detections'
    rec, prec, map, tp, fp, npos = imdb.evaluate_detections(all_boxes, output_dir)
    return rec, prec, map, tp, fp, num_images, npos


def search(fps):
    solution = 4
    imdb = get_imdb(SEARCHPATH)
    queryimdb = get_custom_imdb(QUERYPATH)
    timer = Timer()
    if solution == 1:
        timer.tic()
        query_features, b, net = cls_net(queryimdb, faster=True, onlymax=True, rpndetection=True)
        all_search_features, dets, net = cls_net(imdb, faster=True, onlymax=False, rpndetection=True)
        timer.toc()
    elif solution == 2:
        query_features, b, net = cls_net(queryimdb, faster=False, onlymax=True, rpndetection=False)
        all_search_features, dets, net = cls_net(imdb, faster=True, onlymax=False, rpndetection=True)
    elif solution == 3:
        pass
    elif solution == 4:
        net = caffe.Net(CLASSIFIERPROTO, CLASSIFIERMODEL, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(CLASSIFIERMODEL))[0]
        dets = detect_net(net=None, imdb=imdb, onlymax=False)
        all_search_features = classify(net, imdb, dets)
        query_features = classify(net, queryimdb, None)

    resfilename = 'det_vgg_cnn_m_cls_' + SEARCHPATH + '_results.txt'
    if os.path.isfile(resfilename):
        os.remove(resfilename)

    with open(resfilename, 'a') as f:
        for t in np.arange(0.99, 0.009, -0.01):
            thres = similarity_threshold
            rec, prec, map, tp, fp, num_images, npos = process(net, imdb, query_features, all_search_features, dets, fps, thres)
            if len(rec) == 0:
                rec = [0.0]
            f.write(str(thres) + '\t' + str(fp/float(num_images)) + '\t' + str(tp/npos) + '\t' + str(map) + '\n')
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


    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    

    search(args.fps)
