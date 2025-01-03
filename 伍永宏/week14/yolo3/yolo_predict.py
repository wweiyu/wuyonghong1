import tensorflow as tf
import os
import numpy as np
import scripts.lesson14.yolo.config as config
from scripts.lesson14.yolo.model.yolo_model import yolo3_model
import colorsys
import random

class yolo_predictor():
    def __init__(self, obj_threshold, nms_threshold, anchor_path, class_path):
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.anchors_path = anchor_path
        self.classes_path = class_path

        self.class_names = self._getclasses()
        self.anchors = self._getanchors()

        hsv_tuples = [(x / len(self.class_names), 1., 1.)for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)


    def _getclasses(self):
        path = os.path.expanduser(self.classes_path)
        with open(path, 'r') as ff:
            cc = ff.readlines()
            class_names = [s.strip() for s in cc]
            return class_names

    def _getanchors(self):
        path = os.path.expanduser(self.anchors_path)
        with open(path, 'r') as ff:
            cc = ff.readline()
            anchors = np.array(cc.split(',')).astype(np.float).reshape((-1,2))
            return anchors

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchor = len(anchors)

        anchors_tensor = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,num_anchor,2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats,[-1,grid_size[0],grid_size[1],num_anchor,num_classes+5])

        '给格子分配好坐标'
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]),[1,-1,1,1]),[grid_size[0],1,1,1])
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]),[-1,1,1,1]),[1,grid_size[1],1,1])
        grid = tf.concat([grid_x,grid_y],axis=-1)
        grid = tf.cast(grid,tf.float32)

        'x,y w,h 归一化'
        box_xy = (tf.sigmoid(predictions[...,:2])+ grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[...,2:4]) * anchors_tensor /tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy,box_wh,box_confidence,box_class_probs

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]

        input_shape = tf.cast(input_shape,tf.float32)
        image_shape = tf.cast(image_shape,tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
        offset = (input_shape - new_shape)/2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale
        box_left = box_yx - box_hw*0.5
        box_right = box_yx + box_hw*0.5

        box = tf.concat([box_left[...,0:1],box_left[...,1:2],box_right[...,0:1],box_right[...,1:2]],axis=-1)
        box *= tf.concat([image_shape,image_shape],axis=-1)
        return box

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats,anchors,classes_num,input_shape)
        boxes = self.correct_boxes(box_xy,box_wh,input_shape,image_shape)
        boxes = tf.reshape(boxes,[-1,4])

        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores,[-1,classes_num])
        return boxes,box_scores

    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        # inputshape是416x416
        # image_shape是实际图片的大小
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_


    def predict(self,inputs,image_shape):
        model = yolo3_model(config.norm_epsilon, config.norm_decay)
        outs = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, False)
        boxes,scores,classes = self.eval(outs,image_shape,max_boxes=20)
        return boxes,scores,classes
