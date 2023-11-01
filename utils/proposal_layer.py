import random
import numpy as np
import tensorflow.compat.v1 as tf
from utils.anchors import calc_iou

tf.disable_v2_behavior()


class ProposalLayer:
    def __init__(self, anchors, img_feature_map_size, config):
        self.anchors = anchors
        self.img_feature_map_size = img_feature_map_size
        self.config = config

        # 针对非极大值抑制专门声明个session
        self.placeholder_box = tf.placeholder(dtype='float32', shape=(None, 4))
        self.placeholder_scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.placeholder_box, self.placeholder_scores, config.top_k, config.iou_threshold)
        self.predict_nms = tf.image.non_max_suppression(self.placeholder_box, self.placeholder_scores, config.top_k, config.predict_iou_threshold)
        self.session = tf.Session()

    # 根据offset针对anchors进行偏移，得到框的位置
    def _decode_offset(self, offset):
        # 先验框的宽高
        anchor_wh = self.anchors[:, 2:4] - self.anchors[:, :2]

        # 先验框的中心点
        anchor_centers = (self.anchors[:, 2:4] + self.anchors[:, :2]) / 2

        # 先验框中心点进行偏移dxdy
        new_anchor_centers = anchor_centers + offset[:, :2] * anchor_wh / 4

        # 先验框新的宽高
        new_anchor_wh = np.exp(offset[:, 2:4] / 4) * anchor_wh

        # 先验框偏移后的左上角右下角坐标
        new_anchors = np.zeros((len(self.anchors), 4))
        new_anchors[:, :2] = new_anchor_centers - new_anchor_wh / 2
        new_anchors[:, 2:4] = new_anchor_centers + new_anchor_wh / 2

        new_anchors = np.clip(new_anchors, 0, 1)
        return new_anchors

    def _rpn_decode(self, rpn_pred):
        pred_class = rpn_pred[0]  # (1, all_num_anchors, 1)
        pred_offset = rpn_pred[1]  # (1, all_num_anchors, 4)

        pred_class = pred_class[0]  # (all_num_anchors, 1)
        pred_offset = pred_offset[0]  # (all_num_anchors, 4)

        pred_box = self._decode_offset(pred_offset)  # (all_num_anchors, 4)

        index_to_process = pred_class[:, 0] > self.config.rpn_decode_confidence_threshold

        score_to_process = pred_class[index_to_process, 0]
        box_to_process = pred_box[index_to_process, :]

        feed_dict = {
            self.placeholder_box: box_to_process,
            self.placeholder_scores: score_to_process
        }
        ids = self.session.run(self.nms, feed_dict=feed_dict)

        boxes = box_to_process[ids]

        return boxes

    # 根据rpn的结果，解析成roi的输入
    def __call__(self, rpn_result, boxes=None):
        # rpn_result[0]形状: (1, all_num_anchors, 1)
        # rpn_result[1]形状: (1, all_num_anchors, 4)

        # 解码，根据positive概率超过confidence_threshold的部分，进行非极大值抑制，得到top_k个先验框
        # rpn_pred_decode形状：(top_k, 4)，4是左上角坐标、右下角坐标
        rpn_pred_decode = self._rpn_decode(rpn_result)

        # 拿到top_k个框后，将其尺度变到feature map同尺度上，并设置数据的标签值
        # 同时在feature map尺度上根据iou阈值进一步区分
        top_k_boxes = np.zeros(rpn_pred_decode.shape)
        top_k_boxes[:, [0, 2]] = rpn_pred_decode[:, [0, 2]] * self.img_feature_map_size[1]
        top_k_boxes[:, [1, 3]] = rpn_pred_decode[:, [1, 3]] * self.img_feature_map_size[0]
        top_k_boxes = np.around(top_k_boxes).astype(int)

        # boxes is None表示现在是测试阶段，无需生成标签数据，直接返回rpn预测的框的位置
        if boxes is None:
            top_k_boxes[:, 2:] = top_k_boxes[:, 2:] - top_k_boxes[:, :2]
            return top_k_boxes

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.img_feature_map_size[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.img_feature_map_size[0]
        boxes = np.around(boxes).astype(int)

        # -------------------根据iou进一步区分------------------------

        # shape: (len(boxes), top_k)
        box_anchor_iou = np.apply_along_axis(calc_iou, 1, boxes, top_k_boxes)

        max_iou = box_anchor_iou.max(axis=0)
        max_iou_index = box_anchor_iou.argmax(axis=0)

        background_index = (max_iou <= self.config.classifier_max_overlap) & (max_iou >= self.config.classifier_min_overlap)
        foreground_index = max_iou > self.config.classifier_max_overlap

        background_num = len(max_iou[background_index])
        foreground_num = len(max_iou[foreground_index])

        # -------------------生成标签数据------------------------

        X2 = top_k_boxes
        Y_class = np.zeros((len(X2), self.config.nb_class))
        Y_regress_label = np.zeros((len(X2), 4 * (self.config.nb_class - 1)))
        Y_regress_offset = np.zeros((len(X2), 4 * (self.config.nb_class - 1)))

        # 处理背景部分，即分到第21类
        if background_num:
            Y_class[background_index, -1] = 1

        # 处理前景部分
        if foreground_num:
            boxes_coord = boxes[max_iou_index][foreground_index, :4]  # (foreground_num, 4)
            box_class = boxes[max_iou_index][foreground_index, -1]  # (foreground_num,)

            Y_class[foreground_index, box_class] = 1

            Y_regress_label[foreground_index, 4 * box_class] = 1
            Y_regress_label[foreground_index, 4 * box_class + 1] = 1
            Y_regress_label[foreground_index, 4 * box_class + 2] = 1
            Y_regress_label[foreground_index, 4 * box_class + 3] = 1

            foreground_X = X2[foreground_index, :]
            foreground_X_centers = (foreground_X[:, 2:] + foreground_X[:, :2]) / 2  # (foreground_num, 2)
            foreground_X_wh = foreground_X[:, 2:] - foreground_X[:, :2]  # (foreground_num, 2)

            boxes_centers = (boxes_coord[:, 2:] + boxes_coord[:, :2]) / 2  # (foreground_num, 2)
            boxes_wh = boxes_coord[:, 2:] - boxes_coord[:, :2]  # (foreground_num, 2)

            dxdy = (boxes_centers - foreground_X_centers) / foreground_X_wh
            dwdh = np.log(boxes_wh / foreground_X_wh)

            classifier_regr_std = self.config.classifier_regr_std
            Y_regress_offset[foreground_index, 4 * box_class] = classifier_regr_std[0] * dxdy[:, 0]
            Y_regress_offset[foreground_index, 4 * box_class + 1] = classifier_regr_std[1] * dxdy[:, 1]
            Y_regress_offset[foreground_index, 4 * box_class + 2] = classifier_regr_std[2] * dwdh[:, 0]
            Y_regress_offset[foreground_index, 4 * box_class + 3] = classifier_regr_std[3] * dwdh[:, 1]

        # 拼成结果
        X2[:, 2:] = X2[:, 2:] - X2[:, :2]  # 转成x1,y1,w,h (RoiPoolingConv那里用的是x1,y1,w,h)
        X2 = np.expand_dims(X2, 0)
        Y_class = np.expand_dims(Y_class, 0)
        Y_regress = np.concatenate([Y_regress_label, Y_regress_offset], axis=1)
        Y_regress = np.expand_dims(Y_regress, 0)

        # -------------------进一步挑选出num_rois个区域------------------------

        if foreground_num + background_num < self.config.num_rois:
            return [], []

        select_pos_num = min(self.config.num_rois // 2, foreground_num)
        select_pos_num = max(select_pos_num, self.config.num_rois - background_num)
        select_neg_num = self.config.num_rois - select_pos_num

        # print()
        # print('foreground_num:', foreground_num, 'background_num:', background_num)
        # print('select_pos_num:', select_pos_num, 'select_neg_num:', select_neg_num)

        selected_pos_sample_index = np.where(foreground_index)[0]
        if foreground_num > select_pos_num:
            random_index = random.sample(range(foreground_num), select_pos_num)
            selected_pos_sample_index = selected_pos_sample_index[random_index]

        selected_neg_sample_index = np.where(background_index)[0]
        if background_num > select_neg_num:
            random_index = random.sample(range(background_num), select_neg_num)
            selected_neg_sample_index = selected_neg_sample_index[random_index]

        sel_samples = selected_pos_sample_index.tolist() + selected_neg_sample_index.tolist()
        np.random.shuffle(sel_samples)

        X2 = X2[:, sel_samples, :]
        Y_class = Y_class[:, sel_samples, :]
        Y_regress = Y_regress[:, sel_samples, :]

        return X2, [Y_class, Y_regress]

    # 预测的时候，对最终结构针对每一类进行nms
    def predict_nms_out(self, classifier_result):
        labels = classifier_result[:, 0]
        confidences = classifier_result[:, 1]
        boxes = classifier_result[:, 2:]

        result = []
        for i in range(self.config.nb_class - 1):
            mask = labels == i
            if len(labels[mask]) > 0:
                confidences_c = confidences[mask]
                boxes_c = boxes[mask, :]

                feed_dict = {
                    self.placeholder_box: boxes_c,
                    self.placeholder_scores: confidences_c
                }
                ids = self.session.run(self.predict_nms, feed_dict=feed_dict)

                labels_c = i * np.ones((len(ids), 1))
                confidences_good = confidences_c[ids][:, None]
                boxes_good = boxes_c[ids, :]

                c_out = np.concatenate([labels_c, confidences_good, boxes_good], axis=1)
                result.extend(c_out)
        return np.array(result)
