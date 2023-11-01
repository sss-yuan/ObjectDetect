import numpy as np
import math


def get_img_feature_map_size(config):
    target_size = config.input_shape[:2]
    height, weight = target_size
    rpn_stride = config.rpn_stride
    while rpn_stride > 1:
        height = int(math.ceil(height / 2.0))
        weight = int(math.ceil(weight / 2.0))
        rpn_stride = rpn_stride / 2
    return height, weight


# 返回anchor相对其中心点的坐标
def _base_anchors(config):
    anchor_box_scales = config.anchor_box_scales
    anchor_box_ratios = config.anchor_box_ratios

    num = len(anchor_box_scales) * len(anchor_box_ratios)

    anchors = np.zeros((num, 4))

    scales = np.tile(anchor_box_scales, (2, len(anchor_box_ratios))).T
    ratios = np.repeat(np.array(anchor_box_ratios), len(anchor_box_scales), axis=0)

    anchors[:, 2:] = scales * ratios
    anchors = anchors / 2
    anchors[:, :2] = -anchors[:, 2:]

    return anchors


def get_anchors(img_feature_map_size, config):
    # anchor相对其中心点的坐标
    base_anchors = _base_anchors(config)

    # 中心点的坐标
    centers_x = (np.arange(img_feature_map_size[0]) + 0.5) * config.rpn_stride
    centers_y = (np.arange(img_feature_map_size[1]) + 0.5) * config.rpn_stride

    centers_x, centers_y = np.meshgrid(centers_x, centers_y)

    centers_x = np.reshape(centers_x, [-1])
    centers_y = np.reshape(centers_y, [-1])

    # 此处centers shape：(32 * 32, 4)，32x32是feature map的尺寸
    centers = np.stack([centers_x, centers_y, centers_x, centers_y], axis=0).T

    centers = np.expand_dims(centers, 0).transpose((1, 0, 2))  # shape：(1024, 1, 4)
    base_anchors = np.expand_dims(base_anchors, 0)  # shape：(1, 9, 4)

    anchors = centers + base_anchors  # shape：(1024, 9, 4)
    anchors = np.reshape(anchors, (-1, 4))  # shape：(1024 * 9, 4)

    target_size = config.input_shape[:2]
    height, weight = target_size

    anchors[:, [0, 2]] = anchors[:, [0, 2]] / weight
    anchors[:, [1, 3]] = anchors[:, [1, 3]] / height
    anchors = np.clip(anchors, 0, 1)

    return anchors


# 针对某个box，与所有anchors的iou，输出shape: (num_anchors, 1)
def calc_iou(box, anchors):
    inter_up_left = np.maximum(anchors[:, :2], box[:2])
    inter_bottom_right = np.minimum(anchors[:, 2:4], box[2:4])

    inter_wh = inter_bottom_right - inter_up_left
    inter_wh = np.maximum(0, inter_wh)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    union_area = box_area + anchors_area - inter_area

    iou = inter_area / union_area
    return iou


def _calc_offset(anchors, boxes):
    assert len(anchors) == len(boxes)

    anchor_centers = (anchors[:, :2] + anchors[:, 2:4]) / 2
    anchor_wh = anchors[:, 2:4] - anchors[:, :2]

    box_centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    box_wh = boxes[:, 2:4] - boxes[:, :2]

    offset = np.zeros((len(anchors), 4))
    offset[:, :2] = 4 * (box_centers - anchor_centers) / anchor_wh
    offset[:, 2:] = 4 * np.log(box_wh / anchor_wh)

    return offset


# 得到每个anchor应有的预测结果
def assign_boxes(boxes, anchors, config):

    assignment = np.zeros((len(anchors), 5))

    # shape: (len(boxes), len(anchors))
    box_anchor_iou = np.apply_along_axis(calc_iou, 1, boxes, anchors)

    # shape: (len(anchors),)
    max_iou = box_anchor_iou.max(axis=0)
    max_iou_index = box_anchor_iou.argmax(axis=0)

    # iou在rpn_min_overlap、rpn_max_overlap之间的，忽略掉，不参与loss计算
    ignore_index = (max_iou > config.rpn_min_overlap) & (max_iou < config.rpn_max_overlap)
    assignment[ignore_index, 4] = -1

    # iou小于rpn_min_overlap的，判断为背景，分类为0，无需处理
    # iou大于rpn_max_overlap的，判断为前景，分类为1，需要找到重合度最大的先验框，并计算dxdydwdh
    foreground_index = max_iou > config.rpn_max_overlap
    foreground_box_index = max_iou_index[foreground_index]

    offsets = _calc_offset(anchors[foreground_index], boxes[foreground_box_index])

    assignment[foreground_index, :4] = offsets
    assignment[foreground_index, 4] = 1

    return assignment
