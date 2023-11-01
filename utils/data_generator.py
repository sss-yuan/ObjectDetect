import math
import random

import numpy as np
from PIL import Image, ImageEnhance
from keras.applications.imagenet_utils import preprocess_input
from utils.anchors import assign_boxes


def _rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a


def _get_random_data(line, target_size, jitter=0.1, bright=0.3, color=0.5, contrast=0.1, sharp=0.1):
    line = line.split()
    img = Image.open(line[0])
    boxes = np.array([list(map(int, e.split(","))) for e in line[1:]])

    if len(boxes) == 0:
        return [], []

    iw, ih = img.size
    w, h = target_size

    # jitter，抖动
    new_ar = w / h * _rand(1 - jitter, 1 + jitter)  # 长宽比抖动
    scale = _rand(.9, 1.1)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    img = img.resize((nw, nh), Image.BICUBIC)

    dx = int(_rand(0, w - nw))
    dy = int(_rand(0, h - nh))
    new_img = Image.new("RGB", (w, h), (128, 128, 128))
    new_img.paste(img, (dx, dy))
    img = new_img

    # jitter，抖动后框的位置也需要跟着变
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy

    # flip, 左右翻转
    if _rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]     # 注意，翻转后左上角不再是左上角，新的左上角x坐标是w减去原先的右下角x坐标

    boxes[:, 0:2][boxes[:, :2] < 0] = 0
    boxes[:, 2][boxes[:, 2] > w] = w
    boxes[:, 3][boxes[:, 3] > h] = h

    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]     # 只要宽高大于1的box

    np.random.shuffle(boxes)
    if len(boxes) == 0:
        return np.array(img), []

    # 调整亮度
    enh_bri = ImageEnhance.Brightness(img)
    img = enh_bri.enhance(factor=_rand(1 - bright, 1 + bright))

    # 调整图像的色彩平衡
    enh_color = ImageEnhance.Color(img)
    img = enh_color.enhance(factor=_rand(1 - color, 1 + color))

    # 调整图像的对比度
    enh_contrast = ImageEnhance.Contrast(img)
    img = enh_contrast.enhance(factor=_rand(1 - contrast, 1 + contrast))

    # 调整图像的锐化程度
    enh_sharp = ImageEnhance.Sharpness(img)
    img = enh_sharp.enhance(factor=_rand(1 - sharp, 1 + sharp))

    # # 测试看效果的代码
    # from PIL import ImageDraw
    # imageDraw = ImageDraw.Draw(img)
    # for box in boxes:
    #     x1, y1, x2, y2 = box[:4]
    #     imageDraw.rectangle([x1, y1, x2, y2], outline='red')
    # img.show()

    return np.array(img), boxes


def data_generator(lines, anchors, config):
    target_size = config.input_shape[:2]
    height, weight = target_size
    while True:
        np.random.shuffle(lines)
        for line in lines:
            # 图像增强
            img, boxes = _get_random_data(line, target_size)
            if len(boxes) == 0:
                continue

            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / weight
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / height

            # 根据先验框和真实框，计算每个先验框应该有的预测结果
            # shape: (all_anchors_num, 5) 5表示dxdydwdh4个值 + positive/negative分类
            assignment = assign_boxes(boxes, anchors, config)

            classification = assignment[:, 4]
            regress = assignment[:, :]

            # 背景的样本量一般远远大于前景，这里限定下正负样本量
            num_regions = 128

            num_pos = len(classification[classification > 0])
            num_neg = len(classification[classification == 0])

            if num_pos == 0:
                # 下面将负样本量和正样本量一样了，这样如果正样本量为0，那负样本量也为0，分类结果全部忽略，计算loss的时候会出现nan
                continue

            if num_pos > num_regions:
                index_pos = np.where(classification > 0)[0]
                random_index = random.sample(range(num_pos), int(num_pos - num_regions))
                classification[index_pos[random_index]] = -1
                regress[index_pos[random_index], -1] = -1
                num_pos = num_regions

            if num_neg > num_pos:
                index_neg = np.where(classification == 0)[0]
                random_index = random.sample(range(num_neg), int(num_neg - num_pos))
                classification[index_neg[random_index]] = -1
                regress[index_neg[random_index], -1] = -1

            img = np.expand_dims(img, 0)
            classification = np.reshape(classification, (1, -1, 1))
            regress = np.reshape(regress, (1, -1, 5))

            # 返回值是rpn的训练数据及其标签，即图片数据，各个先验框是否有目标，各个先验框的dxdydwdh，以及anchors、boxes等（生成roi_inputs要用到）
            yield preprocess_input(img, mode="tf"), [classification, regress], boxes
