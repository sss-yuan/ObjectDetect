import cv2
import tensorflow as tf
import keras
import numpy as np

from nets.FasterRCNN import FasterRCNN
from config.config import Config


# fasterRCNN = FasterRCNN(Config())
# # 加载权重
# fasterRCNN.model_all.load_weights('./logs/final_weights.h5')
# tf.saved_model.save(fasterRCNN.model_all, "pack/")

# converter = tf.lite.TFLiteConverter.from_saved_model("pack/") # path to the SavedModel directory
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# tf.compat.v1.disable_eager_execution()

from utils.anchors import get_img_feature_map_size, get_anchors
config = Config()
# 经过base_layer的卷积后，得到的feature_map尺寸
img_feature_map_size = get_img_feature_map_size(config)
# 获取最开始所有anchor的位置：左上角右下角坐标，shape:(all_anchors_num, 4)，该坐标经过了归一化
anchors = get_anchors(img_feature_map_size, config)
anchors_num = len(anchors)

fasterRCNN = FasterRCNN(Config())
fasterRCNN.model_all.load_weights('./logs/final_weights.h5')

inputs = keras.Input(shape=(512, 512, 3))
rpn_class, rpn_regress = fasterRCNN.model_rpn(inputs)
anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

rpn_class = rpn_class[0][:, 0]
rpn_regress = rpn_regress[0]

anchor_wh = tf.subtract(anchors[:, 2:4], anchors[:, :2])
anchor_centers = tf.add(anchors[:, 2:4], anchors[:, :2])
anchor_centers = tf.divide(anchor_centers, 2)
regress_center = tf.divide(tf.multiply(rpn_regress[:, :2], anchor_wh), 4)
new_anchor_centers = tf.add(anchor_centers, regress_center)

regress_wh = tf.exp(tf.divide(rpn_regress[:, 2:4], 4))
new_anchor_wh = tf.multiply(regress_wh, anchor_wh)

new_anchors = tf.concat([
    tf.subtract(new_anchor_centers, tf.divide(new_anchor_wh, 2)),
    tf.add(new_anchor_centers, tf.divide(new_anchor_wh, 2))
], axis=1)
new_anchors = tf.clip_by_value(new_anchors, 0, 1)

nms_ids = tf.image.non_max_suppression(new_anchors, rpn_class, config.top_k, config.predict_iou_threshold)

boxes = tf.gather(new_anchors, nms_ids)
boxes = tf.multiply(boxes, img_feature_map_size[0])
boxes = tf.cast(boxes, dtype=tf.int32)
boxes = tf.concat([
    boxes[:, :2],
    tf.subtract(boxes[:, 2:], boxes[:, :2])
], axis=1)

boxes_indices = tf.where(tf.logical_and(tf.greater(boxes[:, 2], 0), tf.greater(boxes[:, 3], 0)))
boxes_indices = boxes_indices[:, 0]
boxes = tf.gather(boxes, boxes_indices)
boxes = boxes[:config.num_rois, :]
boxes = tf.expand_dims(boxes, axis=0)

output_class, output_regress = fasterRCNN.model_classifier([inputs, boxes])
output_class = output_class[0]
output_regress = output_regress[0]
output_class_index = tf.argmax(output_class[:, :config.nb_class-1], axis=1)
output_confidence = tf.reduce_max(output_class[:, :config.nb_class-1], axis=1)
output_indices = tf.where(tf.not_equal(output_class_index, config.nb_class - 1))

# coord = tf.convert_to_tensor(np.zeros((config.num_rois, 4)), dtype=tf.int32)
coord = []
for i in range(config.num_rois):
    j = output_class_index[i]
    j = tf.cast(j, tf.int32)
    # j = tf.minimum(j, config.nb_class - 2)      # 这里预测为无目标时，边框回归没有数据，j为10会超出上限，所以变为9（只是为了能计算，无目标后面会过滤掉）
    cols = output_regress[i, 4 * j:4 * (j + 1)]
    x1 = tf.cast(boxes[0, i, 0], tf.float32)
    y1 = tf.cast(boxes[0, i, 1], tf.float32)
    w = tf.cast(boxes[0, i, 2], tf.float32)
    h = tf.cast(boxes[0, i, 3], tf.float32)
    old_center_x = tf.add(x1, tf.divide(w, 2))
    old_center_y = tf.add(y1, tf.divide(h, 2))
    center_x = tf.add(old_center_x, tf.multiply(w, tf.divide(cols[0], config.classifier_regr_std[0])))
    center_y = tf.add(old_center_y, tf.multiply(h, tf.divide(cols[1], config.classifier_regr_std[1])))
    w1 = tf.multiply(w, tf.exp(tf.divide(cols[2], config.classifier_regr_std[2])))
    h1 = tf.multiply(h, tf.exp(tf.divide(cols[3], config.classifier_regr_std[3])))
    coord_x1 = tf.subtract(center_x, tf.divide(w1, 2))
    coord_y1 = tf.subtract(center_y, tf.divide(h1, 2))
    coord_x2 = tf.add(center_x, tf.divide(w1, 2))
    coord_y2 = tf.add(center_y, tf.divide(h1, 2))
    cols = tf.stack([
        # 官方salad.tflite解析出的坐标似乎是y1x1y2x2
        tf.multiply(coord_y1, config.rpn_stride),
        tf.multiply(coord_x1, config.rpn_stride),
        tf.multiply(coord_y2, config.rpn_stride),
        tf.multiply(coord_x2, config.rpn_stride),
    ])
    cols = tf.divide(cols, config.input_shape[0])
    # coord = tf.tensor_scatter_nd_update(coord, [i], cols)
    coord.append(cols)


output_coords = tf.stack(coord, axis=0)

# output_coords = tf.gather_nd(output_coords, output_indices)
output_coords = tf.expand_dims(output_coords, axis=0)
# output_class_index = tf.gather(output_class_index, output_indices)
# output_confidence = tf.gather(output_confidence, output_indices)
output_num_detections = tf.expand_dims(tf.shape(output_coords)[1], axis=0)
output_num_detections = tf.cast(output_num_detections, tf.float32)      # 转成float，保持和官方的salad.tflite返回类型一样，不然安卓端detector.detect(image)里面调用c++报错

output_class_index = tf.expand_dims(tf.cast(output_class_index, tf.float32), axis=0)
output_confidence = tf.expand_dims(output_confidence, axis=0)

output_coords = tf.identity(output_coords, name="location")
output_class_index = tf.identity(output_class_index, name="category")
output_confidence = tf.identity(output_confidence, name="score")
output_num_detections = tf.identity(output_num_detections, name="number of detections")

model = keras.Model(inputs=inputs, outputs=[output_coords, output_class_index, output_confidence, output_num_detections])
model.summary()

# model2 = keras.Model(inputs=inputs, outputs=[output_class_index, output_regress, boxes])
# input_data = cv2.imread('street.jpg')
# input_data = cv2.resize(input_data, (512, 512))
# input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
# input_data = np.array(input_data, dtype=np.float32)
# input_data /= 127.5
# input_data -= 1
# input_data = np.expand_dims(input_data, axis=0)
# out = model.predict(input_data)
# print(out)

# 第一步 saved_model
tf.saved_model.save(model, "pack/")

