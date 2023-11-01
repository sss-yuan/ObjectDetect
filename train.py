from tensorflow import keras
import numpy as np
import time
import os

from nets.FasterRCNN import FasterRCNN
from utils.data_generator import data_generator
from utils.proposal_layer import ProposalLayer
from utils.anchors import get_img_feature_map_size, get_anchors
from config.config import Config

import tensorflow as tf

gpu_list = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpu_list[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)


if __name__ == '__main__':
    config = Config()

    fasterRCNN = FasterRCNN(config)

    model_rpn = fasterRCNN.model_rpn
    model_classifier = fasterRCNN.model_classifier
    model_all = fasterRCNN.model_all

    # 经过base_layer的卷积后，得到的feature_map尺寸
    img_feature_map_size = get_img_feature_map_size(config)
    # 获取最开始所有anchor的位置：左上角右下角坐标，shape:(all_anchors_num, 4)，该坐标经过了归一化
    anchors = get_anchors(img_feature_map_size, config)

    proposal_layer = ProposalLayer(anchors, img_feature_map_size, config)

    annotation_path = 'data/2007_trainval.txt'
    with open(annotation_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    np.random.seed(10001)
    np.random.shuffle(lines)
    np.random.seed(None)
    generator = data_generator(lines, anchors, config)

    EPOCH = 200                  # 训练epoch轮
    EPOCH_LENGTH = 2000         # 每轮epoch_length张图

    # 记录rpn_class_loss, rpn_regress_loss, classifier_class_loss, classifier_regress_loss, classifier_class的accuracy
    losses = np.zeros((EPOCH_LENGTH, 5))

    start_time = time.time()
    print('Starting training')
    if os.path.exists(config.init_weights_path):
        print("Load weights from " + config.init_weights_path)
        model_all.load_weights(config.init_weights_path)

    if not config.base_conv_trainable:
        print("冻结基础卷积层...")
        fasterRCNN.base_conv_layers.trainable = False
        fasterRCNN.update_lr(config.lr, config.lr)
    # fasterRCNN.update_lr(1e-6, 1e-5)

    fasterRCNN.model_all.summary()
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch + 1, EPOCH))
        progbar = keras.utils.Progbar(EPOCH_LENGTH)

        # if epoch == 120:
        #     fasterRCNN.update_lr(1e-6, 1e-6)

        for iter_num in range(EPOCH_LENGTH):

            X, Y, boxes = next(generator)

            rpn_loss = model_rpn.train_on_batch(X, Y)
            rpn_result = model_rpn.predict_on_batch(X)

            # rpn的结果经过proposal_layer得到roi_input，
            # 注意roi_input是在feature map尺度上的框坐标xywh，且没有经过归一化（在RoiPoolingConv层根据索引提取框，所以不能归一化）
            roi_input, Y2 = proposal_layer(rpn_result, boxes)
            if len(roi_input) == 0:
                continue

            classifier_loss = model_classifier.train_on_batch([X, roi_input], Y2)

            losses[iter_num, 0] = rpn_loss[1]
            losses[iter_num, 1] = rpn_loss[2]
            losses[iter_num, 2] = classifier_loss[1]
            losses[iter_num, 3] = classifier_loss[2]
            losses[iter_num, 4] = classifier_loss[3]

            progbar.update(iter_num, [
                ('rpn_cls', losses[iter_num, 0]),
                ('rpn_regr', losses[iter_num, 1]),
                ('classifi_cls', losses[iter_num, 2]),
                ('classifi_regr', losses[iter_num, 3]),
                ('total_loss', np.sum(losses[iter_num, :4]))
            ])

            if iter_num == EPOCH_LENGTH - 1:
                avg_rpn_cls = np.mean(losses[:, 0])
                avg_rpn_regr = np.mean(losses[:, 1])
                avg_classifi_cls = np.mean(losses[:, 2])
                avg_classifi_regr = np.mean(losses[:, 3])
                avg_classifi_acc = np.mean(losses[:, 4])

                if config.verbose:
                    print()
                    print('Avg Loss RPN classifier: {}'.format(avg_rpn_cls))
                    print('Avg Loss RPN regression: {}'.format(avg_rpn_regr))
                    print('Avg Loss Classifier classifier: {}'.format(avg_classifi_cls))
                    print('Avg Loss Classifier regression: {}'.format(avg_classifi_regr))
                    print('Avg Classifier accuracy for bounding boxes from RPN: {}'.format(avg_classifi_acc))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                start_time = time.time()

                total_avg_loss = avg_rpn_cls + avg_rpn_regr + avg_classifi_cls + avg_classifi_regr
                rpn_loss = avg_rpn_cls + avg_rpn_regr
                classifi_loss = avg_classifi_cls + avg_classifi_regr

                weigth_name = "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(epoch+1,total_avg_loss,rpn_loss,classifi_loss)+".h5"
                model_all.save_weights(config.model_path + weigth_name)
