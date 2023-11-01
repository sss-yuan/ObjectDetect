import keras
import tensorflow as tf
import keras.backend as K
from keras import Input
from keras.layers import TimeDistributed, Flatten, Dense, AveragePooling2D, Conv2D, Reshape

from nets.RoiPoolingConv import RoiPoolingConv


def _get_rpn_class_loss():
    """
       y_pred: (batch_size, all_anchor_num, 1)
       y_true: (batch_size, all_anchor_num, 1)
       假如featureMap的宽高是32x32，则all_anchor_num = 32 * 32 * 9，9是featureMap上每个点对应原图每个区域的anchor_num
       所以这些先验框很多，大部分是negative的先验框，如果全部参与loss计算会对训练不利
       因此，在生成训练数据的时候，需要标识哪些先验框参与loss计算，比如用-1表示不参与loss计算，类似于torch的ignored_index参数
       这样，就可以使正负样本相对均衡进行loss的计算
    """

    def _rpn_class_loss(y_true, y_pred, ignored_index=-1):
        y_true = y_true[0, :, :]
        y_pred = y_pred[0, :, :]
        all_losses = K.binary_crossentropy(y_true, y_pred)
        actual_loss_indices = tf.where(K.not_equal(y_true, ignored_index))
        actual_loss = tf.gather_nd(all_losses, actual_loss_indices)
        return K.mean(actual_loss)

    return _rpn_class_loss


def _get_rpn_regress_loss():
    """
        边框回归只取正样本，即positive的先验框进行loss的计算
    """

    def _smooth_l1(y_true, y_pred, sigma=1.0):
        sigma_squared = sigma ** 2

        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1


def classifier_cls_loss(y_true, y_pred):
    """
        分类器是针对每个roi进行多分类，因此相当于是一个roi一个样本，交叉熵的最终输入形状是(num_rois, nb_class)
    """
    return K.mean(K.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def _get_classifier_regress_loss(nb_class):
    """
         模型输出的是每个类别的回归量dxdydwdh，但只有正确类别下对应的那部分dxdydwdh才参与loss的计算
     """
    epsilon = 1e-4
    def classifier_regress_loss(y_true, y_pred):
        x = y_true[:, :, 4 * nb_class:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(y_true[:, :, :4 * nb_class] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4 * nb_class])
        return loss

    return classifier_regress_loss


class FasterRCNN:
    def __init__(self, config):
        self.config = config

        self.base_conv_layers = keras.models.load_model('model_weights/base_mobilenet_weights.h5')

        self.inputs = Input(shape=self.config.input_shape)
        self.roi_inputs = Input((None, 4))

        # rpn输出原图上所有anchor的positive概率及bbox，output_shapes：(all_anchor_num, 1)、(all_anchor_num, 4)
        self.model_rpn = self._get_rpn()  # Region Proposal Networks
        self.model_classifier = self._get_classifier()

        self.model_rpn.compile(loss={
            'rpn_classification': _get_rpn_class_loss(),
            'rpn_regress': _get_rpn_regress_loss()
        }, optimizer=keras.optimizers.adam_v2.Adam(learning_rate=self.config.lr))

        self.model_classifier.compile(loss={
            "classifier_out_class": classifier_cls_loss,
            "classifier_out_regress": _get_classifier_regress_loss(self.config.nb_class - 1),
        },  metrics={"classifier_out_class": "accuracy"},
            optimizer=keras.optimizers.adam_v2.Adam(learning_rate=self.config.lr))

        self.model_all = keras.Model(
            [self.inputs, self.roi_inputs],
            self.model_rpn.outputs + self.model_classifier.outputs
        )

    def _get_rpn(self):
        inputs = self.inputs
        base_model_output = self.base_conv_layers(inputs)
        anchor_num = len(self.config.anchor_box_scales) * len(self.config.anchor_box_ratios)

        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv1')(base_model_output)

        # output_shape: (batch_size, M/16, N/16, anchor_num)、(batch_size, M/16, N/16, 4 * anchor_num)
        x_class = Conv2D(anchor_num, (1, 1), activation='sigmoid', name='rpn_out_class')(x)
        x_regress = Conv2D(4 * anchor_num, (1, 1), activation='linear', name='rpn_out_regress')(x)

        # reshape拍平一下，即形状变成是(原图上所有的anchor数, 1)，方便算loss
        x_class = Reshape((-1, 1), name='rpn_classification')(x_class)
        x_regress = Reshape((-1, 4), name='rpn_regress')(x_regress)
        return keras.Model(inputs, [x_class, x_regress])

    def _get_classifier(self):
        inputs = self.inputs
        roi_inputs = self.roi_inputs
        base_model_output = self.base_conv_layers(inputs)

        # 接收featureMap和roi_inputs，在featureMap尺度上针对各个roi_input的框进行pooling
        out_roi_pooling = RoiPoolingConv(self.config.roi_pool_size, self.config.num_rois)([base_model_output, roi_inputs])

        # 使用TimeDistributed包一层，这样第二个维度就不会被影响到，output_shape：(batch_size, num_rois, pool_size, pool_size, 512)
        output = TimeDistributed(Conv2D(512, (3, 3), padding='same'), name='classifier_conv1')(out_roi_pooling)
        # output_shape：(batch_size, num_rois, 1, 1, 512)
        output = TimeDistributed(AveragePooling2D((self.config.roi_pool_size, self.config.roi_pool_size)), name='classifier_avg_pool')(output)
        # output_shape: (batch_size, num_rois, 512)，如果没有TimeDistributed应该输出(batch_size, num_rois*512)
        output = TimeDistributed(Flatten())(output)

        output = Dense(512, activation='relu', name='classifier_dense1')(output)
        output = Dense(512, activation='relu', name='classifier_dense2')(output)

        output_class = Dense(self.config.nb_class, activation='softmax', name='classifier_out_class')(output)
        output_regress = Dense(4 * (self.config.nb_class - 1), activation='linear', name='classifier_out_regress')(output)

        return keras.Model([inputs, roi_inputs], [output_class, output_regress])

    def update_lr(self, new_lr1, new_lr2):
        self.model_rpn.compile(loss={
            'rpn_classification': _get_rpn_class_loss(),
            'rpn_regress': _get_rpn_regress_loss()
        }, optimizer=keras.optimizers.adam_v2.Adam(learning_rate=new_lr1))

        self.model_classifier.compile(loss={
            "classifier_out_class": classifier_cls_loss,
            "classifier_out_regress": _get_classifier_regress_loss(self.config.nb_class - 1),
        }, metrics={"classifier_out_class": "accuracy"},
            optimizer=keras.optimizers.adam_v2.Adam(learning_rate=new_lr2))
        print("Learning rate update：rpn: {}, classifier: {}".format(str(new_lr1), str(new_lr2)))
