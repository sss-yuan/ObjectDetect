
class Config:

    def __init__(self):
        # self.anchor_box_scales = [128, 256, 384]
        self.anchor_box_scales = [96, 192, 384]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        self.roi_pool_size = 7
        self.verbose = True
        self.base_conv_trainable = True
        self.model_path = "logs"
        self.init_weights_path = "logs/init_weights.h5"
        self.lr = 1e-5

        # rpn训练数据标签准备中，iou小于rpn_min_overlap表示背景
        self.rpn_min_overlap = 0.2
        # rpn训练数据标签准备中，iou大于rpn_max_overlap表示前景
        self.rpn_max_overlap = 0.5

        # rpn预测结构解析中，分类得分高于rpn_decode_confidence_threshold的部分进行非极大值抑制，取出top_k个
        self.rpn_decode_confidence_threshold = 0.0
        self.top_k = 300
        self.iou_threshold = 0.6    # 非极大值抑制重叠框判断的iou阈值
        self.predict_iou_threshold = 0.4    # 非极大值抑制重叠框判断的iou阈值
        self.predict_confidence_threshold = 0.7

        # classifier训练数据标签准备，top_k中，
        # iou > classifier_max_overlap作为分类正样本，并参与二次边框回归loss计算
        # iou >= classifier_min_overlap & iou <= classifier_max_overlap作为背景，分为第21类
        # iou < classifier_min_overlap则忽略
        # 最终取出num_rois个作为roi_inputs
        """
        为什么不直接 iou<classifier_max_overlap作为负样本呢？因为这样可以避免背景框是一条线(即长宽至少有一个为0)，
        如果背景为一条线（这在feature map尺度上是很有可能的），那在roi pool层中tf.image.resize_images的时候，会报错(提示input image must be of non-zero size)
        所以设置一个iou下限，这样背景框中就不会有一条线的情况
        """
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.num_rois = 24
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.input_shape = (512, 512, 3)
        self.nb_class = 11  # voc2007是20类，这里取其中10类，然后加1类无目标，classifier需输出共11类
