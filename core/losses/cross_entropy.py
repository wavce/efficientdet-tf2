import tensorflow as tf


class BinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=True,
                 label_smoothing=0.0,
                 weight=1.,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name="BinaryCrossEntropy"):
        super(BinaryCrossEntropy, self).__init__(reduction=reduction, name=name)

        assert from_logits
        self.weight = weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        smooth_y_true = tf.cond(tf.greater(self.label_smoothing, 0),
                                lambda: y_true * (1. - self.label_smoothing) +
                                        self.label_smoothing / (tf.cast(tf.shape(y_true)[-1], y_true.dtype) - 1.),
                                lambda: y_true)

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_y_true, 
                                                       logits=y_pred) * self.weight


class CrossEntropy(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=True,
                 label_smoothing=0.01,
                 weight=1.,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name="CrossEntropy"):
        super(CrossEntropy, self).__init__(reduction=reduction, name=name)

        self.weight = weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        smooth_y_true = tf.cond(tf.greater(self.label_smoothing, 0),
                                lambda: (y_true * (1. - self.label_smoothing) +
                                         self.label_smoothing / (tf.cast(
                                            tf.shape(y_true)[-1], y_true.dtype) - 1.)),
                                lambda: y_true)

        return tf.nn.softmax_cross_entropy_with_logits(labels=smooth_y_true, 
                                                       logits=y_pred) * self.weight
