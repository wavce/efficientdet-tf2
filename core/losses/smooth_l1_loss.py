import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Huber):
    def __init__(self, delta=1.0, weight=1., reduction=tf.keras.losses.Reduction.NONE):
        super(SmoothL1Loss, self).__init__(delta=delta,
                                           reduction=reduction,
                                           name="SmoothL1Loss")
        self.weight = weight

    def call(self, y_true, y_pred):
        loss = super(SmoothL1Loss, self).call(y_true, y_pred)

        return loss * self.weight
