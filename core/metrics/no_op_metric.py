import tensorflow as tf 


class NoOpMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(NoOpMetric, self).__init__(**kwargs)

        self.value = self.add_weight(name="no_op_value", initializer="zeros")
    
    def update_state(self, value, sample_weight=None):
        self.value.assign(tf.cast(value, self.value.dtype))

    def result(self):
        return self.value
