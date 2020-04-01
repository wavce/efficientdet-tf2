import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.distribute import distribution_strategy_context


class LookaheadOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5, name=None, **kwargs):
        super(LookaheadOptimizer, self).__init__(name=name, **kwargs)

        self.k = tf.constant(k, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self._optimizer = optimizer

        self._iterations = self._optimizer.iterations
        self.slow_weights = []

        self.add_slow_weights = True

        self.replica_context = tf.distribute.get_replica_context()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "slow")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        pass

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        pass

    def _update_weights(self, fast_weights, slow_weights, alpha):
        def _update_slow_weight(slow_weight, fast_weight, a):
            slow_weight.assign_add(a * (fast_weight - slow_weight))

        def _update_fast_weight(fast_weight, slow_weight):
            fast_weight.assign(slow_weight)

        if tf.equal(tf.cast(self._iterations, tf.float32) % self.k, 0):
            if distribution_strategy_context.has_strategy():
                distribution = distribution_strategy_context.get_replica_context()

                for fast, slow in zip(fast_weights, slow_weights):
                    distribution.extended.call_for_each_replica(_update_slow_weight,
                                                                args=(slow, fast.value(), alpha))
                    distribution.extended.call_for_each_replica(_update_fast_weight,
                                                                args=(fast, slow.value()))
            else:
                for fast, slow in zip(fast_weights, slow_weights):
                    _update_slow_weight(slow, fast.value(), alpha)
                    _update_fast_weight(fast, slow.value())

    def apply_gradients(self, grads_and_vars, name=None):
        fast_weights = [v for _, v in grads_and_vars]
        if self.add_slow_weights:
            self.slow_weights = [
                tf.Variable(initial_value=w.value(),
                            trainable=False,
                            name=w.name.split(":")[0] + "/slow")
                for w in fast_weights
            ]
            self.add_slow_weights = False

        res = self._optimizer.apply_gradients(grads_and_vars, name=name)

        self._update_weights(fast_weights, self.slow_weights, self.alpha)

        return res

    def get_config(self):
        config = self._optimizer.get_config()

        return config

    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._optimizer.learning_rate = value

    @property
    def lr(self):
        return self._optimizer.lr

    @lr.setter
    def lr(self, lr):
        self._optimizer.lr = lr

    def get_weights(self):
        return self._optimizer.get_weights()

    def set_weights(self, weights):
        return self._optimizer.set_weights(weights)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    def get_slot_names(self):
        return self._optimizer.get_slot_names()

    def variables(self):
        return self._optimizer.variables()

    @property
    def weights(self):
        return self._optimizer.weights

