from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer


class AccumOptimizer(Optimizer):
    """继承Optimizer类，包装原有优化器，实现梯度累积。
       # 参数
           optimizer：优化器实例，支持目前所有的keras优化器；
           steps_per_update：累积的步数。
       # 返回
           一个新的keras优化器
       Inheriting Optimizer class, wrapping the original optimizer
       to achieve a new corresponding optimizer of gradient accumulation.
       # Arguments
           optimizer: an instance of keras optimizer (supporting
                       all keras optimizers currently available);
           steps_per_update: the steps of gradient accumulation
       # Returns
           a new keras optimizer.
       """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(AccumOptimizer, self).__init__(**kwargs)

        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.steps_per_update = steps_per_update
            self.iterations = K.variable(0, "int64", "iteration")
            self.cond = K.equal(self.iterations % steps_per_update, 0)
            self.lr = self.optimizer.lr

            self.accum_grads = None

            self.optimizer.lr = K.switch(self.cond, self.lr, 0)
            for attr in ["momentum", "rho", "beta_1", "beta_2"]:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, 1. - 1e-7)

            for cfg in self.optimizer.get_config():
                if not hasattr(self, cfg):
                    value = getattr(self.optimizer, cfg)
                    setattr(self, cfg, value)

            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss, params):
                return [ag / self.steps_per_update for ag in self.accum_grads]

            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):
        self.updates = [
            K.update_add(self.iterations, 1),
            K.update_add(self.optimizer.iterations, K.constant(self.cond, "int64"))
        ]

        # accumulate gradients
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        grads = self.get_gradients(loss, params)
        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(K.update(ag, K.switch(self.cond, ag * 0, ag + g)))

        self.updates.extend(self.optimizer.get_updates()[1:])
        self.weights.extend(self.optimizer.weights)

        return self.updates

    def get_config(self):
        iterations = K.eval(self.iterations)
        K.set_value(self.iterations, 0)
        config = self.optimizer.get_config()
        K.set_value(self.iterations, iterations)

        return config

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.optimizer.learning_rate = value

