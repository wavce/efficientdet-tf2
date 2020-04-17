import tensorflow as tf


class Box2Delta(object):
    def __init__(self, mean=(0., 0., 0., 0.), std=(0.1, 0.1, 0.2, 0.2), epsilon=1e-5, dtype=tf.float32):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.dtype = dtype

    def __call__(self, proposals, boxes):
        proposals = tf.cast(proposals, self.dtype)
        boxes = tf.cast(boxes, self.dtype)
        byx = (boxes[..., 0:2] + boxes[..., 2:4]) * 0.5
        bhw = boxes[..., 2:4] - boxes[..., 0:2]

        pyx = (proposals[..., 0:2] + proposals[..., 2:4]) * 0.5
        phw = proposals[..., 2:4] - proposals[..., 0:2]

        dyx = (byx - pyx) / (phw + self.epsilon)
        dhw = tf.math.log(bhw / phw + self.epsilon)

        delta = tf.concat([dyx, dhw], axis=-1)

        if self.mean is not None and self.std is not None:
            mean = tf.convert_to_tensor(self.mean, dtype=boxes.dtype)
            std = tf.convert_to_tensor(self.std, dtype=boxes.dtype)
            delta = (delta - mean) / std

        return delta


class Delta2Box(object):
    def __init__(self, mean=(0., 0., 0., 0.), std=(0.1, 0.1, 0.2, 0.2)):
        self.mean = mean
        self.std = std

    def __call__(self, proposals, delta):
        pyx = (proposals[..., 0:2] + proposals[..., 2:4]) * 0.5
        phw = proposals[..., 2:4] - proposals[..., 0:2]

        if self.mean is not None and self.std is not None:
            mean = tf.convert_to_tensor(self.mean, dtype=delta.dtype)
            std = tf.convert_to_tensor(self.std, dtype=delta.dtype)
            delta = delta * std + mean

        byx = delta[..., 0:2] * phw + pyx
        bhw = tf.math.exp(delta[..., 2:4]) * phw

        boxes = tf.concat([byx - bhw * 0.5, byx + bhw * 0.5], axis=-1)

        return boxes


class Distance2Box(object):
    def __call__(self, distances, grid_y, grid_x):
        # grid_y = tf.expand_dims(grid_y, 0)
        # grid_x = tf.expand_dims(grid_x, 0)

        boxes = tf.stack([grid_y - distances[..., 0],
                          grid_x - distances[..., 1],
                          grid_y + distances[..., 2],
                          grid_x + distances[..., 3]], axis=-1)
        
        return boxes
