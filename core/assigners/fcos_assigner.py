import tensorflow as tf

INF = 1e10


class FCOSAssigner(object):
    def __init__(self, sampling_radius=.0, epsilon=1e-4, **kwargs):
        self.sampling_radius = sampling_radius
        self.epsilon = epsilon

    def get_sample_region(self, gt_boxes, grid_y, grid_x, strides):
        with tf.name_scope("get_sample_region"):
            gt_boxes = tf.tile(gt_boxes, [tf.shape(grid_x)[0], 1, 1])
            center_y = (gt_boxes[..., 0] + gt_boxes[..., 2]) * 0.5
            center_x = (gt_boxes[..., 1] + gt_boxes[..., 3]) * 0.5

            y_min = center_y - strides * self.sampling_radius
            x_min = center_x - strides * self.sampling_radius
            y_max = center_y + strides * self.sampling_radius
            x_max = center_x + strides * self.sampling_radius

            center_gt_y_min = tf.where(y_min > gt_boxes[..., 0], y_min, gt_boxes[..., 0])
            center_gt_x_min = tf.where(x_min > gt_boxes[..., 1], x_min, gt_boxes[..., 1])
            center_gt_y_max = tf.where(y_max > gt_boxes[..., 2], gt_boxes[..., 2], y_max)
            center_gt_x_max = tf.where(x_max > gt_boxes[..., 3], gt_boxes[..., 3], x_max)

            top = grid_y[:, None] - center_gt_y_min
            left = grid_x[:, None] - center_gt_x_min
            bottom = center_gt_y_max - grid_y[:, None]
            right = center_gt_x_max - grid_x[:, None]
            center_box = tf.stack([top, left, bottom, right], -1)

            return tf.greater(tf.reduce_min(center_box, -1), 0)

    def assign(self, gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest):
        with tf.name_scope("assigner"):
            h, w = tf.shape(grid_x)[0], tf.shape(grid_y)[1]
            num_grid = h * w   # h * w

            grid_y = tf.reshape(grid_y, [h * w])
            grid_x = tf.reshape(grid_x, [h * w])

            valid_mask = tf.logical_not(tf.reduce_all(gt_boxes == 0, 1))
            gt_boxes = tf.boolean_mask(gt_boxes, valid_mask)
            gt_boxes = tf.concat([tf.zeros([1, 4], gt_boxes.dtype), gt_boxes], 0)
            gt_labels = tf.concat([tf.zeros([1], gt_labels.dtype), gt_labels], 0)
            # normalizer = tf.cast([h * strides, w * strides, h * strides, w * strides], gt_boxes.dtype)
            # gt_boxes *= normalizer

            gt_boxes = tf.expand_dims(gt_boxes, 0)  # (1, n, 4)
            gt_areas = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])  # (1, n)


            avoiding_equal_areas = tf.random.uniform(tf.shape(gt_areas), 0, 0.001, dtype=gt_areas.dtype)
            gt_areas = tf.where(gt_areas == 0, gt_areas, gt_areas + avoiding_equal_areas)

            distances = tf.stack([grid_y[:, None] - gt_boxes[..., 0],
                                  grid_x[:, None] - gt_boxes[..., 1],
                                  gt_boxes[..., 2] - grid_y[:, None],
                                  gt_boxes[..., 3] - grid_x[:, None]], axis=2)  # (h * w, n, 4)

            # if self.sampling_radius > 0:
            #     in_box_mask = self.get_sample_region(gt_boxes, grid_y, grid_x)
            # else:
            in_box_mask = tf.greater_equal(tf.reduce_min(distances, 2), 0)  # (h * w, n)

            max_distances = tf.reduce_max(distances, 2)
            in_level_mask = tf.logical_and(tf.greater_equal(max_distances, object_size_of_interest[0]),
                                           tf.less(max_distances, object_size_of_interest[1]))  # (h * w, n)
            
            gt_areas = tf.tile(gt_areas, [num_grid, 1])  # (h * w, n)
            
            mask = tf.logical_and(in_box_mask, in_level_mask)
            gt_areas = tf.where(mask, gt_areas, tf.ones_like(gt_areas) * INF)

            min_gt_area_indices = tf.argmin(gt_areas, 1)
            indices = tf.stack([tf.cast(tf.range(num_grid), dtype=min_gt_area_indices.dtype), 
                                min_gt_area_indices], axis=1)
            target_distances = tf.gather_nd(distances * tf.cast(tf.expand_dims(mask, 2), tf.float32), indices)

            target_boxes = tf.gather_nd(tf.tile(gt_boxes, [num_grid, 1, 1]), indices)
            target_labels = tf.gather(gt_labels, min_gt_area_indices)

            target_centerness = self.compute_centerness(target_distances)

            target_boxes = tf.reshape(target_boxes, [h * w, 4])
             
            target_centerness = tf.reshape(target_centerness, [h * w, 1])

            return target_boxes, target_labels, target_centerness

    def compute_centerness(self, target_distances):
        with tf.name_scope("compute_centerness"):
            min_tb = tf.minimum(target_distances[..., 0], target_distances[..., 2])
            max_tb = tf.maximum(tf.maximum(target_distances[..., 0], target_distances[..., 2]), self.epsilon)
            min_lr = tf.minimum(target_distances[..., 1], target_distances[..., 3])
            max_lr = tf.maximum(tf.minimum(target_distances[..., 1], target_distances[..., 3]), self.epsilon)
            ctr = tf.math.sqrt((min_tb / max_tb) * (min_lr / max_lr))
            ctr = tf.clip_by_value(ctr, 0, 1)
            return ctr

    def __call__(self, gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest):
        return self.assign(gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest)
