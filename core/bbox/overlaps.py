import tensorflow as tf


def aligned_box_iou(boxes1, boxes2):
    """Calculate overlap between two set of aligned boxes.
        'aligned' mean len(boxes1) == len(boxes2).
    Args:
        boxes1 (tensor): shape (n, 4).
        boxes2 (tensor): shape (n, 4),

    Returns:
        ious (Tensor): shape (m)
    """
    lt = tf.maximum(boxes1[:, 0:2], boxes2[:, 0:2])  # [m, 2]
    rb = tf.minimum(boxes1[:, 2:4], boxes2[:, 2:4])  # [m, 2]

    wh = tf.maximum(0.0, rb - lt)  # [m, 2]
    overlap = tf.reduce_prod(wh, axis=1)  # [m]
    area1 = tf.reduce_prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis=1)  # [m]
    area2 = tf.reduce_prod(boxes2[:, 2:4] - boxes2[:, 0:2], axis=1)

    ious = overlap / (area1 + area2 - overlap)

    return ious


def unaligned_box_iou(boxes1, boxes2):
    """Calculate overlap between two set of unaligned boxes.
        'unaligned' mean len(boxes1) != len(boxes2).

        Args:
            boxes1 (tensor): shape (n, 4).
            boxes2 (tensor): shape (m, 4), m not equal n.

        Returns:
            ious (Tensor): shape (m, n)
    """
    boxes1 = boxes1[:, None, :]   # (n, 1, 4)
    boxes2 = boxes2[None, :, :]   # (1, m, 4)
    lt = tf.maximum(boxes1[..., 0:2], boxes2[..., 0:2])  # (n, m, 2)
    rb = tf.minimum(boxes1[..., 2:4], boxes2[..., 2:4])  # (n, m, 2)

    wh = tf.maximum(0.0, rb - lt)  # (n, m, 2)
    overlap = tf.reduce_prod(wh, axis=2)  # (n, m)
    area1 = tf.reduce_prod(boxes1[..., 2:4] - boxes1[..., 0:2], axis=2)  # (n, m)
    area2 = tf.reduce_prod(boxes2[..., 2:4] - boxes2[..., 0:2], axis=2)

    ious = overlap / (area1 + area2 - overlap)

    return ious


def aligned_box_iof(boxes1, boxes2):
    """Calculate the overlap between two set of aligned boxes over
    boxes1, 'aligned' mean len(boxes1) == len(boxes2).

    Args:
        boxes1 (tensor): shape (n, 4).
        boxes2 (tensor): shape (n, 4),

    Returns:
        ious (Tensor): shape (m)
    """
    lt = tf.maximum(boxes1[:, 0:2], boxes2[:, 0:2])  # [m, 2]
    rb = tf.minimum(boxes1[:, 2:4], boxes2[:, 2:4])  # [m, 2]

    wh = tf.maximum(0.0, rb - lt)  # [m, 2]
    overlap = tf.reduce_prod(wh, axis=1)  # [m]
    area1 = tf.reduce_prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis=1)  # [m]

    ious = overlap / area1

    return ious


def unaligned_box_iof(boxes1, boxes2):
    """Calculate overlap between two set of unaligned boxes over
    boxes1, 'unaligned' mean len(boxes1) != len(boxes2).

        Args:
            boxes1 (tensor): shape (n, 4).
            boxes2 (tensor): shape (m, 4), m not equal n.

        Returns:
            ious (Tensor): shape (m, n)
    """
    boxes1 = tf.expand_dims(boxes1, 1)  # (n, 1, 4)
    boxes2 = tf.expand_dims(boxes2, 0)  # (1, m, 4)
    lt = tf.maximum(boxes1[..., 0:2], boxes2[..., 0:2])  # (n, m, 2)
    rb = tf.minimum(boxes1[..., 2:4], boxes2[..., 2:4])  # (n, m, 2)

    wh = tf.maximum(0.0, rb - lt)  # (n, m, 2)
    overlap = tf.reduce_prod(wh, axis=2)  # (n, m)
    area1 = tf.reduce_prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis=2)  # (n, m)

    ious = overlap / area1

    return ious


def compute_quadrilateral_area(quadrilateral):
    """Compute the quadrilateral area.

        Args:
            quadrilateral(Tensor): [?, 8] -> ((y1, x1), (y2, x2), (y3, x3), (y4, x4)),
                quadrilateral should not normalize to [0, 1].
    """
    y1, x1, y2, x2, y3, x3, y4, x4 = tf.split(quadrilateral, 8, 1)
    slope1 = tf.sets
