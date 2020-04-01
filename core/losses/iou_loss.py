import math
import tensorflow as tf


class IoULoss(tf.keras.losses.Loss):
    def __init__(self,
                 eps=1e-6,
                 reduction=tf.losses.Reduction.NONE,
                 weight=1.,
                 return_iou=False,
                 name="IoULoss"):
        super(IoULoss, self).__init__(reduction=reduction, name=name)

        self.eps = eps
        self.weight = weight
        self.return_iou = return_iou

    def _iou_loss(self, y_true, y_pred, eps):
        """IoU loss.
            Computing the IoU loss between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
            Args:
                y_pred (Tensor): Predicted bboxes of format (y1, x1, y2, x2),
                    shape (n, 4).
                y_true (Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Eps to avoid log(0).
            Returns:
                Tensor: Loss tensor.
        """
        ty1, tx1, ty2, tx2 = tf.unstack(y_true, 4, -1)
        py1, px1, py2, px2 = tf.unstack(y_pred, 4, -1)

        # intersection
        iy1 = tf.math.maximum(ty1, py1)
        ix1 = tf.math.maximum(tx1, px1)
        iy2 = tf.math.minimum(ty2, py2)
        ix2 = tf.math.minimum(tx2, px2)

        ih = tf.math.maximum(iy2 - iy1, 0.)
        iw = tf.math.maximum(ix2 - ix1, 0.)
        i_area = ih * iw

        t_area = (ty2 - ty1) * (tx2 - tx1)
        p_area = (py2 - py1) * (px2 - px1)

        iou = tf.math.divide(i_area, t_area + p_area - i_area)

        loss = -tf.math.log(iou + eps)
        # loss = 1. - iou

        weighted_loss = loss * self.weight
        
        if self.return_iou:
            return weighted_loss, iou
        
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._iou_loss(y_true, y_pred, self.eps)


class BoundedIoULoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.2, eps=1e-6, weight=1., reduction=tf.losses.Reduction.NONE, name="BoundedIoULoss"):
        super(BoundedIoULoss, self).__init__(reduction=reduction, name=name)

        self.beta = beta
        self.eps = eps
        self.weight = weight

    def _bounded_iou_loss(self, y_true, y_pred, beta=0.2, eps=1e-3):
        """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
           https://arxiv.org/abs/1711.00164.
           Args:
               y_pred (tensor): Predicted bboxes.
               y_true (tensor): Target bboxes.
               beta (float): beta parameter in smoothl1.
               eps (float): eps to avoid NaN.
        """
        py_ctr = (y_pred[:, 0] + y_pred[:, 2]) * 0.5
        px_ctr = (y_pred[:, 2] + y_pred[:, 3]) * 0.5
        ph = y_pred[:, 2] - y_pred[:, 0]
        pw = y_pred[:, 3] - y_pred[:, 1]

        ty_ctr = tf.stop_gradient((y_true[:, 0] + y_true[:, 2]) * 0.5)
        tx_ctr = tf.stop_gradient((y_true[:, 1] + y_true[:, 3]) * 0.5)
        th = tf.stop_gradient(y_true[:, 2] - y_true[:, 0])
        tw = tf.stop_gradient(y_true[:, 3] - y_true[:, 1])

        dx = ty_ctr - py_ctr
        dy = tx_ctr - px_ctr

        dx_loss = 1. - tf.maximum(
            (tw - 2. * tf.abs(dx)) / (tw + 2. * tf.abs(dx) + eps), 0.)
        dy_loss = 1. - tf.maximum(
            (th - 2. * tf.abs(dy)) / (th + 2. * tf.abs(dy) + eps), 0.)
        dw_loss = 1. - tf.minimum(tw / (pw + eps), pw / (tw + eps))
        dh_loss = 1. - tf.minimum(th / (ph + eps), ph / (th + eps))

        loss_comb = tf.stack([dy_loss, dx_loss, dh_loss, dw_loss], axis=-1)

        loss = tf.where(tf.less(loss_comb, beta),
                        0.5 * loss_comb * loss_comb / beta,
                        loss_comb - 0.5 * beta)

        return loss * self.weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._bounded_iou_loss(y_true, y_pred, self.beta, self.eps)


class GIoULoss(tf.keras.losses.Loss):
    def __init__(self, eps=1e-6, weight=10., return_iou=False, reduction=tf.losses.Reduction.NONE, name="GIoULoss"):
        super(GIoULoss, self).__init__(reduction, name)

        self.eps = eps
        self.weight = weight
        self.return_iou = return_iou

    def _giou_loss(self, y_true, y_pred, eps):
        """IoU loss.
            Computing the IoU loss between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
            Args:
                y_pred (Tensor): Predicted bboxes of format (y1, x1, y2, x2),
                    shape (n, 4).
                y_true (Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Eps to avoid log(0).
            Returns:
                Tensor: Loss tensor.
        """
        ty1, tx1, ty2, tx2 = tf.unstack(y_true, 4, -1)
        py1, px1, py2, px2 = tf.unstack(y_pred, 4, -1)

        # py1 = tf.math.minimum(py1, py2)
        # px1 = tf.math.minimum(px1, px2)
        # py2 = tf.math.maximum(py1, py2)
        # px2 = tf.math.maximum(px1, px2)

        # intersection
        iy1 = tf.math.maximum(ty1, py1)
        ix1 = tf.math.maximum(tx1, px1)
        iy2 = tf.math.minimum(ty2, py2)
        ix2 = tf.math.minimum(tx2, px2)

        ih = tf.math.maximum(iy2 - iy1, 0.)
        iw = tf.math.maximum(ix2 - ix1, 0.)
        i_area = ih * iw

        t_area = (ty2 - ty1) * (tx2 - tx1)
        p_area = (py2 - py1) * (px2 - px1)

        # Union
        u_area = t_area + p_area - i_area
        
        # Enclosing box
        ey1 = tf.math.minimum(ty1, py1)
        ex1 = tf.math.minimum(tx1, px1)
        ey2 = tf.math.maximum(ty2, py2)
        ex2 = tf.math.maximum(tx2, px2)
        e_area = (ey2 - ey1) * (ex2 - ex1)

        iou = tf.math.divide(i_area, u_area + eps)

        giou = iou - tf.math.divide(e_area - u_area,  e_area)
        loss = 1. - giou

        # if tf.reduce_any(tf.math.is_nan(giou)):
        #     tf.print("giou nan y_true", tf.boolean_mask(y_true, tf.math.is_nan(giou)))
        #     tf.print("giou nan y_pred", tf.boolean_mask(y_pred, tf.math.is_nan(giou)))
        #     tf.print("e_area", tf.boolean_mask(e_area, tf.math.is_nan(giou)))
        #     tf.print("u_area", tf.boolean_mask(u_area, tf.math.is_nan(u_area)))

        weighted_loss = loss * self.weight
        
        if self.return_iou:
            return weighted_loss, iou
        
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._giou_loss(y_true, y_pred, self.eps)


class DIoULoss(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-5, weight=12., return_iou=False, reduction=tf.losses.Reduction.NONE, name="DIoULoss"):
        super(DIoULoss, self).__init__(reduction=reduction, name=name)

        self.epsilon = epsilon
        self.weight = weight

        self.return_iou = return_iou

    def _diou_loss(self, y_true, y_pred):
        ty1, tx1, ty2, tx2 = tf.unstack(y_true, 4, -1)
        py1, px1, py2, px2 = tf.unstack(y_pred, 4, -1)

        # intersection
        iy1 = tf.math.maximum(ty1, py1)
        ix1 = tf.math.maximum(tx1, px1)
        iy2 = tf.math.minimum(ty2, py2)
        ix2 = tf.math.minimum(tx2, px2)

        ih = tf.math.maximum(iy2 - iy1, 0.)
        iw = tf.math.maximum(ix2 - ix1, 0.)
        i_area = ih * iw

        t_area = (ty2 - ty1) * (tx2 - tx1)
        p_area = (py2 - py1) * (px2 - px1)

        iou = tf.math.divide(i_area, t_area + p_area - i_area)

        # The digonal distance is diagnoal of the smallest enclosed boxes
        ey1 = tf.math.minimum(ty1, py1)
        ex1 = tf.math.minimum(tx1, px1)
        ey2 = tf.math.maximum(ty2, py2)
        ex2 = tf.math.maximum(tx2, px2)
        
        diagonal_dist = (ey2 - ey1) ** 2. + (ex2 - ex1) ** 2.

        # center points:
        t_ctr_y = (ty1 + ty2) * 0.5
        t_ctr_x = (tx1 + tx2) * 0.5
        p_ctr_y = (py1 + py2) * 0.5
        p_ctr_x = (px1 + px2) * 0.5
        center_point_dist = (t_ctr_y - p_ctr_y) ** 2. + (t_ctr_x - p_ctr_x) ** 2.

        loss = 1. - iou + center_point_dist / diagonal_dist

        weighted_loss = loss * self.weight
        
        if self.return_iou:
            return iou, weighted_loss
        
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._diou_loss(y_true, y_pred)


class CIoULoss(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-5, weight=12., return_iou=False, reduction=tf.losses.Reduction.NONE, name="DIoULoss"):
        super(CIoULoss, self).__init__(reduction=reduction, name=name)

        self.epsilon = epsilon
        self.weight = weight
        self.return_iou = return_iou

    def _ciou_loss(self, y_true, y_pred):
        ty1, tx1, ty2, tx2 = tf.unstack(y_true, 4, -1)
        py1, px1, py2, px2 = tf.unstack(y_pred, 4, -1)

        # intersection
        iy1 = tf.math.maximum(ty1, py1)
        ix1 = tf.math.maximum(tx1, px1)
        iy2 = tf.math.minimum(ty2, py2)
        ix2 = tf.math.minimum(tx2, px2)

        ih = tf.math.maximum(iy2 - iy1, 0.)
        iw = tf.math.maximum(ix2 - ix1, 0.)
        i_area = ih * iw

        t_area = (ty2 - ty1) * (tx2 - tx1)
        p_area = (py2 - py1) * (px2 - px1)

        iou = tf.math.divide(i_area, t_area + p_area - i_area)

        # The digonal distance is diagnoal of the smallest enclosed boxes
        ey1 = tf.math.minimum(ty1, py1)
        ex1 = tf.math.minimum(tx1, px1)
        ey2 = tf.math.maximum(ty2, py2)
        ex2 = tf.math.maximum(tx2, px2)
        
        diagonal_dist = (ey2 - ey1) ** 2. + (ex2 - ex1) ** 2.

        # center points:
        t_ctr_y = (ty1 + ty2) * 0.5
        t_ctr_x = (tx1 + tx2) * 0.5
        p_ctr_y = (py1 + py2) * 0.5
        p_ctr_x = (px1 + px2) * 0.5
        center_point_dist = (t_ctr_y - p_ctr_y) ** 2. + (t_ctr_x - p_ctr_x) ** 2.

        # aspect ratios:
        t_h = ty2 - ty1
        t_w = tx2 - tx1
        p_h = py2 - py1
        p_w = px2 - px1

        arctan = tf.stop_gradient(tf.math.atan(t_w / t_h) - tf.math.atan(p_w / p_h))
        v = 4. / (math.pi ** 2.) * tf.pow(tf.math.atan(t_w / t_h) - tf.math.atan(p_w / p_h), 2.)
        v = tf.stop_gradient(v)
        s = tf.stop_gradient(1. - iou)
        alpha = tf.stop_gradient(v / (s + v))
        w_temp = tf.stop_gradient(2. * p_w)

        ar = (8. / (math.pi ** 2)) * arctan * ((p_w - w_temp) * p_h)   # 1 / (w**2 + h**2) replace by 1
        loss = 1. - iou + center_point_dist / diagonal_dist + ar * alpha

        weighted_loss = loss * self.weight
        
        if self.return_iou:
            return iou, weighted_loss
        
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._ciou_loss(y_true, y_pred)

