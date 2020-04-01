import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=True,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 label_smoothing=0.0,
                 reduction=tf.keras.losses.Reduction.NONE,
                 weight=1.,
                 name="FocalLoss"):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)

        assert use_sigmoid, "Only support sigmoid."
        self.use_sigmoid = use_sigmoid

        assert from_logits, "Only support logits."
        self.from_logits = from_logits

        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def _focal_loss(self, y_true, y_pred, gamma, alpha, label_smoothing, from_logits):
        smooth_y_true = tf.cond(tf.greater(label_smoothing, 0),
                                lambda: y_true * (1. - label_smoothing) +
                                        label_smoothing / (tf.cast(tf.shape(y_true)[-1], y_true.dtype) - 1.),
                                lambda: y_true)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_y_true,
                                                                logits=y_pred)       

        # Below are comments/derivations for computing modulator.
        # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
        # for positive samples and 1 - sigmoid(x) for negative examples.
        #
        # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
        # computation. For r > 0, it puts more weights on hard examples, and less
        # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
        # its back-propagation is not stable when r < 1. The implementation here
        # resolves the issue.
        #
        # For positive samples (labels being 1),
        #    (1 - p_t)^r
        #  = (1 - sigmoid(x))^r
        #  = (1 - (1 / (1 + exp(-x))))^r
        #  = (exp(-x) / (1 + exp(-x)))^r
        #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
        #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
        #  = exp(- r * x - r * log(1 + exp(-x)))
        #
        # For negative samples (labels being 0),
        #    (1 - p_t)^r
        #  = (sigmoid(x))^r
        #  = (1 / (1 + exp(-x)))^r
        #  = exp(log((1 / (1 + exp(-x)))^r))
        #  = exp(-r * log(1 + exp(-x)))
        #
        # Therefore one unified form for positive (z = 1) and negative (z = 0)
        # samples is:
        #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
        negative_logits = -1.0 * y_pred
        modulator = tf.math.exp(gamma * y_true * negative_logits -
                                gamma * tf.math.log1p(tf.math.exp(negative_logits)))
        loss = modulator * cross_entropy

        positive_label_mask = tf.equal(y_true, 1.0)
        weighted_loss = tf.where(positive_label_mask, alpha * loss, (1. - alpha) * loss)
        weighted_loss *= self.weight

        # positive_label_mask = tf.equal(y_true, 1.0)
        # probs = tf.sigmoid(y_pred)
        # probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
        # # With small gamma, the implementation could produce NaN during back prop.
        # modulator = tf.pow(1.0 - probs_gt, gamma)
        # loss = modulator * cross_entropy
        # weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss) * self.weight

        return weighted_loss

    def call(self, y_true, y_pred):
        return self._focal_loss(y_true,
                                y_pred,
                                self.gamma,
                                self.alpha,
                                self.label_smoothing,
                                self.from_logits)
