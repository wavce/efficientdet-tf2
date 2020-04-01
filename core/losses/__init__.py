from .focal_loss import FocalLoss
from .smooth_l1_loss import SmoothL1Loss
from .regularization import regularization_loss
from .cross_entropy import CrossEntropy, BinaryCrossEntropy
from .iou_loss import IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss 

LOSS_DICT = {
    "focal_loss": FocalLoss,
    "smooth_l1_loss": SmoothL1Loss,
    "cross_entropy": CrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "iou_loss": IoULoss,
    "bounded_iou_loss": BoundedIoULoss,
    "giou_loss": GIoULoss,
    "diou_loss": DIoULoss,
    "ciou_loss": CIoULoss
}


def build_loss(**kwargs):
    loss = kwargs.pop("loss")
    return LOSS_DICT[loss](**kwargs)


__all__ = ["build_loss"]