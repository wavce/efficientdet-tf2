from .max_iou_assigner import MaxIoUAssigner
from .mask_max_iou_assigner import MaskMaxIoUAssigner
from .scale_compensation_max_iou_assigner import ScaleCompensationMaxIoUAssigner
from .scale_compensation_mask_max_iou_assigner import ScaleCompensationMaskMaxIoUAssigner
from .fcos_assigner import FCOSAssigner
from .atss_assigner import ATSSAssigner


ASSIGNERS = {
    "atss_assigner": ATSSAssigner,
    "fcos_assigner": FCOSAssigner,
    "max_iou_assigner": MaxIoUAssigner,
    "mask_max_iou_assigner": MaskMaxIoUAssigner,
    "scale_compensation_max_iou_assigner": ScaleCompensationMaxIoUAssigner,
    "scale_compensation_mask_max_iou_assigner": ScaleCompensationMaskMaxIoUAssigner
}


def build_assigner(assigner, **kwargs):
    return ASSIGNERS[assigner](**kwargs)


