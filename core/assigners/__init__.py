from .max_iou_assigner import MaxIoUAssigner


ASSIGNERS = {"max_iou_assigner": MaxIoUAssigner}


def build_assigner(assigner, **kwargs):
    return ASSIGNERS[assigner](**kwargs)


