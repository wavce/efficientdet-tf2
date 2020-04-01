from .box_label_heads import prediction_head

from .head import Head


from .retinanet_head import RetinaNetHead


ALLS = {
    "RetinaNetHead": RetinaNetHead
}


def build_head(cfg):
    return ALLS[cfg.head.head](cfg)


__all__ = ["Head", "prediction_head", "build_head"]
