from .fpn import fpn
from .bifpn import bifpn
from .nas_fpn import nas_fpn


NECKS = {
    "fpn": fpn,
    "bifpn": bifpn,
    "nas_fpn": nas_fpn,
}


def build_neck(neck, **kwargs):
    return NECKS[neck](**kwargs)


__all__ = [
    "build_neck",
]
