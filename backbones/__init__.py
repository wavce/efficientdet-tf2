from .resnet_v1 import ResNetV1
from .resnet_v2 import ResNetV2
from .efficientnet import EfficientNet


def build_backbone(backbone_name, **kwargs):
    if backbone_name == "resnet50":
        return ResNetV1(backbone_name, **kwargs)
    elif backbone_name == "resnet101":
        return ResNetV1(backbone_name, **kwargs)
    elif backbone_name == "resnet152":
        return ResNetV1(backbone_name, **kwargs)
    elif backbone_name == "resnet_v2_50":
        return ResNetV2(backbone_name, **kwargs)
    elif backbone_name == "resnet_v2_101":
        return ResNetV2(backbone_name, **kwargs)
    elif backbone_name == "resnet_v2_152":
        return ResNetV2(backbone_name, **kwargs)
    elif "efficientnet" in backbone_name:
        return EfficientNet(backbone_name, **kwargs)
    else:
        raise ValueError("backbone_name[`%s`] error." % backbone_name)
