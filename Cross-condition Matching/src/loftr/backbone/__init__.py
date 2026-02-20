from .Convnext_tiny import Convnext_tiny


def build_backbone(config):
    return Convnext_tiny(config)

