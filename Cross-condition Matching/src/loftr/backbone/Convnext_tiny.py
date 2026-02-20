import torch.nn as nn
import torch.nn.functional as F
import torch
import timm



class Convnext_tiny(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        model_name  = cfg.get('backbone_name', 'convnext_tiny')
        pretrained  = cfg.get('pretrained', True)
        in_chans    = cfg.get('in_chans', 1)        
        out_indices = tuple(cfg.get('out_indices', (3,)))

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        # x: [B, in_chans, H, W]
        c5 = self.backbone(x)[0]      # 1/32
        return c5


