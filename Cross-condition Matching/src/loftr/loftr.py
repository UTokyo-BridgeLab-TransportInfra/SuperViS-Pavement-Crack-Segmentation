import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
# from .utils.fine_matching import FineMatching
# from .loftr_module import FinePreprocess


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ======= config & modules ======= #
        self.config = config

        # backbone（你的 build_backbone 可以返回 Tensor / (Tensor, Tensor) / list 等）
        self.backbone = build_backbone(config)

        # 仅保留 coarse 分支
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix']
        )
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])

        # self.fine_preprocess = FinePreprocess(config)
        # self.loftr_fine = LocalFeatureTransformer(config["fine"])
        # self.fine_matching = FineMatching()

    @staticmethod
    def _normalize_out(outs):
        """
        把 backbone 的输出统一成 (coarse_tensor, fine_tensor_or_None) 的形式。
        - 直接是 Tensor：视为 coarse，fine=None
        - list/tuple:
            * len==2 -> (coarse, fine)
            * len==1 -> (outs[0], None)
          若子元素还是 list/tuple，则取最后一个（例如某些返回多尺度）。
        """
        if torch.is_tensor(outs):
            return outs, None

        if isinstance(outs, (list, tuple)):
            if len(outs) == 2:
                c, f = outs
                if isinstance(c, (list, tuple)):
                    c = c[-1]
                if isinstance(f, (list, tuple)):
                    f = f[-1]
                return c, f
            elif len(outs) == 1:
                o = outs[0]
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                return o, None

        raise TypeError(f"Unsupported backbone output type: {type(outs)} with len={getattr(outs,'__len__',lambda: 'NA')()}")

    def forward(self, data):
        """
        Args in data:
            image0: (N, 1 or 3, H0, W0)
            image1: (N, 1 or 3, H1, W1)
            mask0/mask1 (optional): (N, H, W)  0 表示 padding
        """
        # ======= 1) Backbone 特征 ======= #
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:],  # (H0, W0)
            'hw1_i': data['image1'].shape[2:]   # (H1, W1)
        })
        bs = data['bs']

        if data['hw0_i'] == data['hw1_i']:
            # 同尺寸：cat 一次性前向，利于 BN 收敛 & 更快
            x = torch.cat([data['image0'], data['image1']], dim=0)  # [2*bs, C, H, W]
            outs = self.backbone(x)
            feats_c, _ = self._normalize_out(outs)

            # 断言便于发现错配
            assert isinstance(feats_c, torch.Tensor), f"feats_c must be Tensor, got {type(feats_c)}"
            assert feats_c.shape[0] == 2 * bs, f"feats_c batch={feats_c.shape[0]} != 2*bs={2*bs}"

            # 在 batch 维切分
            feat_c0, feat_c1 = torch.split(feats_c, bs, dim=0)
        else:
            # 不同尺寸：分别前向
            outs0 = self.backbone(data['image0'])
            outs1 = self.backbone(data['image1'])
            feat_c0, _ = self._normalize_out(outs0)
            feat_c1, _ = self._normalize_out(outs1)

            # 断言类型
            assert isinstance(feat_c0, torch.Tensor) and isinstance(feat_c1, torch.Tensor), \
                f"Backbone must return Tensor or (Tensor,Tensor). Got {type(feat_c0)} / {type(feat_c1)}"

        data.update({
            'hw0_c': feat_c0.shape[2:],  # (Hc0, Wc0)
            'hw1_c': feat_c1.shape[2:]   # (Hc1, Wc1)
        })

        # ======= 2) Coarse LoFTR ======= #
        # 加位置编码 -> 拉平成 [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        # 训练时可带 mask（会在 self.loftr_coarse 内部做 attention masking）
        mask_c0 = mask_c1 = None
        if 'mask0' in data and 'mask1' in data:
            mask_c0 = data['mask0'].flatten(-2)  # [N, H*W]
            mask_c1 = data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # ======= 3) Coarse Matching ======= #
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 本实现保持与原版一致：结果写回到 data（如 mkpts0_f/mkpts1_f/mconf 等）
        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        # 兼容旧权重中以 'matcher.' 为前缀的键
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
