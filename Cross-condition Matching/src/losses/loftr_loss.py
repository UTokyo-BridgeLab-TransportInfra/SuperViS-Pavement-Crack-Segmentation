from loguru import logger
import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_config = config['loftr']['loss']

        # coarse-level
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.correct_thr = self.loss_config['fine_correct_thr']

    # --------------------------------------------------------
    # 32*32 pixel local windows loss
    # --------------------------------------------------------
    def conf_cross_entropy_loss(self, conf_matrix_gt, conf_matrix):
        conf_matrix_gt = conf_matrix_gt.squeeze(1)
        pos_mask = conf_matrix_gt == 1
        neg_mask = conf_matrix_gt == 0

        conf = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)
        loss_pos = -torch.log(conf[pos_mask])
        loss_neg = -torch.log(1 - conf[neg_mask])

        loss = (loss_pos.mean() if loss_pos.numel() > 0 else 0.0) + loss_neg.mean()
        return loss

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': scalar,
                'loss_scalars': dict
            }
        """
        loss_scalars = {}

        loss_c = self.conf_cross_entropy_loss(
            data['gtconf_matrix'],
            data['conf_matrix']
        )
        loss = loss_c * self.loss_config['coarse_weight']

        loss_scalars['loss_c'] = loss_c.detach().cpu()
        data.update({'loss': loss, 'loss_scalars': loss_scalars})
