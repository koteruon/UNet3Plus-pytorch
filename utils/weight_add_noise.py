import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from config.config import cfg


def weights_add_noise(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1 or classname.find("BatchNorm") != -1:
        # 計算每個通道的最大值
        channel_max_values = (
            cfg.model.n_tr * torch.max(torch.abs(m.weight.view(*m.weight.size()[:-2], -1)), dim=-1).values
        )

        # 計算每個元素的標準差
        sigma_delta_W_tr = channel_max_values.view(m.weight.size()[:-2] + (1,) * (len(m.weight.size()) - 2)).expand(
            m.weight.size()
        )

        # 生成正態分佈的隨機數
        delta_Gij_l = torch.normal(mean=0.0, std=sigma_delta_W_tr)

        # 將超過閾值的值截斷
        delta_Gij_l.clamp_(-cfg.model.alpha * sigma_delta_W_tr, cfg.model.alpha * sigma_delta_W_tr)

        # 將權重加上隨機數
        m.weight.data += delta_Gij_l
    else:
        raise Exception


def weight_add_noise(m):
    if isinstance(m, nn.Conv2d):
        weights_add_noise(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_add_noise(m)
    else:
        pass
