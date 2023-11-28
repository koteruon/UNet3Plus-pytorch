import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from config.config import cfg


def weights_add_noise(m):
    torch.cuda.manual_seed_all(cfg.model.noise_seed)
    torch.manual_seed(cfg.model.noise_seed)
    np.random.seed(cfg.model.noise_seed)
    random.seed(cfg.model.noise_seed)

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        sigma_delta_W_tr = cfg.model.n_tr * m.weight.abs().max()
        delta_Gij_l = torch.randn_like(m.weight) * sigma_delta_W_tr
        m.weight.data = m.weight.data + delta_Gij_l
    elif classname.find("Linear") != -1:
        sigma_delta_W_tr = cfg.model.n_tr * m.weight.abs().max()
        delta_Gij_l = torch.randn_like(m.weight) * sigma_delta_W_tr
        m.weight.data = m.weight.data + delta_Gij_l
    elif classname.find("BatchNorm") != -1:
        sigma_delta_W_tr = cfg.model.n_tr * m.weight.abs().max()
        delta_Gij_l = torch.randn_like(m.weight) * sigma_delta_W_tr
        m.weight.data = m.weight.data + delta_Gij_l

    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)


def weight_add_noise(m):
    if isinstance(m, nn.Conv2d):
        weights_add_noise(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_add_noise(m)
    else:
        pass
