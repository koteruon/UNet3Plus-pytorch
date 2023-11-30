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
    if classname.find("Conv") != -1 or classname.find("Linear") != -1 or classname.find("BatchNorm") != -1:
        channel_max_values = (
            cfg.model.n_tr * torch.max(torch.abs(m.weight.view(*m.weight.size()[:-2], -1)), dim=-1).values
        )
        expand_dims = m.weight.size()[:-2] + (1,) * (len(m.weight.size()) - 2)
        sigma_delta_W_tr = channel_max_values.view(expand_dims).expand(m.weight.size())
        delta_Gij_l = torch.normal(mean=0.0, std=sigma_delta_W_tr)
        m.weight.data = m.weight.data + delta_Gij_l
        with open("weight_track.csv", "a") as f:
            if channel_max_values.dim() == 0:
                list_as_string = channel_max_values.cpu().detach().numpy().tolist()
            elif channel_max_values.dim() == 1:
                list_as_string = ",".join(map(str, channel_max_values.cpu().detach().numpy().tolist()))
            elif channel_max_values.dim() == 2:
                list_as_string = ",".join(
                    [",".join(map(str, row)) for row in channel_max_values.cpu().detach().numpy().tolist()]
                )
            else:
                raise Exception
            f.write(f"{m.weight._cdata},{list_as_string}")
    else:
        raise Exception

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
