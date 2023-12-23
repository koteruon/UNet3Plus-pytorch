import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from config.config import cfg

ORI_WEIGHT = {}


def weights_clone(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1 or classname.find("BatchNorm") != -1:
        ORI_WEIGHT[id(m.weight)] = m.weight.data.clone()
    else:
        raise Exception


def weight_clone(m):
    if isinstance(m, nn.Conv2d):
        weights_clone(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_clone(m)
    else:
        pass
