import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from config.config import cfg
from utils.weight_clone import ORI_WEIGHT


def weights_recover(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1 or classname.find("BatchNorm") != -1:
        m.weight.data = ORI_WEIGHT[id(m.weight)]
    else:
        raise Exception


def weight_recover(m):
    if isinstance(m, nn.Conv2d):
        weights_recover(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_recover(m)
    else:
        pass
